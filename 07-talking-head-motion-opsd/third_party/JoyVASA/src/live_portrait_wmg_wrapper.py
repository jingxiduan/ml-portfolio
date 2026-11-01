# coding: utf-8

"""
Wrappers for LivePortrait core functions
"""

import contextlib
import os.path as osp
import os
import pickle
import numpy as np
import cv2
import torch
import yaml
import math
import librosa
import torch.nn.functional as F
from rich.progress import track

from .utils.timer import Timer
from .utils.helper import load_model, concat_feat
from .utils.camera import headpose_pred_to_degree, get_rotation_matrix
from .utils.retargeting_utils import calc_eye_close_ratio, calc_lip_close_ratio
from .config.inference_config import InferenceConfig
from .utils.rprint import rlog as log
from .utils.filter import smooth_


class LivePortraitWrapper(object):
    """
    Wrapper for Human
    """

    def __init__(self, inference_cfg: InferenceConfig):

        self.inference_cfg = inference_cfg
        self.device_id = inference_cfg.device_id
        self.compile = inference_cfg.flag_do_torch_compile
        if inference_cfg.flag_force_cpu:
            self.device = 'cpu'
        else:
            try:
                if torch.backends.mps.is_available():
                    self.device = 'mps'
                else:
                    self.device = 'cuda:' + str(self.device_id)
            except:
                self.device = 'cuda:' + str(self.device_id)
        


        model_config = yaml.load(open(inference_cfg.models_config, 'r'), Loader=yaml.SafeLoader)
        # init F
        self.appearance_feature_extractor = load_model(inference_cfg.checkpoint_F, model_config, self.device, 'appearance_feature_extractor')
        log(f'Load appearance_feature_extractor from {osp.realpath(inference_cfg.checkpoint_F)} done.')
        # init M
        self.motion_extractor = load_model(inference_cfg.checkpoint_M, model_config, self.device, 'motion_extractor')
        log(f'Load motion_extractor from {osp.realpath(inference_cfg.checkpoint_M)} done.')
        # init W
        self.warping_module = load_model(inference_cfg.checkpoint_W, model_config, self.device, 'warping_module')
        log(f'Load warping_module from {osp.realpath(inference_cfg.checkpoint_W)} done.')
        # init G
        self.spade_generator = load_model(inference_cfg.checkpoint_G, model_config, self.device, 'spade_generator')
        log(f'Load spade_generator from {osp.realpath(inference_cfg.checkpoint_G)} done.')
        # init S and R
        if inference_cfg.checkpoint_S is not None and osp.exists(inference_cfg.checkpoint_S):
            self.stitching_retargeting_module = load_model(inference_cfg.checkpoint_S, model_config, self.device, 'stitching_retargeting_module')
            log(f'Load stitching_retargeting_module from {osp.realpath(inference_cfg.checkpoint_S)} done.')
        else:
            self.stitching_retargeting_module = None
        # Optimize for inference
        if self.compile:
            torch._dynamo.config.suppress_errors = True  # Suppress errors and fall back to eager execution
            self.warping_module = torch.compile(self.warping_module, mode='max-autotune')
            self.spade_generator = torch.compile(self.spade_generator, mode='max-autotune')

        self.model_config = model_config
        self.timer = Timer()

        # Motion Genertor
        self.motion_generator, self.motion_generator_args = load_model(inference_cfg.checkpoint_MotionGenerator, model_config, self.device, 'motion_generator')
        log(f'Load motion_generator from {osp.realpath(inference_cfg.checkpoint_MotionGenerator)} done.')
        self.n_motions = self.motion_generator_args.n_motions
        self.n_prev_motions = self.motion_generator_args.n_prev_motions
        self.fps = self.motion_generator_args.fps
        self.audio_unit = 16000. / self.fps  # num of samples per frame
        self.n_audio_samples = round(self.audio_unit * self.n_motions)
        self.pad_mode = self.motion_generator_args.pad_mode
        self.use_indicator = self.motion_generator_args.use_indicator
        self.templete_dict = pickle.load(open(inference_cfg.motion_template_path, 'rb'))


    def inference_ctx(self):
        if self.device == "mps":
            ctx = contextlib.nullcontext()
        else:
            ctx = torch.autocast(device_type=self.device[:4], dtype=torch.float16,
                                 enabled=self.inference_cfg.flag_use_half_precision)
        return ctx

    def update_config(self, user_args):
        for k, v in user_args.items():
            if hasattr(self.inference_cfg, k):
                setattr(self.inference_cfg, k, v)

    def prepare_source(self, img: np.ndarray) -> torch.Tensor:
        """ construct the input as standard
        img: HxWx3, uint8, 256x256
        """
        h, w = img.shape[:2]
        if h != self.inference_cfg.input_shape[0] or w != self.inference_cfg.input_shape[1]:
            x = cv2.resize(img, (self.inference_cfg.input_shape[0], self.inference_cfg.input_shape[1]))
        else:
            x = img.copy()

        if x.ndim == 3:
            x = x[np.newaxis].astype(np.float32) / 255.  # HxWx3 -> 1xHxWx3, normalized to 0~1
        elif x.ndim == 4:
            x = x.astype(np.float32) / 255.  # BxHxWx3, normalized to 0~1
        else:
            raise ValueError(f'img ndim should be 3 or 4: {x.ndim}')
        x = np.clip(x, 0, 1)  # clip to 0~1
        x = torch.from_numpy(x).permute(0, 3, 1, 2)  # 1xHxWx3 -> 1x3xHxW
        x = x.to(self.device)
        return x

    def prepare_videos(self, imgs) -> torch.Tensor:
        """ construct the input as standard
        imgs: NxBxHxWx3, uint8
        """
        if isinstance(imgs, list):
            _imgs = np.array(imgs)[..., np.newaxis]  # TxHxWx3x1
        elif isinstance(imgs, np.ndarray):
            _imgs = imgs
        else:
            raise ValueError(f'imgs type error: {type(imgs)}')

        y = _imgs.astype(np.float32) / 255.
        y = np.clip(y, 0, 1)  # clip to 0~1
        y = torch.from_numpy(y).permute(0, 4, 3, 1, 2)  # TxHxWx3x1 -> Tx1x3xHxW
        y = y.to(self.device)

        return y

    def extract_feature_3d(self, x: torch.Tensor) -> torch.Tensor:
        """ get the appearance feature of the image by F
        x: Bx3xHxW, normalized to 0~1
        """
        with torch.no_grad(), self.inference_ctx():
            feature_3d = self.appearance_feature_extractor(x)

        return feature_3d.float()

    def get_kp_info(self, x: torch.Tensor, **kwargs) -> dict:
        """ get the implicit keypoint information
        x: Bx3xHxW, normalized to 0~1
        flag_refine_info: whether to trandform the pose to degrees and the dimention of the reshape
        return: A dict contains keys: 'pitch', 'yaw', 'roll', 't', 'exp', 'scale', 'kp'
        """
        with torch.no_grad(), self.inference_ctx():
            kp_info = self.motion_extractor(x)

            if self.inference_cfg.flag_use_half_precision:
                # float the dict
                for k, v in kp_info.items():
                    if isinstance(v, torch.Tensor):
                        kp_info[k] = v.float()

        flag_refine_info: bool = kwargs.get('flag_refine_info', True)
        if flag_refine_info:
            bs = kp_info['kp'].shape[0]
            kp_info['pitch'] = headpose_pred_to_degree(kp_info['pitch'])[:, None]  # Bx1
            kp_info['yaw'] = headpose_pred_to_degree(kp_info['yaw'])[:, None]  # Bx1
            kp_info['roll'] = headpose_pred_to_degree(kp_info['roll'])[:, None]  # Bx1
            kp_info['kp'] = kp_info['kp'].reshape(bs, -1, 3)  # BxNx3
            kp_info['exp'] = kp_info['exp'].reshape(bs, -1, 3)  # BxNx3

        return kp_info

    def get_pose_dct(self, kp_info: dict) -> dict:
        pose_dct = dict(
            pitch=headpose_pred_to_degree(kp_info['pitch']).item(),
            yaw=headpose_pred_to_degree(kp_info['yaw']).item(),
            roll=headpose_pred_to_degree(kp_info['roll']).item(),
        )
        return pose_dct

    def get_fs_and_kp_info(self, source_prepared, driving_first_frame):

        # get the canonical keypoints of source image by M
        source_kp_info = self.get_kp_info(source_prepared, flag_refine_info=True)
        source_rotation = get_rotation_matrix(source_kp_info['pitch'], source_kp_info['yaw'], source_kp_info['roll'])

        # get the canonical keypoints of first driving frame by M
        driving_first_frame_kp_info = self.get_kp_info(driving_first_frame, flag_refine_info=True)
        driving_first_frame_rotation = get_rotation_matrix(
            driving_first_frame_kp_info['pitch'],
            driving_first_frame_kp_info['yaw'],
            driving_first_frame_kp_info['roll']
        )

        # get feature volume by F
        source_feature_3d = self.extract_feature_3d(source_prepared)

        return source_kp_info, source_rotation, source_feature_3d, driving_first_frame_kp_info, driving_first_frame_rotation

    def transform_keypoint(self, kp_info: dict):
        """
        transform the implicit keypoints with the pose, shift, and expression deformation
        kp: BxNx3
        """
        kp = kp_info['kp']    # (bs, k, 3)
        pitch, yaw, roll = kp_info['pitch'], kp_info['yaw'], kp_info['roll']

        t, exp = kp_info['t'], kp_info['exp']
        scale = kp_info['scale']

        pitch = headpose_pred_to_degree(pitch)
        yaw = headpose_pred_to_degree(yaw)
        roll = headpose_pred_to_degree(roll)

        bs = kp.shape[0]
        if kp.ndim == 2:
            num_kp = kp.shape[1] // 3  # Bx(num_kpx3)
        else:
            num_kp = kp.shape[1]  # Bxnum_kpx3

        rot_mat = get_rotation_matrix(pitch, yaw, roll)    # (bs, 3, 3)

        # Eqn.2: s * (R * x_c,s + exp) + t
        kp_transformed = kp.view(bs, num_kp, 3) @ rot_mat + exp.view(bs, num_kp, 3)
        kp_transformed *= scale[..., None]  # (bs, k, 3) * (bs, 1, 1) = (bs, k, 3)
        kp_transformed[:, :, 0:2] += t[:, None, 0:2]  # remove z, only apply tx ty

        return kp_transformed

    def retarget_eye(self, kp_source: torch.Tensor, eye_close_ratio: torch.Tensor) -> torch.Tensor:
        """
        kp_source: BxNx3
        eye_close_ratio: Bx3
        Return: Bx(3*num_kp)
        """
        feat_eye = concat_feat(kp_source, eye_close_ratio)

        with torch.no_grad():
            delta = self.stitching_retargeting_module['eye'](feat_eye)

        return delta.reshape(-1, kp_source.shape[1], 3)

    def retarget_lip(self, kp_source: torch.Tensor, lip_close_ratio: torch.Tensor) -> torch.Tensor:
        """
        kp_source: BxNx3
        lip_close_ratio: Bx2
        Return: Bx(3*num_kp)
        """
        feat_lip = concat_feat(kp_source, lip_close_ratio)

        with torch.no_grad():
            delta = self.stitching_retargeting_module['lip'](feat_lip)

        return delta.reshape(-1, kp_source.shape[1], 3)

    def stitch(self, kp_source: torch.Tensor, kp_driving: torch.Tensor) -> torch.Tensor:
        """
        kp_source: BxNx3
        kp_driving: BxNx3
        Return: Bx(3*num_kp+2)
        """
        feat_stiching = concat_feat(kp_source, kp_driving)

        with torch.no_grad():
            delta = self.stitching_retargeting_module['stitching'](feat_stiching)

        return delta

    def stitching(self, kp_source: torch.Tensor, kp_driving: torch.Tensor) -> torch.Tensor:
        """ conduct the stitching
        kp_source: Bxnum_kpx3
        kp_driving: Bxnum_kpx3
        """

        if self.stitching_retargeting_module is not None:

            bs, num_kp = kp_source.shape[:2]

            kp_driving_new = kp_driving.clone()
            delta = self.stitch(kp_source, kp_driving_new)

            delta_exp = delta[..., :3*num_kp].reshape(bs, num_kp, 3)  # 1x20x3
            delta_tx_ty = delta[..., 3*num_kp:3*num_kp+2].reshape(bs, 1, 2)  # 1x1x2

            kp_driving_new += delta_exp
            kp_driving_new[..., :2] += delta_tx_ty

            return kp_driving_new

        return kp_driving

    def warp_decode(self, feature_3d: torch.Tensor, kp_source: torch.Tensor, kp_driving: torch.Tensor) -> torch.Tensor:
        """ get the image after the warping of the implicit keypoints
        feature_3d: Bx32x16x64x64, feature volume
        kp_source: BxNx3
        kp_driving: BxNx3
        """
        # The line 18 in Algorithm 1: D(W(f_s; x_s, x′_d,i)）
        with torch.no_grad(), self.inference_ctx():
            if self.compile:
                # Mark the beginning of a new CUDA Graph step
                torch.compiler.cudagraph_mark_step_begin()
            # get decoder input
            ret_dct = self.warping_module(feature_3d, kp_source=kp_source, kp_driving=kp_driving)
            # decode
            ret_dct['out'] = self.spade_generator(feature=ret_dct['out'])

            # float the dict
            if self.inference_cfg.flag_use_half_precision:
                for k, v in ret_dct.items():
                    if isinstance(v, torch.Tensor):
                        ret_dct[k] = v.float()

        return ret_dct

    def parse_output(self, out: torch.Tensor) -> np.ndarray:
        """ construct the output as standard
        return: 1xHxWx3, uint8
        """
        out = np.transpose(out.data.cpu().numpy(), [0, 2, 3, 1])  # 1x3xHxW -> 1xHxWx3
        out = np.clip(out, 0, 1)  # clip to 0~1
        out = np.clip(out * 255, 0, 255).astype(np.uint8)  # 0~1 -> 0~255

        return out

    def calc_ratio(self, lmk_lst):
        input_eye_ratio_lst = []
        input_lip_ratio_lst = []
        for lmk in lmk_lst:
            # for eyes retargeting
            input_eye_ratio_lst.append(calc_eye_close_ratio(lmk[None]))
            # for lip retargeting
            input_lip_ratio_lst.append(calc_lip_close_ratio(lmk[None]))
        return input_eye_ratio_lst, input_lip_ratio_lst

    def calc_combined_eye_ratio(self, c_d_eyes_i, source_lmk):
        c_s_eyes = calc_eye_close_ratio(source_lmk[None])
        c_s_eyes_tensor = torch.from_numpy(c_s_eyes).float().to(self.device)
        c_d_eyes_i_tensor = torch.Tensor([c_d_eyes_i[0][0]]).reshape(1, 1).to(self.device)
        # [c_s,eyes, c_d,eyes,i]
        combined_eye_ratio_tensor = torch.cat([c_s_eyes_tensor, c_d_eyes_i_tensor], dim=1)
        return combined_eye_ratio_tensor

    def calc_combined_lip_ratio(self, c_d_lip_i, source_lmk):
        c_s_lip = calc_lip_close_ratio(source_lmk[None])
        c_s_lip_tensor = torch.from_numpy(c_s_lip).float().to(self.device)
        c_d_lip_i_tensor = torch.Tensor([c_d_lip_i[0]]).to(self.device).reshape(1, 1) # 1x1
        # [c_s,lip, c_d,lip,i]
        combined_lip_ratio_tensor = torch.cat([c_s_lip_tensor, c_d_lip_i_tensor], dim=1) # 1x2
        return combined_lip_ratio_tensor
    
    def gen_motion_sequence(self, args):
       # preprocess audio
        log(f"start loading audio from {args.audio}")
        audio, _ = librosa.load(args.audio, sr=16000, mono=True)
        log(f"audio loaded! {audio.shape}")
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).to(self.device)
        assert audio.ndim == 1, 'Audio must be 1D tensor.'
        log(f"loading audio from: {args.audio}")
        audio = F.pad(audio, (1280, 640), "constant", 0)


        # crop audio into n_subdivision according to n_motions 
        clip_len = int(len(audio) / 16000 * self.fps)
        stride = self.n_motions
        if clip_len <= self.n_motions:
            n_subdivision = 1
        else:
            n_subdivision = math.ceil(clip_len / stride)

        # padding
        n_padding_audio_samples = self.n_audio_samples * n_subdivision - len(audio)
        n_padding_frames = math.ceil(n_padding_audio_samples / self.audio_unit)
        if n_padding_audio_samples > 0:
            if self.pad_mode == 'zero':
                padding_value = 0
            elif self.pad_mode == 'replicate':
                padding_value = audio[-1]
            else:
                raise ValueError(f'Unknown pad mode: {self.pad_mode}')
            audio = F.pad(audio, (0, n_padding_audio_samples), value=padding_value)

        # generate motions
        coef_list = []
        for i in range(0, n_subdivision):
            start_idx = i * stride
            end_idx = start_idx + self.n_motions
            indicator = torch.ones((1, self.n_motions)).to(self.device) if self.use_indicator else None
            if indicator is not None and i == n_subdivision - 1 and n_padding_frames > 0:
                indicator[:, -n_padding_frames:] = 0
            audio_in = audio[round(start_idx * self.audio_unit):round(end_idx * self.audio_unit)].unsqueeze(0)

            if i == 0:
                motion_feat, noise, prev_audio_feat = self.motion_generator.sample(audio_in,
                                                                        indicator=indicator, cfg_mode=args.cfg_mode,
                                                                        cfg_cond=args.cfg_cond, cfg_scale=args.cfg_scale,
                                                                        dynamic_threshold=0)
            else:
                motion_feat, noise, prev_audio_feat = self.motion_generator.sample(audio_in,
                                                                        prev_motion_feat, prev_audio_feat, noise,
                                                                        indicator=indicator, cfg_mode=args.cfg_mode,
                                                                        cfg_cond=args.cfg_cond, cfg_scale=args.cfg_scale,
                                                                        dynamic_threshold=0)
            prev_motion_feat = motion_feat[:, -self.n_prev_motions:].clone()
            prev_audio_feat = prev_audio_feat[:, -self.n_prev_motions:]

            motion_coef = motion_feat
            if i == n_subdivision - 1 and n_padding_frames > 0:
                motion_coef = motion_coef[:, :-n_padding_frames]  # delete padded frames
            coef_list.append(motion_coef)
            motion_coef = torch.cat(coef_list, dim=1)
            # motion_coef = self.reformat_motion(args, motion_coef)

        motion_coef = motion_coef.squeeze() #.cpu().numpy().astype(np.float32)
        motion_list = []
        for idx in track(range(motion_coef.shape[0]), description='🚀Generating Motion Sequence...', total=motion_coef.shape[0]):
            exp = motion_coef[idx][:63].cpu() * self.templete_dict["std_exp"] + self.templete_dict["mean_exp"]
            scale = motion_coef[idx][63:64].cpu() * (self.templete_dict["max_scale"] - self.templete_dict["min_scale"]) + self.templete_dict["min_scale"]
            t = motion_coef[idx][64:67].cpu() * (self.templete_dict["max_t"] - self.templete_dict["min_t"]) + self.templete_dict["min_t"]
            pitch = motion_coef[idx][67:68].cpu() * (self.templete_dict["max_pitch"] - self.templete_dict["min_pitch"]) + self.templete_dict["min_pitch"]
            yaw = motion_coef[idx][68:69].cpu() * (self.templete_dict["max_yaw"] - self.templete_dict["min_yaw"]) + self.templete_dict["min_yaw"]
            roll = motion_coef[idx][69:70].cpu() * (self.templete_dict["max_roll"] - self.templete_dict["min_roll"]) + self.templete_dict["min_roll"]

            R = get_rotation_matrix(pitch, yaw, roll)
            R = R.reshape(1, 3, 3).cpu().numpy().astype(np.float32)
            
            exp = exp.reshape(1, 21, 3).cpu().numpy().astype(np.float32)
            scale = scale.reshape(1, 1).cpu().numpy().astype(np.float32)
            t = t.reshape(1, 3).cpu().numpy().astype(np.float32)
            pitch = pitch.reshape(1, 1).cpu().numpy().astype(np.float32)
            yaw = yaw.reshape(1, 1).cpu().numpy().astype(np.float32)
            roll = roll.reshape(1, 1).cpu().numpy().astype(np.float32)
            
            motion_list.append({"exp": exp, "scale": scale, "R": R, "t": t, "pitch": pitch, "yaw": yaw, "roll": roll})
        tgt_motion = {'n_frames': motion_coef.shape[0], 'output_fps': 25, 'motion': motion_list}

        if args.is_smooth_motion:
            tgt_motion = smooth_(tgt_motion, method="ema")
        return tgt_motion
    

class LivePortraitWrapperAnimal(LivePortraitWrapper):
    """
    Wrapper for Animal
    """
    def __init__(self, inference_cfg: InferenceConfig):
        # super().__init__(inference_cfg)  # 调用父类的初始化方法

        self.inference_cfg = inference_cfg
        self.device_id = inference_cfg.device_id
        self.compile = inference_cfg.flag_do_torch_compile
        if inference_cfg.flag_force_cpu:
            self.device = 'cpu'
        else:
            try: 
                if torch.backends.mps.is_available():
                    self.device = 'mps'
                else:
                    self.device = 'cuda:' + str(self.device_id)
            except:
                    self.device = 'cuda:' + str(self.device_id)

        model_config = yaml.load(open(inference_cfg.models_config, 'r'), Loader=yaml.SafeLoader)
        # init F
        self.appearance_feature_extractor = load_model(inference_cfg.checkpoint_F_animal, model_config, self.device, 'appearance_feature_extractor')
        log(f'Load appearance_feature_extractor from {osp.realpath(inference_cfg.checkpoint_F_animal)} done.')
        # init M
        self.motion_extractor = load_model(inference_cfg.checkpoint_M_animal, model_config, self.device, 'motion_extractor')
        log(f'Load motion_extractor from {osp.realpath(inference_cfg.checkpoint_M_animal)} done.')
        # init W
        self.warping_module = load_model(inference_cfg.checkpoint_W_animal, model_config, self.device, 'warping_module')
        log(f'Load warping_module from {osp.realpath(inference_cfg.checkpoint_W_animal)} done.')
        # init G
        self.spade_generator = load_model(inference_cfg.checkpoint_G_animal, model_config, self.device, 'spade_generator')
        log(f'Load spade_generator from {osp.realpath(inference_cfg.checkpoint_G_animal)} done.')
        # init S and R
        if inference_cfg.checkpoint_S_animal is not None and osp.exists(inference_cfg.checkpoint_S_animal):
            self.stitching_retargeting_module = load_model(inference_cfg.checkpoint_S_animal, model_config, self.device, 'stitching_retargeting_module')
            log(f'Load stitching_retargeting_module from {osp.realpath(inference_cfg.checkpoint_S_animal)} done.')
        else:
            self.stitching_retargeting_module = None

        # Optimize for inference
        if self.compile:
            torch._dynamo.config.suppress_errors = True  # Suppress errors and fall back to eager execution
            self.warping_module = torch.compile(self.warping_module, mode='max-autotune')
            self.spade_generator = torch.compile(self.spade_generator, mode='max-autotune')

        self.timer = Timer()

        # Motion Genertor
        self.motion_generator, self.motion_generator_args = load_model(inference_cfg.checkpoint_MotionGenerator, model_config, self.device, 'motion_generator')
        log(f'Load motion_generator from {osp.realpath(inference_cfg.checkpoint_MotionGenerator)} done.')
        self.n_motions = self.motion_generator_args.n_motions
        self.n_prev_motions = self.motion_generator_args.n_prev_motions
        self.fps = self.motion_generator_args.fps
        self.audio_unit = 16000. / self.fps  # num of samples per frame
        self.n_audio_samples = round(self.audio_unit * self.n_motions)
        self.pad_mode = self.motion_generator_args.pad_mode
        self.use_indicator = self.motion_generator_args.use_indicator
        self.templete_dict = pickle.load(open(inference_cfg.motion_template_path, 'rb'))
