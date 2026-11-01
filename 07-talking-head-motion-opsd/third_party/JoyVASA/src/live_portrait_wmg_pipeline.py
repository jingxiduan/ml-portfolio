# coding: utf-8

"""
Pipeline of LivePortrait with Audio-Driven Motion Generation (Human)
"""

import torch
torch.backends.cudnn.benchmark = True # disable CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR warning

import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)
import os
import os.path as osp
 
from rich.progress import track

from .config.argument_config import ArgumentConfig
from .config.inference_config import InferenceConfig
from .config.crop_config import CropConfig
from .utils.cropper import Cropper
from .utils.camera import get_rotation_matrix
from .utils.video import images2video, add_audio_to_video
from .utils.crop import prepare_paste_back, paste_back
from .utils.io import load_image_rgb, resize_to_limit
from .utils.helper import mkdir, basename, dct2device, is_image, calc_motion_multiplier
from .utils.rprint import rlog as log
from .utils.viz import viz_lmk, plot_3d_scatter, plot_vectors, plot_vector_pairs
from .live_portrait_wmg_wrapper import LivePortraitWrapper

def make_abs_path(fn):
    return osp.join(osp.dirname(osp.realpath(__file__)), fn)

class LivePortraitPipeline(object):
    def __init__(self, inference_cfg: InferenceConfig, crop_cfg: CropConfig):
        self.live_portrait_wrapper: LivePortraitWrapper = LivePortraitWrapper(inference_cfg=inference_cfg)
        self.cropper: Cropper = Cropper(crop_cfg=crop_cfg)
    
    def execute(self, args: ArgumentConfig, ):
        inf_cfg = self.live_portrait_wrapper.inference_cfg
        device = self.live_portrait_wrapper.device
        crop_cfg = self.cropper.crop_cfg

        ######## load reference image ########
        if is_image(args.reference):
            img_rgb = load_image_rgb(args.reference)
            img_rgb = resize_to_limit(img_rgb, inf_cfg.source_max_dim, inf_cfg.source_division)
            log(f"Load reference image from {args.reference}")
            source_rgb_lst = [img_rgb]
        else:
            raise Exception(f"Unknown reference image format: {args.reference}")

        ######## generate motion sequence ########
        driving_template_dct = self.live_portrait_wrapper.gen_motion_sequence(args)
        n_frames = driving_template_dct['n_frames']

        ######## prepare for pasteback ########
        I_p_pstbk_lst = None
        if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
            I_p_pstbk_lst = []
            log("Prepared pasteback mask done.")
        I_p_lst = []
        R_d_0, x_d_0_info = None, None
        flag_normalize_lip = inf_cfg.flag_normalize_lip  # not overwrite
        flag_source_video_eye_retargeting = inf_cfg.flag_source_video_eye_retargeting  # not overwrite
        lip_delta_before_animation, eye_delta_before_animation = None, None

        ######## process source info ########
        if inf_cfg.flag_do_crop:
            crop_info = self.cropper.crop_source_image(source_rgb_lst[0], crop_cfg)
            if crop_info is None:
                raise Exception("No face detected in the source image!")
            source_lmk = crop_info['lmk_crop']
            img_crop_256x256 = crop_info['img_crop_256x256']
        else:
            source_lmk = self.cropper.calc_lmk_from_cropped_image(source_rgb_lst[0])
            img_crop_256x256 = cv2.resize(source_rgb_lst[0], (256, 256))  # force to resize to 256x256
        I_s = self.live_portrait_wrapper.prepare_source(img_crop_256x256)
        x_s_info = self.live_portrait_wrapper.get_kp_info(I_s)
        x_c_s = x_s_info['kp']
        R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
        f_s = self.live_portrait_wrapper.extract_feature_3d(I_s)
        x_s = self.live_portrait_wrapper.transform_keypoint(x_s_info)

        # let lip-open scalar to be 0 at first
        if flag_normalize_lip and inf_cfg.flag_relative_motion and source_lmk is not None:
            c_d_lip_before_animation = [0.]
            combined_lip_ratio_tensor_before_animation = self.live_portrait_wrapper.calc_combined_lip_ratio(c_d_lip_before_animation, source_lmk)
            if combined_lip_ratio_tensor_before_animation[0][0] >= inf_cfg.lip_normalize_threshold:
                lip_delta_before_animation = self.live_portrait_wrapper.retarget_lip(x_s, combined_lip_ratio_tensor_before_animation)

        if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
            mask_ori_float = prepare_paste_back(inf_cfg.mask_crop, crop_info['M_c2o'], dsize=(source_rgb_lst[0].shape[1], source_rgb_lst[0].shape[0]))

        ######## animate ########
        for i in track(range(n_frames), description='🚀Animating Image with Generated Motions...', total=n_frames):
            x_d_i_info = driving_template_dct['motion'][i]
            x_d_i_info = dct2device(x_d_i_info, device)
            
            # R
            R_d_i = x_d_i_info['R'] if 'R' in x_d_i_info.keys() else x_d_i_info['R_d']  # compatible with previous keys
            if i == 0:  # cache the first frame
                R_d_0 = R_d_i
                x_d_0_info = x_d_i_info.copy()
            if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                R_new = (R_d_i @ R_d_0.permute(0, 2, 1)) @ R_s
            else:
                R_new = R_s

            # delta
            delta_new = x_s_info['exp'].clone()
            if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "exp":
                # 相对
                delta_new = x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp'])
                for lip_idx in [6, 12, 14, 17, 19, 20]:
                    delta_new[:, lip_idx, :] = x_d_i_info['exp'][:, lip_idx, :]
                # # 绝对
                # delta_new = x_s_info['exp'].clone()
                # for idx in [1,2,6,11,12,13,14,15,16,17,18,19,20]:
                #     delta_new[:, idx, :] = x_d_i_info['exp'][:, idx, :]
                # delta_new[:, 3:5, 1] =  x_d_i_info['exp'][:, 3:5, 1]
                # delta_new[:, 5, 2] =  x_d_i_info['exp'][:, 5, 2]
                # delta_new[:, 8, 2] =  x_d_i_info['exp'][:, 8, 2]
                # delta_new[:, 9, 1:] = x_d_i_info['exp'][:, 9, 1:]
            elif inf_cfg.animation_region == "lip":
                for lip_idx in [6, 12, 14, 17, 19, 20]:
                    delta_new[:, lip_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp']))[:, lip_idx, :]
            # scale
            scale_new = x_s_info['scale'] * (x_d_i_info['scale'] / x_d_0_info['scale'])

            # translation
            t_new = x_s_info['t'] + (x_d_i_info['t'] - x_d_0_info['t'])
            # t_new = x_s_info['t']
            t_new[..., 2].fill_(0)  # zero tz

            x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new

            if inf_cfg.flag_relative_motion and inf_cfg.driving_option == "expression-friendly":
                if i == 0:
                    x_d_0_new = x_d_i_new
                    motion_multiplier = calc_motion_multiplier(x_s, x_d_0_new)
                x_d_diff = (x_d_i_new - x_d_0_new) * motion_multiplier
                x_d_i_new = x_d_diff + x_s

            # Algorithm 1 in Liveportrait:
            if not inf_cfg.flag_stitching and not inf_cfg.flag_eye_retargeting and not inf_cfg.flag_lip_retargeting:
                # without stitching or retargeting
                if flag_normalize_lip and lip_delta_before_animation is not None:
                    x_d_i_new += lip_delta_before_animation
                if flag_source_video_eye_retargeting and eye_delta_before_animation is not None:
                    x_d_i_new += eye_delta_before_animation
                else:
                    pass
            elif inf_cfg.flag_stitching and not inf_cfg.flag_eye_retargeting and not inf_cfg.flag_lip_retargeting:
                # with stitching and without retargeting
                if flag_normalize_lip and lip_delta_before_animation is not None:
                    x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new) + lip_delta_before_animation
                else:
                    x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new)
                if flag_source_video_eye_retargeting and eye_delta_before_animation is not None:
                    x_d_i_new += eye_delta_before_animation
            else:
                eyes_delta, lip_delta = None, None
                if inf_cfg.flag_relative_motion:  # use x_s
                    x_d_i_new = x_s + \
                        (eyes_delta if eyes_delta is not None else 0) + \
                        (lip_delta if lip_delta is not None else 0)
                else:  # use x_d,i
                    x_d_i_new = x_d_i_new + \
                        (eyes_delta if eyes_delta is not None else 0) + \
                        (lip_delta if lip_delta is not None else 0)

                if inf_cfg.flag_stitching:
                    x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new)

            x_d_i_new = x_s + (x_d_i_new - x_s) * inf_cfg.driving_multiplier
            out = self.live_portrait_wrapper.warp_decode(f_s, x_s, x_d_i_new)
            I_p_i = self.live_portrait_wrapper.parse_output(out['out'])[0]
            I_p_lst.append(I_p_i)

            if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
                # TODO: the paste back procedure is slow, considering optimize it using multi-threading or GPU
                I_p_pstbk = paste_back(I_p_i, crop_info['M_c2o'], source_rgb_lst[0], mask_ori_float)
                I_p_pstbk_lst.append(I_p_pstbk)

        # save the animated result
        mkdir(args.output_dir)
        temp_video = osp.join(args.output_dir, f'{basename(args.reference)}_{basename(args.audio)}_temp.mp4')
        if I_p_pstbk_lst is not None and len(I_p_pstbk_lst) > 0:
            images2video(I_p_pstbk_lst, wfp=temp_video, fps=inf_cfg.output_fps)
        else:
            images2video(I_p_lst, wfp=temp_video, fps=inf_cfg.output_fps)
        final_video = osp.join(args.output_dir, f'{basename(args.reference)}_{basename(args.audio)}.mp4')
        add_audio_to_video(temp_video, args.audio, final_video)
        return final_video