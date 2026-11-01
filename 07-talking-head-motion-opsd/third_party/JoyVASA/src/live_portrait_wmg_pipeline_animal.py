# coding: utf-8

"""
Pipeline of LivePortrait (Animal)
"""

import warnings
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.")
warnings.filterwarnings("ignore", message="torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly.")
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True. Gradients will be None")

import torch
torch.backends.cudnn.benchmark = True # disable CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR warning

import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)
import numpy as np
import os
import os.path as osp
from rich.progress import track

from .config.argument_config import ArgumentConfig
from .config.inference_config import InferenceConfig
from .config.crop_config import CropConfig
from .utils.cropper import Cropper
from .utils.camera import get_rotation_matrix
from .utils.video import images2video, concat_frames, get_fps, add_audio_to_video, has_audio_stream, video2gif
from .utils.crop import _transform_img, prepare_paste_back, paste_back
from .utils.io import load_image_rgb, load_video, resize_to_limit, dump, load
from .utils.helper import mkdir, basename, dct2device, is_video, is_template, remove_suffix, is_image, calc_motion_multiplier
from .utils.rprint import rlog as log
from .live_portrait_wmg_wrapper import LivePortraitWrapperAnimal


def make_abs_path(fn):
    return osp.join(osp.dirname(osp.realpath(__file__)), fn)

class LivePortraitPipelineAnimal(object):

    def __init__(self, inference_cfg: InferenceConfig, crop_cfg: CropConfig):
        self.live_portrait_wrapper_animal: LivePortraitWrapperAnimal = LivePortraitWrapperAnimal(inference_cfg=inference_cfg)
        self.cropper: Cropper = Cropper(crop_cfg=crop_cfg, image_type='animal_face', flag_use_half_precision=inference_cfg.flag_use_half_precision)

    def execute(self, args: ArgumentConfig):
        inf_cfg = self.live_portrait_wrapper_animal.inference_cfg
        device = self.live_portrait_wrapper_animal.device
        crop_cfg = self.cropper.crop_cfg

        ######## load reference image ########
        if is_image(args.reference):
            img_rgb = load_image_rgb(args.reference)
            img_rgb = resize_to_limit(img_rgb, inf_cfg.source_max_dim, inf_cfg.source_division)
            log(f"Load reference image from {args.reference}")
        else:
            raise Exception(f"Unknown reference image format: {args.reference}")

        ######## generate motion sequence ########
        driving_template_dct = self.live_portrait_wrapper_animal.gen_motion_sequence(args)
        n_frames = driving_template_dct['n_frames']

        ######## prepare for pasteback ########
        I_p_pstbk_lst = None
        if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
            I_p_pstbk_lst = []
            log("Prepared pasteback mask done.")

        ######## process source info ########
        if inf_cfg.flag_do_crop:
            crop_info = self.cropper.crop_source_image(img_rgb, crop_cfg)
            if crop_info is None:
                raise Exception("No animal face detected in the source image!")
            img_crop_256x256 = crop_info['img_crop_256x256']
        else:
            img_crop_256x256 = cv2.resize(img_rgb, (256, 256))  # force to resize to 256x256
        I_s = self.live_portrait_wrapper_animal.prepare_source(img_crop_256x256)
        x_s_info = self.live_portrait_wrapper_animal.get_kp_info(I_s)
        x_c_s = x_s_info['kp']
        # print("x_c_s.shape: ", x_c_s.shape)
        R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
        f_s = self.live_portrait_wrapper_animal.extract_feature_3d(I_s)
        x_s = self.live_portrait_wrapper_animal.transform_keypoint(x_s_info)

        if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
            mask_ori_float = prepare_paste_back(inf_cfg.mask_crop, crop_info['M_c2o'], dsize=(img_rgb.shape[1], img_rgb.shape[0]))

        ######## animate ########
        I_p_lst = []
        for i in track(range(n_frames), description='🚀Animating Image with Generated Motions...', total=n_frames):
            x_d_i_info = driving_template_dct['motion'][i]
            x_d_i_info = dct2device(x_d_i_info, device)

            R_d_i = x_d_i_info['R'] if 'R' in x_d_i_info.keys() else x_d_i_info['R_d']  # compatible with previous keys
            if i == 0:  # cache the first frame
                R_d_0 = R_d_i
                x_d_0_info = x_d_i_info
            R_d_i = (R_d_i @ R_d_0.permute(0, 2, 1)) @ R_s

            # expression
            # 原始
            # delta_new = x_d_i_info['exp']
            delta_new = x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp'])
            # 绝对
            # delta_new = x_s_info['exp'].clone() # 源图像的exp信息，这里delta_new用于保存源图像变化后的exp
            # for idx in [1,2,6,11,12,13,14,15,16,17,18,19,20]:
            #     delta_new[:, idx, :] = x_d_i_info['exp'][:, idx, :]
            # delta_new[:, 3:5, 1] =  x_d_i_info['exp'][:, 3:5, 1]
            # delta_new[:, 5, 2] =  x_d_i_info['exp'][:, 5, 2]
            # delta_new[:, 8, 2] =  x_d_i_info['exp'][:, 8, 2]
            # delta_new[:, 9, 1:] = x_d_i_info['exp'][:, 9, 1:]

            # scale
            scale_new = x_s_info['scale']

            # translation
            # 绝对
            # t_new = x_d_i_info['t']
            # 相对
            t_new = x_s_info['t'] + (x_d_i_info['t'] - x_d_0_info['t'])
            t_new[..., 2].fill_(0)  # zero tz

            # 公式二
            x_d_i = scale_new * (x_c_s @ R_d_i + delta_new) + t_new

            if i == 0:
                x_d_0 = x_d_i
                motion_multiplier = calc_motion_multiplier(x_s, x_d_0)

            x_d_diff = (x_d_i - x_d_0) * motion_multiplier
            x_d_i = x_d_diff + x_s

            if not inf_cfg.flag_stitching:
                pass
            else:
                x_d_i = self.live_portrait_wrapper_animal.stitching(x_s, x_d_i)

            x_d_i = x_s + (x_d_i - x_s) * inf_cfg.driving_multiplier
            out = self.live_portrait_wrapper_animal.warp_decode(f_s, x_s, x_d_i)
            I_p_i = self.live_portrait_wrapper_animal.parse_output(out['out'])[0]
            I_p_lst.append(I_p_i)

            if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
                I_p_pstbk = paste_back(I_p_i, crop_info['M_c2o'], img_rgb, mask_ori_float)
                I_p_pstbk_lst.append(I_p_pstbk)

        # save the animated result
        if not os.path.exists(args.output_dir):
            mkdir(args.output_dir)
        temp_video = osp.join(args.output_dir, f'{basename(args.reference)}_{basename(args.audio)}_temp.mp4')
        if I_p_pstbk_lst is not None and len(I_p_pstbk_lst) > 0:
            images2video(I_p_pstbk_lst, wfp=temp_video, fps=inf_cfg.output_fps)
        else:
            images2video(I_p_lst, wfp=temp_video, fps=inf_cfg.output_fps)
        final_video = osp.join(args.output_dir, f'{basename(args.reference)}_{basename(args.audio)}.mp4')
        add_audio_to_video(temp_video, args.audio, final_video)
        return final_video