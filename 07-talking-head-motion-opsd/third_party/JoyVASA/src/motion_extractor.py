# coding: utf-8
import torch
torch.backends.cudnn.benchmark = True # disable CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR warning

import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)
import numpy as np
import os
import os.path as osp
import tyro
import subprocess
from rich.progress import track

from .config.argument_config import ArgumentConfig
from .config.inference_config import InferenceConfig
from .config.crop_config import CropConfig
from .utils.cropper import Cropper
from .utils.camera import get_rotation_matrix
from .utils.video import  get_fps
from .utils.io import load_video, dump
from .utils.helper import is_video, is_template, remove_suffix, is_square_video
from .utils.rprint import rlog as log
from .live_portrait_wmg_wrapper import LivePortraitWrapper


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False

def fast_check_args(args: ArgumentConfig):
    if not osp.exists(args.reference):
        raise FileNotFoundError(f"reference info not found: {args.reference}")
    if not osp.exists(args.driving):
        raise FileNotFoundError(f"driving info not found: {args.driving}")

def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

def make_abs_path(fn):
    return osp.join(osp.dirname(osp.realpath(__file__)), fn)

class LivePortraitMotionExtractor(object):

    def __init__(self, inference_cfg: InferenceConfig, crop_cfg: CropConfig):
        self.live_portrait_wrapper: LivePortraitWrapper = LivePortraitWrapper(inference_cfg=inference_cfg)
        self.cropper: Cropper = Cropper(crop_cfg=crop_cfg)

    def make_motion_template(self, I_lst, c_eyes_lst, c_lip_lst, **kwargs):
        n_frames = I_lst.shape[0]
        template_dct = {
            'n_frames': n_frames,
            'output_fps': kwargs.get('output_fps', 25),
            'motion': [],
            'c_eyes_lst': [],
            'c_lip_lst': [],
        }

        for i in track(range(n_frames), description='Making motion templates...', total=n_frames):
            # collect s, R, δ and t for inference
            I_i = I_lst[i]
            x_i_info = self.live_portrait_wrapper.get_kp_info(I_i)
            x_s = self.live_portrait_wrapper.transform_keypoint(x_i_info)
            R_i = get_rotation_matrix(x_i_info['pitch'], x_i_info['yaw'], x_i_info['roll'])

            item_dct = {
                'scale': x_i_info['scale'].cpu().numpy().astype(np.float32),
                'R': R_i.cpu().numpy().astype(np.float32),
                'exp': x_i_info['exp'].cpu().numpy().astype(np.float32),
                't': x_i_info['t'].cpu().numpy().astype(np.float32),
                'kp': x_i_info['kp'].cpu().numpy().astype(np.float32),
                'x_s': x_s.cpu().numpy().astype(np.float32),
                'pitch': x_i_info['pitch'].cpu().numpy().astype(np.float32),
                'yaw': x_i_info['yaw'].cpu().numpy().astype(np.float32),
                'roll': x_i_info['roll'].cpu().numpy().astype(np.float32)
            }

            template_dct['motion'].append(item_dct)

            c_eyes = c_eyes_lst[i].astype(np.float32)
            template_dct['c_eyes_lst'].append(c_eyes)

            c_lip = c_lip_lst[i].astype(np.float32)
            template_dct['c_lip_lst'].append(c_lip)

        return template_dct

    def execute(self, args, suffix=".pkl"):
        # for convenience
        inf_cfg = self.live_portrait_wrapper.inference_cfg

        ######## process driving info ########
        flag_load_from_template = is_template(args.driving)
        driving_rgb_crop_256x256_lst = None
        wfp_template = None
        wfp_template = remove_suffix(args.driving) + suffix
        if os.path.exists(wfp_template):
            log("motion generated ...")
            return

        if osp.exists(args.driving):
            if is_video(args.driving):
                flag_is_driving_video = True
                # load from video file, AND make motion template
                output_fps = int(get_fps(args.driving))
                log(f"Load driving video from: {args.driving}, FPS is {output_fps}")
                driving_rgb_lst = load_video(args.driving)
            else:
                raise Exception(f"{args.driving} is not a supported type!")

            ######## make motion template ########
            log("Start making driving motion template...")
            driving_n_frames = len(driving_rgb_lst)
            n_frames = driving_n_frames

            # 裁剪，resize
            if inf_cfg.flag_crop_driving_video or (not is_square_video(args.driving)):
                print("croping: ", inf_cfg.flag_crop_driving_video)
                ret_d = self.cropper.crop_driving_video(driving_rgb_lst)
                log(f'Driving video is cropped, {len(ret_d["frame_crop_lst"])} frames are processed.')
                if len(ret_d["frame_crop_lst"]) is not n_frames and flag_is_driving_video:
                    n_frames = min(n_frames, len(ret_d["frame_crop_lst"]))
                driving_rgb_crop_lst, driving_lmk_crop_lst = ret_d['frame_crop_lst'], ret_d['lmk_crop_lst']
                driving_rgb_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in driving_rgb_crop_lst]
            else:
                print("without crop ...")
                driving_lmk_crop_lst = self.cropper.calc_lmks_from_cropped_video(driving_rgb_lst)
                driving_rgb_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in driving_rgb_lst]  # force to resize to 256x256

            c_d_eyes_lst, c_d_lip_lst = self.live_portrait_wrapper.calc_ratio(driving_lmk_crop_lst)
            # save the motion template
            I_d_lst = self.live_portrait_wrapper.prepare_videos(driving_rgb_crop_256x256_lst)
            driving_template_dct = self.make_motion_template(I_d_lst, c_d_eyes_lst, c_d_lip_lst, output_fps=output_fps)

            wfp_template = remove_suffix(args.driving) + suffix
            dump(wfp_template, driving_template_dct)
            log(f"Dump motion template to {wfp_template}")
        else:
            raise Exception(f"{args.driving} does not exist!")


def make_motion_templete(args, driving_video, suffix=".pkl"):
    # configs
    args.driving = driving_video
    fast_check_args(args)
    inference_cfg = partial_fields(InferenceConfig, args.__dict__)
    crop_cfg = partial_fields(CropConfig, args.__dict__)

    # ffmpeg
    ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg")
    if osp.exists(ffmpeg_dir):
        os.environ["PATH"] += (os.pathsep + ffmpeg_dir)
    if not fast_check_ffmpeg():
        raise ImportError( "FFmpeg is not installed. Please install FFmpeg (including ffmpeg and ffprobe) before running this script. https://ffmpeg.org/download.html")

    try:
        # feature_extract
        motion_extractor = LivePortraitMotionExtractor(
            inference_cfg=inference_cfg,
            crop_cfg=crop_cfg
        )
        motion_extractor.execute(args, suffix=suffix)
    except Exception as e:
        print(f"Exception in motion extractor: {e}")