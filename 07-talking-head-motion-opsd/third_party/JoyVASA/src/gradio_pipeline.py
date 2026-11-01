# coding: utf-8

"""
Pipeline for gradio
"""

import os.path as osp
import gradio as gr
import torch
from .config.argument_config import ArgumentConfig
from .live_portrait_wmg_pipeline import LivePortraitPipeline
from .live_portrait_wmg_pipeline_animal import LivePortraitPipelineAnimal
from .utils.rprint import rlog as log

def update_args(args, user_args):
    """update the args according to user inputs
    """
    for k, v in user_args.items():
        if hasattr(args, k):
            setattr(args, k, v)
    return args

class GradioPipeline(LivePortraitPipeline):
    """gradio for human
    """
    def __init__(self, inference_cfg, crop_cfg, args: ArgumentConfig):
        super().__init__(inference_cfg, crop_cfg)
        self.args = args

    @torch.no_grad()
    def execute_a2v(
        self,
        input_image=None,
        input_audio=None,
        flag_normalize_lip=False,
        flag_relative_motion=True,
        driving_multiplier=1.0,
        animation_mode="human",
        driving_option_input="pose-friendly",
        flag_do_crop_input=True,
        scale=2.3,
        vx_ratio=0.0,
        vy_ratio=-0.125,
        flag_stitching_input=True,
        flag_remap_input=True,
        cfg_scale=1.2,
    ):
        if input_image is not None and input_audio is not None:
            args_user = {
                'reference': input_image,
                'audio': input_audio,
                'flag_normalize_lip' : flag_normalize_lip,
                'flag_relative_motion': flag_relative_motion,
                'driving_multiplier': driving_multiplier,
                'animation_mode': animation_mode, 
                'driving_option': driving_option_input,
                'flag_do_crop': flag_do_crop_input,
                'scale': scale,
                'vx_ratio': vx_ratio,
                'vy_ratio': vy_ratio,
                'flag_pasteback': flag_remap_input,
                'flag_stitching': flag_stitching_input,
                'cfg_scale': cfg_scale,
            }
            # update config from user input
            self.args = update_args(self.args, args_user)
            self.live_portrait_wrapper.update_config(self.args.__dict__)
            self.cropper.update_config(self.args.__dict__)

            # generate
            output_path = self.execute(self.args)
            gr.Info("Run successfully!", duration=2)

            return output_path
        else:
            raise gr.Error("Please upload the source portrait or source video, and driving video 🤗🤗🤗", duration=5)


class GradioPipelineAnimal(LivePortraitPipelineAnimal):
    """gradio for animal
    """
    def __init__(self, inference_cfg, crop_cfg, args: ArgumentConfig):
        super().__init__(inference_cfg, crop_cfg)
        self.args = args

    @torch.no_grad()
    def execute_a2v(
        self,
        input_image=None,
        input_audio=None,
        flag_normalize_lip=False,
        flag_relative_motion=True,
        driving_multiplier=1.0,
        animation_mode="animal",
        driving_option_input="pose-friendly",
        flag_do_crop_input=True,
        scale=2.3,
        vx_ratio=0.0,
        vy_ratio=-0.125,
        flag_stitching_input=True,
        flag_remap_input=True,
        cfg_scale=1.2,
    ):
        if input_image is not None and input_audio is not None:
            args_user = {
                'reference': input_image,
                'audio': input_audio,
                'flag_normalize_lip' : flag_normalize_lip,
                'flag_relative_motion': flag_relative_motion,
                'driving_multiplier': driving_multiplier,
                'animation_mode': animation_mode, 
                'driving_option': driving_option_input,
                'flag_do_crop': flag_do_crop_input,
                'scale': scale,
                'vx_ratio': vx_ratio,
                'vy_ratio': vy_ratio,
                'flag_pasteback': flag_remap_input,
                'flag_stitching': flag_stitching_input,
                'cfg_scale': cfg_scale,
            }
            # update config from user input
            self.args = update_args(self.args, args_user)
            self.live_portrait_wrapper_animal.update_config(self.args.__dict__)
            self.cropper.update_config(self.args.__dict__)

            # generate
            output_path = self.execute(self.args)
            gr.Info("Run successfully!", duration=2)

            return output_path
        else:
            raise gr.Error("Please upload the source portrait or source video, and driving video 🤗🤗🤗", duration=5)
