import os
import tyro
import subprocess
import gradio as gr
import os.path as osp
import platform
from src.utils.helper import load_description
from src.gradio_pipeline import GradioPipeline, GradioPipelineAnimal
from src.config.crop_config import CropConfig
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig

if platform.system() == "Windows":
    import pathlib
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False

ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg")
if osp.exists(ffmpeg_dir):
    os.environ["PATH"] += (os.pathsep + ffmpeg_dir)
if not fast_check_ffmpeg():
    raise ImportError(
        "FFmpeg is not installed. Please install FFmpeg (including ffmpeg and ffprobe) before running this script. https://ffmpeg.org/download.html"
    )

# set tyro theme
tyro.extras.set_accent_color("bright_cyan")
args = tyro.cli(ArgumentConfig)

# specify configs for inference
inference_cfg = partial_fields(InferenceConfig, args.__dict__)  # use attribute of args to initial InferenceConfig
crop_cfg = partial_fields(CropConfig, args.__dict__)  # use attribute of args to initial CropConfig


############# Functions #################
if args.gradio_temp_dir not in (None, ''):
    os.environ["GRADIO_TEMP_DIR"] = args.gradio_temp_dir
    os.makedirs(args.gradio_temp_dir, exist_ok=True)

gradio_pipeline_human = GradioPipeline(
    inference_cfg=inference_cfg,
    crop_cfg=crop_cfg,
    args=args
)
gradio_pipeline_animal = GradioPipelineAnimal(
    inference_cfg=inference_cfg,
    crop_cfg=crop_cfg,
    args=args
)
def gpu_wrapped_execute_a2v(*args, **kwargs):
    # print("args: ", args, args[5])
    if args[5] == "animal":
        return gradio_pipeline_animal.execute_a2v(*args, **kwargs)
    else:
        return gradio_pipeline_human.execute_a2v(*args, **kwargs)


################# GUI ################
title_md = "assets/gradio/gradio_title.md"
example_reference_dir = "assets/examples/imgs"
example_audio_dir = "assets/examples/audios"
data_examples_a2v = [
    [osp.join(example_reference_dir, "joyvasa_001.png"), osp.join(example_audio_dir, "joyvasa_001.wav"), "animal", False, 4.0],
    [osp.join(example_reference_dir, "joyvasa_002.png"), osp.join(example_audio_dir, "joyvasa_002.wav"), "animal", False, 4.0],
    [osp.join(example_reference_dir, "joyvasa_003.png"), osp.join(example_audio_dir, "joyvasa_003.wav"), "human", False, 4.0],
    [osp.join(example_reference_dir, "joyvasa_004.png"), osp.join(example_audio_dir, "joyvasa_004.wav"), "human", False, 4.0],
    [osp.join(example_reference_dir, "joyvasa_005.png"), osp.join(example_audio_dir, "joyvasa_005.wav"), "human", False, 4.0],
    [osp.join(example_reference_dir, "joyvasa_006.png"), osp.join(example_audio_dir, "joyvasa_006.wav"), "human", False, 4.0],
]

with gr.Blocks(theme=gr.themes.Soft(font=[gr.themes.GoogleFont("Plus Jakarta Sans")])) as demo:
    gr.HTML(load_description(title_md))
    
    # Inputs & Outputs
    gr.Markdown(load_description("assets/gradio/gradio_description_upload.md"))
    with gr.Row():
        with gr.Accordion(open=True, label="🖼️ Reference Image"):
            input_image = gr.Image(type="filepath", width=512, label="Reference Image")
            gr.Examples(
                examples=[
                    [osp.join(example_reference_dir, "joyvasa_001.png")],
                    [osp.join(example_reference_dir, "joyvasa_002.png")],
                    [osp.join(example_reference_dir, "joyvasa_003.png")],
                    [osp.join(example_reference_dir, "joyvasa_004.png")],
                    [osp.join(example_reference_dir, "joyvasa_005.png")],
                    [osp.join(example_reference_dir, "joyvasa_006.png")],
                ],
                inputs=[input_image],
                cache_examples=False,
            )
        with gr.Accordion(open=True, label="🎵 Input Audio"):
            input_audio = gr.Audio(type="filepath", label="Input Audio")
            gr.Examples(
                examples=[
                    [osp.join(example_audio_dir, "joyvasa_001.wav")],
                    [osp.join(example_audio_dir, "joyvasa_002.wav")],
                    [osp.join(example_audio_dir, "joyvasa_003.wav")],
                    [osp.join(example_audio_dir, "joyvasa_004.wav")],
                    [osp.join(example_audio_dir, "joyvasa_005.wav")],
                    [osp.join(example_audio_dir, "joyvasa_006.wav")],
                ],
                inputs=[input_audio],
                cache_examples=False,
            )
        with gr.Accordion(open=True, label="🎬 Output Video",):
            output_video = gr.Video(autoplay=False, interactive=False, width=512)

    # Configs        
    gr.Markdown(load_description("assets/gradio/gradio_description_configuration.md"))

    with gr.Column():
        with gr.Accordion(open=True, label="Key Animation Options"):
            with gr.Row():
                animation_mode =gr.Radio(['human', 'animal'], value="human", label="Animation Mode") 
                flag_do_crop_input = gr.Checkbox(value=True, label="do crop (image)")
                cfg_scale = gr.Number(value=4.0, label="cfg_scale", minimum=0.0, maximum=10.0, step=0.5)
        with gr.Accordion(open=False, label="Optional Animation Options"):
            with gr.Row():
                driving_option_input = gr.Radio(['expression-friendly', 'pose-friendly'], value="expression-friendly", label="driving option")
                driving_multiplier = gr.Number(value=1.0, label="driving multiplier", minimum=0.0, maximum=2.0, step=0.02)
            with gr.Row():
                flag_normalize_lip = gr.Checkbox(value=True, label="normalize lip")
                flag_relative_motion = gr.Checkbox(value=True, label="relative motion")
                flag_remap_input = gr.Checkbox(value=True, label="paste-back")
                flag_stitching_input = gr.Checkbox(value=True, label="stitching")
        with gr.Accordion(open=False, label="Optional Options for Reference Image"):
            with gr.Row():
                scale = gr.Number(value=2.3, label="image crop scale", minimum=1.8, maximum=4.0, step=0.05)
                vx_ratio = gr.Number(value=0.0, label="image crop x", minimum=-0.5, maximum=0.5, step=0.01)
                vy_ratio = gr.Number(value=-0.125, label="image crop y", minimum=-0.5, maximum=0.5, step=0.01)

    # Generate
    gr.Markdown(load_description("assets/gradio/gradio_description_generate.md"))
    with gr.Row():
        process_button_generate = gr.Button("🚀 Generate", variant="primary")
    
    # Examples
    gr.Examples(
        examples=data_examples_a2v,
        inputs=[input_image,
                input_audio,
                animation_mode, 
                flag_do_crop_input,
                cfg_scale,
                ],
        outputs=[output_video],
        cache_examples=False
    )

    # Binding Functions for Buttons
    generation_func = gpu_wrapped_execute_a2v
    process_button_generate.click(
        fn=generation_func,
        inputs=[
            input_image,
            input_audio,
            flag_normalize_lip,
            flag_relative_motion,
            driving_multiplier,
            animation_mode,
            driving_option_input,
            flag_do_crop_input,
            scale,
            vx_ratio,
            vy_ratio,
            flag_stitching_input,
            flag_remap_input,
            cfg_scale,
        ],
        outputs=[
            output_video,
        ],
        show_progress=True
    )

demo.launch(
    server_port=args.server_port,
    share=args.share,
    server_name=args.server_name
)