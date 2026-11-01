import os
import tyro
import multiprocessing
import sys
sys.path.append(os.path.dirname(os.path.abspath("../")))

from src.config.argument_config import ArgumentConfig
from src.motion_extractor import make_motion_templete


def process_videos(args, video_list, suffix):
    params = [(args, driving_video, suffix) for driving_video in video_list]
    with multiprocessing.Pool(processes=2) as pool:
        pool.starmap(make_motion_templete, params)

args = tyro.cli(ArgumentConfig)
args.flag_do_crop = False
args.scale = 2.3

root_dir = "/path/to/data"
video_names = sorted([os.path.join(root_dir, filename) for filename in os.listdir(root_dir) if filename.endswith("mp4")])
process_videos(args, video_names, suffix=".pkl")