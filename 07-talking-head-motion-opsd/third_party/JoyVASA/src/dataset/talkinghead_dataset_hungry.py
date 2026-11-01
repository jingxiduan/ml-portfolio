import json
import os
import io
import torchaudio
import numpy as np
import torch
from torch.utils import data
import pickle
import warnings
import torch.nn.functional as F

torchaudio.set_audio_backend('soundfile')
warnings.filterwarnings('ignore', message='PySoundFile failed. Trying audioread instead.')

class TalkingHeadDatasetHungry(data.Dataset):
    def __init__(self, root_dir, motion_filename="talking_face.pkl", motion_templete_filename="motion_templete.pkl", 
                 split="train", coef_fps=25, n_motions=100, crop_strategy="random", normalize_type="mix"):
        self.templete_dir = os.path.join(root_dir, motion_templete_filename)
        self.templete_dict = pickle.load(open(self.templete_dir, 'rb'))
        self.motion_dir = os.path.join(root_dir, motion_filename)
        self.eps = 1e-9
        self.normalize_type = normalize_type

        if split == "train":
            self.root_dir = os.path.join(root_dir, "train.json")
        else:
            self.root_dir = os.path.join(root_dir, "test.json")

        with open(self.root_dir, 'r') as f:
            json_data = json.load(f)
        self.all_data = json_data
        self.motion_data = pickle.load(open(self.motion_dir, "rb"))
        print("load all motion data done...")

        self.coef_fps = coef_fps
        self.audio_unit = 16000. / self.coef_fps  # num of samples per frame
        self.n_motions = n_motions
        self.n_audio_samples = round(self.audio_unit * self.n_motions)
        self.coef_total_len = self.n_motions * 2
        self.audio_total_len = round(self.audio_unit * self.coef_total_len)
        self.crop_strategy = crop_strategy
        
    def __len__(self, ):
        return len(self.all_data)
    
    def check_motion_length(self, motion_data):
        seq_len = motion_data["n_frames"]
        
        if seq_len > self.coef_total_len + 2: 
            return motion_data
        else:
            exp_list, t_list, scale_list, pitch_list, yaw_list, roll_list = [], [], [], [], [], []
            for frame_index in range(motion_data["n_frames"]):
                exp_list.append(motion_data["motion"][frame_index]["exp"])
                t_list.append(motion_data["motion"][frame_index]["t"])
                scale_list.append(motion_data["motion"][frame_index]["scale"])
                pitch_list.append(motion_data["motion"][frame_index]["pitch"])
                yaw_list.append(motion_data["motion"][frame_index]["yaw"])
                roll_list.append(motion_data["motion"][frame_index]["roll"])

            repeat = 0 
            while len(exp_list) < self.coef_total_len + 2:
                exp_list = exp_list * 2
                t_list = t_list * 2
                scale_list = scale_list * 2
                pitch_list = pitch_list * 2
                yaw_list = yaw_list * 2
                roll_list = roll_list * 2
                repeat += 1
            
            motion_new = {"motion": []}
            for i in range(len(exp_list)):
                motion = {
                    "exp": exp_list[i],
                    "t": t_list[i],
                    "scale": scale_list[i],
                    "pitch": pitch_list[i],
                    "yaw": yaw_list[i],
                    "roll": roll_list[i],
                }
                motion_new["motion"].append(motion)
            motion_new["n_frames"] = len(exp_list)
            motion_new["repeat"] = repeat
        return motion_new
    
    def __getitem__(self, index):
        has_valid_audio = False
        while not has_valid_audio:
            # read motion
            metadata = self.all_data[index]
            motion_data = self.motion_data[metadata["audio_name"]]
            motion_data = self.check_motion_length(motion_data)
            
            # crop audo and coef, count start_frame
            seq_len = motion_data["n_frames"]
            if self.crop_strategy == 'random':
                end = seq_len - self.coef_total_len
                if end < 0:
                    print(f"current data invalid: {os.path.basename(metadata['audio_name'])}, n_frames: {seq_len}")
                    has_valid_audio = False 
                    continue
                start_frame = np.random.randint(0, seq_len - self.coef_total_len)
            elif self.crop_strategy == 'begin':
                start_frame = 0
            elif self.crop_strategy == 'end':
                start_frame = seq_len - self.coef_total_len
            else:
                raise ValueError(f'Unknown crop strategy: {self.crop_strategy}')
            end_frame = start_frame + self.coef_total_len

            exp, scale, t, pitch, yaw, roll = [], [], [], [], [], []
            for frame_idx in range(motion_data["n_frames"]):
                exp.append((motion_data['motion'][frame_idx]["exp"].flatten() - self.templete_dict["mean_exp"]) / (self.templete_dict["std_exp"] + self.eps))
                scale.append((motion_data['motion'][frame_idx]["scale"].flatten() - self.templete_dict["mean_scale"]) / (self.templete_dict["std_scale"] + self.eps))
                t.append((motion_data['motion'][frame_idx]["t"].flatten() - self.templete_dict["mean_t"]) / (self.templete_dict["std_t"] + self.eps))
                pitch.append((motion_data['motion'][frame_idx]["pitch"].flatten() - self.templete_dict["mean_pitch"]) / (self.templete_dict["std_pitch"] + self.eps))
                yaw.append((motion_data['motion'][frame_idx]["yaw"].flatten() - self.templete_dict["mean_yaw"]) / (self.templete_dict["std_yaw"] + self.eps))
                roll.append((motion_data['motion'][frame_idx]["roll"].flatten() - self.templete_dict["mean_roll"]) / (self.templete_dict["std_roll"] + self.eps))

            # load motion & normalize
            coef_keys = ["exp", "pose"] # exp - > exp, ['scale', 't', 'yaw', 'pitch', 'roll'] -> "pose"
            coef_dict = {k: [] for k in coef_keys}
            audio = []
            for frame_idx in range(start_frame, end_frame):
                for coef_key in coef_keys:
                    if coef_key == "exp":
                        if self.normalize_type == "mix":
                            normalized_exp = (motion_data['motion'][frame_idx]["exp"].flatten() - self.templete_dict["mean_exp"]) / (self.templete_dict["std_exp"] + self.eps)
                        else:
                            raise RuntimeError("error")
                        coef_dict[coef_key].append([normalized_exp, ])
                    elif coef_key == "pose":
                        if self.normalize_type == "mix":
                            pose_data = np.concatenate((
                                (motion_data['motion'][frame_idx]["scale"].flatten() - self.templete_dict["min_scale"]) / (self.templete_dict["max_scale"] - self.templete_dict["min_scale"] + self.eps),
                                (motion_data['motion'][frame_idx]["t"].flatten() - self.templete_dict["min_t"]) / (self.templete_dict["max_t"] - self.templete_dict["min_t"] + self.eps),
                                (motion_data['motion'][frame_idx]["pitch"].flatten() - self.templete_dict["min_pitch"]) / (self.templete_dict["max_pitch"] - self.templete_dict["min_pitch"] + self.eps),
                                (motion_data['motion'][frame_idx]["yaw"].flatten() - self.templete_dict["min_yaw"]) / (self.templete_dict["max_yaw"] - self.templete_dict["min_yaw"] + self.eps),
                                (motion_data['motion'][frame_idx]["roll"].flatten() - self.templete_dict["min_roll"]) / (self.templete_dict["max_roll"] - self.templete_dict["min_roll"] + self.eps),
                            ))
                        else:
                            raise RuntimeError("pose data error")

                        coef_dict[coef_key].append([pose_data, ])
                    else:
                        raise RuntimeError("coef_key error: ", coef_key)
                        
            coef_dict = {k: torch.tensor(np.concatenate(coef_dict[k], axis=0)) for k in coef_keys}
            assert coef_dict['exp'].shape[0] == self.coef_total_len, f'Invalid coef length: {coef_dict["exp"].shape[0]}'

            # load audio & normalize
            audio_path = metadata["audio_name"]
            audio_clip, sr = torchaudio.load(audio_path)
            audio_clip = audio_clip.squeeze()
            if "repeat" in motion_data:
                for _ in range(motion_data["repeat"]):
                    audio_clip = torch.cat((audio_clip, audio_clip), dim=0)

            assert sr == 16000, f'Invalid sampling rate: {sr}'
            audio.append(audio_clip[round(start_frame * self.audio_unit):round(end_frame * self.audio_unit)])
            audio = torch.cat(audio, dim=0)
            if not (audio.shape[0] == self.coef_total_len * self.audio_unit):
                print(f"audio length invalid! audio: {audio.shape[0]}, coef: {self.coef_total_len * self.audio_unit}")
                has_valid_audio = False 
                continue

            # Extract two consecutive audio/coef clips
            keys = ['exp', 'pose']
            audio_pair = [audio[:self.n_audio_samples].clone(), audio[-self.n_audio_samples:].clone()]
            coef_pair = [{k: coef_dict[k][:self.n_motions].clone() for k in keys},
                        {k: coef_dict[k][-self.n_motions:].clone() for k in keys}]
            has_valid_audio = True
            return audio_pair, coef_pair


if __name__ == "__main__":
    data_root = "./"
    metainfo_filename = "labels.json"
    motion_filename = "motions.pkl"
    motion_templete_filename = "templete.pkl"
    normalize_type = "mix" 

    train_dataset = TalkingHeadDatasetHungry(data_root, 
                                             motion_filename=motion_filename, 
                                             motion_templete_filename=motion_templete_filename,
                                             split="train", coef_fps=25, n_motions=100, crop_strategy="random", normalize_type=normalize_type)
    train_loader = data.DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=8, pin_memory=True)
    for audio_pair, coef_pair, (audio_mean, audio_std) in train_loader:
        print(f"audio: {audio_pair[0].shape}, {audio_pair[1].shape}, exp: {coef_pair[0]['exp'].shape}, \
              {coef_pair[1]['exp'].shape}, pose: {coef_pair[0]['pose'].shape}, {coef_pair[1]['pose'].shape}")