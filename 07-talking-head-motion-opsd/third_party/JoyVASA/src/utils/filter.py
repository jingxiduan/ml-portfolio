# coding: utf-8

import torch
import numpy as np
from pykalman import KalmanFilter
PI = np.pi

device = "cuda"
def get_rotation_matrix(pitch_, yaw_, roll_):
    """ the input is in degree
    """
    # transform to radian
    pitch = pitch_ / 180 * PI
    yaw = yaw_ / 180 * PI
    roll = roll_ / 180 * PI

    device = pitch.device

    if pitch.ndim == 1:
        pitch = pitch.unsqueeze(1)
    if yaw.ndim == 1:
        yaw = yaw.unsqueeze(1)
    if roll.ndim == 1:
        roll = roll.unsqueeze(1)

    # calculate the euler matrix
    bs = pitch.shape[0]
    ones = torch.ones([bs, 1]).to(device)
    zeros = torch.zeros([bs, 1]).to(device)
    x, y, z = pitch, yaw, roll

    rot_x = torch.cat([
        ones, zeros, zeros,
        zeros, torch.cos(x), -torch.sin(x),
        zeros, torch.sin(x), torch.cos(x)
    ], dim=1).reshape([bs, 3, 3])

    rot_y = torch.cat([
        torch.cos(y), zeros, torch.sin(y),
        zeros, ones, zeros,
        -torch.sin(y), zeros, torch.cos(y)
    ], dim=1).reshape([bs, 3, 3])

    rot_z = torch.cat([
        torch.cos(z), -torch.sin(z), zeros,
        torch.sin(z), torch.cos(z), zeros,
        zeros, zeros, ones
    ], dim=1).reshape([bs, 3, 3])

    rot = rot_z @ rot_y @ rot_x
    return rot.permute(0, 2, 1)  # transpose

def smooth(x_d_lst, shape, device, observation_variance=3e-7, process_variance=1e-5):
    x_d_lst_reshape = [x.reshape(-1) for x in x_d_lst]
    x_d_stacked = np.vstack(x_d_lst_reshape)
    kf = KalmanFilter(
        initial_state_mean=x_d_stacked[0],
        n_dim_obs=x_d_stacked.shape[1],
        transition_covariance=process_variance * np.eye(x_d_stacked.shape[1]),
        observation_covariance=observation_variance * np.eye(x_d_stacked.shape[1])
    )
    smoothed_state_means, _ = kf.smooth(x_d_stacked)
    x_d_lst_smooth = [torch.tensor(state_mean.reshape(shape[-2:]), dtype=torch.float32, device=device) for state_mean in smoothed_state_means]
    return x_d_lst_smooth

class ExponentialMovingAverageFilter:
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.smoothed_value = None

    def update(self, new_value):
        if self.smoothed_value is None:
            self.smoothed_value = new_value
        else:
            self.smoothed_value = self.alpha * new_value + (1 - self.alpha) * self.smoothed_value
        return self.smoothed_value

class MovingAverageFilter:
    def __init__(self, window_size):
        self.window_size = window_size
        self.buffer = np.zeros((window_size, 7))
        self.index = 0
        self.full = False

    def update(self, new_value):
        # 更新队列
        self.buffer[self.index] = new_value
        self.index = (self.index + 1) % self.window_size
        
        # 如果队列未满，则只计算已有的元素
        if not self.full and self.index == 0:
            self.full = True

        # 计算平均值
        return np.mean(self.buffer[:self.window_size if self.full else self.index], axis=0)

class MedianFilter:
    def __init__(self, window_size):
        self.window_size = window_size
        self.buffer = np.zeros((window_size, 7))
        self.index = 0
        self.full = False

    def update(self, new_value):
        # 更新队列
        self.buffer[self.index] = new_value
        self.index = (self.index + 1) % self.window_size
        
        # 如果队列未满，则只计算已有的元素
        if not self.full and self.index == 0:
            self.full = True

        # 计算中值
        return np.median(self.buffer[:self.window_size if self.full else self.index], axis=0)

def smooth_(ori_data, method="median"):
    # 均值滤波 & 中值滤波
    data_array = []
    for frame_idx in range(ori_data["n_frames"]):
        data_array.append(
            np.concatenate((
                ori_data['motion'][frame_idx]["scale"].flatten(),
                ori_data['motion'][frame_idx]["t"].flatten(),
                ori_data['motion'][frame_idx]["pitch"].flatten(),
                ori_data['motion'][frame_idx]["yaw"].flatten(),
                ori_data['motion'][frame_idx]["roll"].flatten(),
            ))
        )
    data_array = np.array(data_array).astype(np.float32)
    # print("data_array.shape: ", data_array.shape)
    
    # 滑动窗口大小
    if method == "median":
        window_size = 3
        ma_filter = MedianFilter(window_size)
    elif method == "ema":
        ma_filter = ExponentialMovingAverageFilter(alpha=0.01)
    else: 
        window_size = 10
        ma_filter = MovingAverageFilter(window_size)
    smoothed_data = []
    for value in data_array:
        smoothed_value = ma_filter.update(value)
        smoothed_data.append(smoothed_value)
    smoothed_data = np.array(smoothed_data).astype(np.float32)
    # print("smoothed_data_mean.shape: ", smoothed_data.shape)

    # 整理结果
    motion_list = []
    for idx in range(smoothed_data.shape[0]):
        exp = ori_data["motion"][idx]["exp"]
        scale = smoothed_data[idx][0:1].reshape(1, 1)
        # scale = 1.2 * np.ones((1, 1)).reshape(1, 1).astype(np.float32)
        t = smoothed_data[idx][1:4].reshape(1, 3).astype(np.float32)
        pitch = smoothed_data[idx][4:5].reshape(1, 1).astype(np.float32)
        yaw = smoothed_data[idx][5:6].reshape(1, 1).astype(np.float32)
        roll = smoothed_data[idx][6:7].reshape(1, 1).astype(np.float32)
        R = get_rotation_matrix(torch.FloatTensor(pitch), torch.FloatTensor(yaw), torch.FloatTensor(roll))
        R = R.reshape(1, 3, 3).cpu().numpy().astype(np.float32)

        motion_list.append({"exp": exp, "scale": scale, "t": t, "pitch": pitch, "yaw": yaw, "roll": roll, "R": R})
    # print(f"exp: {exp.shape}, scale: {scale.shape}, t: {t.shape}, pitch: {pitch.shape}, yaw: {yaw.shape}, roll: {roll.shape}, R: {R.shape}")
    tgt_motion = {'n_frames': smoothed_data.shape[0], 'output_fps': 25, 'motion': motion_list}
    return tgt_motion
