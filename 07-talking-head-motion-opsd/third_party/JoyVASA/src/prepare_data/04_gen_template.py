import pickle
import numpy as np

data_root = "motions.pkl"
motions = pickle.load(open(data_root, 'rb'))

scale_list = []
R_list = []
pitch_list = []
yaw_list = []
roll_list = []
t_list = []
exp_list = []

audio_names = motions.keys()
for audio_name in audio_names:
    motion_data = motions[audio_name]
    seq_len = motion_data["n_frames"]
    for frame_idx in range(seq_len):
        scale_list.append(motion_data['motion'][frame_idx]["scale"].flatten())
        R_list.append(motion_data['motion'][frame_idx]["R"].flatten())
        t_list.append(motion_data['motion'][frame_idx]["t"].flatten())
        exp_list.append(motion_data['motion'][frame_idx]["exp"].flatten())
        pitch_list.append(motion_data['motion'][frame_idx]["pitch"].flatten())
        yaw_list.append(motion_data['motion'][frame_idx]["yaw"].flatten())
        roll_list.append(motion_data['motion'][frame_idx]["roll"].flatten())

scale_array = np.array(scale_list)
R_array = np.array(R_list)
t_array = np.array(t_list)
exp_array = np.array(exp_list)
pitch_array = np.array(pitch_list)
yaw_array = np.array(yaw_list)
roll_array = np.array(roll_list)
print(scale_array.shape, R_array.shape, t_array.shape, exp_array.shape, pitch_array.shape, yaw_array.shape, roll_array.shape)
lip_lst_array = np.array([data.flatten() for data in motion_data['c_lip_lst']]).astype(np.float32)
eyes_lst_array = np.array([data.flatten() for data in motion_data['c_eyes_lst']]).astype(np.float32)
print(f"lip_aray: {lip_lst_array.shape}, eyes_lst_array: {eyes_lst_array.shape}")

# abs max
abs_max_scale = np.max(abs(scale_array), axis=0)
abs_max_R = np.max(abs(R_array), axis=0)
abs_max_t = np.max(abs(t_array), axis=0)
abs_max_exp = np.max(abs(exp_array), axis=0)
abs_max_pitch = np.max(abs(pitch_array), axis=0)
abs_max_yaw = np.max(abs(yaw_array), axis=0)
abs_max_roll = np.max(abs(roll_array), axis=0)
abs_max_lip = np.max(abs(lip_lst_array), axis=0)
abs_max_eyes = np.max(abs(eyes_lst_array), axis=0)
print("absmax: ", abs_max_scale.shape, abs_max_R.shape, abs_max_t.shape, abs_max_exp.shape, abs_max_pitch.shape, abs_max_pitch.shape, abs_max_roll.shape, abs_max_lip.shape, abs_max_eyes.shape)

# max
max_scale = np.max(scale_array, axis=0)
max_R = np.max(R_array, axis=0)
max_t = np.max(t_array, axis=0)
max_exp = np.max(exp_array, axis=0)
max_pitch = np.max(pitch_array, axis=0)
max_yaw = np.max(yaw_array, axis=0)
max_roll = np.max(roll_array, axis=0)
max_lip = np.max(lip_lst_array, axis=0)
max_eyes = np.max(eyes_lst_array, axis=0)
print("max: ", max_scale.shape, max_R.shape, max_t.shape, max_exp.shape, max_pitch.shape, max_pitch.shape, max_roll.shape, max_lip.shape, max_eyes.shape)

# min
min_scale = np.min(scale_array, axis=0)
min_R = np.min(R_array, axis=0)
min_t = np.min(t_array, axis=0)
min_exp = np.min(exp_array, axis=0)
min_pitch = np.min(pitch_array, axis=0)
min_yaw = np.min(yaw_array, axis=0)
min_roll = np.min(roll_array, axis=0)
min_lip = np.min(lip_lst_array, axis=0)
min_eyes = np.min(eyes_lst_array, axis=0)
print("min: ", min_scale.shape, min_R.shape, min_t.shape, min_exp.shape, min_pitch.shape, min_pitch.shape, min_roll.shape, min_lip.shape, min_eyes.shape)

# mean
mean_scale = np.mean(scale_array, axis=0)
mean_R = np.mean(R_array, axis=0)
mean_t = np.mean(t_array, axis=0)
mean_exp = np.mean(exp_array, axis=0)
mean_pitch = np.mean(pitch_array, axis=0)
mean_yaw = np.mean(yaw_array, axis=0)
mean_roll = np.mean(roll_array, axis=0)
mean_lip = np.mean(lip_lst_array, axis=0)
mean_eyes = np.mean(eyes_lst_array, axis=0)
print("mean: ", mean_scale.shape, mean_R.shape, mean_t.shape, mean_exp.shape, mean_pitch.shape, mean_yaw.shape, mean_roll.shape, mean_lip.shape, mean_eyes.shape)

# std
std_scale = np.std(scale_array, axis=0)
std_R = np.std(R_array, axis=0)
std_t = np.std(t_array, axis=0)
std_exp = np.std(exp_array, axis=0)
std_pitch = np.std(pitch_array, axis=0)
std_yaw = np.std(yaw_array, axis=0)
std_roll = np.std(roll_array, axis=0)
std_lip = np.std(lip_lst_array, axis=0)
std_eyes = np.std(eyes_lst_array, axis=0)
print("std: ", std_scale.shape, std_R.shape, std_t.shape, std_exp.shape, std_pitch.shape, std_yaw.shape, std_roll.shape, std_lip.shape, std_eyes.shape)

motion_template = {
    "mean_scale": mean_scale,
    "mean_R": mean_R,
    "mean_t": mean_t,
    "mean_exp": mean_exp,
    "mean_pitch": mean_pitch,
    "mean_yaw": mean_yaw,
    "mean_roll": mean_roll,
    "mean_lip": mean_lip,
    "mean_eyes": mean_eyes,
    "std_scale": std_scale,
    "std_R": std_R,
    "std_t": std_t,
    "std_exp": std_exp, 
    "std_pitch": std_pitch, 
    "std_yaw": std_yaw, 
    "std_roll": std_roll, 
    "std_lip": std_lip,
    "std_eyes": std_eyes,
    "max_scale": max_scale,
    "max_R": max_R,
    "max_t": max_t,
    "max_exp": max_exp,
    "max_pitch": max_pitch,
    "max_yaw": max_yaw,
    "max_roll": max_roll,
    "max_lip": max_lip,
    "max_eyes": max_eyes,
    "min_scale": min_scale,
    "min_R": min_R,
    "min_t": min_t,
    "min_exp": min_exp,
    "min_pitch": min_pitch,
    "min_yaw": min_yaw,
    "min_roll": min_roll,
    "min_lip": min_lip,
    "min_eyes": min_eyes,
    "abs_max_scale": abs_max_scale,
    "abs_max_R": abs_max_R,
    "abs_max_t": abs_max_t,
    "abs_max_exp": abs_max_exp,
    "abs_max_pitch": abs_max_pitch,
    "abs_max_yaw": abs_max_yaw,
    "abs_max_roll": abs_max_roll,
    "abs_max_lip": abs_max_lip,
    "abs_max_eyes": abs_max_eyes,
}
pickle.dump(motion_template, open(f"motion_templete.pkl", 'wb'))