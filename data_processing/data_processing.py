import numpy as np
from typing import List, Dict
from pathlib import Path

angle_min = -2.09442138672
angle_increment = 0.00581718236208
angles = np.arange(721) * angle_increment + angle_min
der_x_robot = 0.329578
der_l_lm = 0.02
obstacle_info = [[1.052, -2.695], [4.072, -1.752], [6.028, - 3.324]]


def get_from_mappings(mappings: List[Dict], key, default=None):
    return [m.get(key, default) for m in mappings]


def process_lidar(pose: List[float], lidar: List[float], obstacle_info: List[List[float]]):
    if obstacle_info is None:
        return None
    x, y, theta = pose
    pos = np.array([x, y]) + der_x_robot * np.array([np.cos(theta), np.sin(theta)])  # shape: (2, )
    theta = -theta
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    lidar = np.array(lidar)  # shape: (721, )
    centers = (np.array(obstacle_info) - pos).dot(rot.T)  # shape: (n, 2)
    point_xy = np.stack([lidar * np.cos(angles), lidar * np.sin(angles)], axis=1)  # shape: (721, 2)
    obs_ranges = []
    obs_angles = []
    for center in centers:
        dist = np.linalg.norm(point_xy - center, axis=1)  # shape: (721, )
        # print(dist)
        target = (dist < 1.9)
        # print(np.sum(target))
        if np.sum(target) == 0:
            return None
        range = lidar[target].mean()
        angle = angles[target].mean()
        obs_ranges.append(range)
        obs_angles.append(angle)
    return obs_ranges + obs_angles


def post_process(filepath, obstacle_info, min_len):
    data = np.load(filepath, allow_pickle=True)['arr_0']  # data: list of optional [dict | None]
    processed = []
    temp_list = []
    abnormal = 0
    for d in data:
        if d is None:
            if temp_list:
                processed.append(temp_list)
                temp_list = []
        else:
            d['obs'] = process_lidar(d['state'], d['lidar'], obstacle_info)
            if d['obs'] is None:
                abnormal += 1
                temp_list.clear()
                continue
            temp_list.append(d)
            if len(temp_list) >= min_len:
                processed.append(temp_list)
                temp_list = []
    # filter too short
    processed = [p for p in processed if len(p) >= min_len]
    state = [get_from_mappings(p, 'state') for p in processed]
    action = [get_from_mappings(p, 'control') for p in processed]
    time = [get_from_mappings(p, 'time') for p in processed]
    obs = [get_from_mappings(p, 'obs') for p in processed]
    lidar = [get_from_mappings(p, 'lidar') for p in processed]
    print('num of abnormal frame: ', abnormal)
    return state, action, time, obs, lidar


def load_data(filepath, min_len):
    state, action, time, obs, lidar = post_process(filepath, obstacle_info, min_len=min_len)
    return state, action, obs


if __name__ == '__main__':
    filepath = r'/data_processing\20230216-140452.npz'
    min_len = 100
    state, action, obs = load_data(filepath, min_len)
