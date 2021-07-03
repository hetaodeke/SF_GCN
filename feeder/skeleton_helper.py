import os
import logging
import json
import numpy as np
import random

import torch


logger = logging.getLogger(__name__)


def auto_pading(data_numpy, size, random_pad=False):
    C, T, V, M = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V, M))
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy


def random_choose(data_numpy, size, auto_pad=True):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        else:
            return data_numpy
    else:
        begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]


def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1]):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])

    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy


def random_shift(data_numpy):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]

    return data_shift

def data_preprocess(input_path, output_path):
    video_lst = os.listdir(input_path)
    for video in video_lst:
        video_path = os.path.join(input_path, video)
        output_video_path = os.path.join(output_path, video)
        if not os.path.exists(output_video_path):
            os.makedirs(output_video_path)
        frame_lst = os.listdir(video_path)
        for frame in frame_lst:
            frame_path = os.path.join(video_path, frame)
            frame_id = int(frame_lst.split('_')[4])
            output_json = {'frame_id':frame_id}
            output_json['skeleton'] = []
            with open(frame_path, 'r+') as f:
                frame_info = json.load(f)
            for pose_keypoints in frame_info['people']:
                pose = []
                score = []
                skeleton_info = {}
                for idx, items in enumerate(pose_keypoints['pose_keypoints_2d']):
                    if (idx + 1)%3 == 0:
                        score.append(items)
                    else:
                        pose.append(items)
                skeleton_info["pose"] = pose
                skeleton_info["score"] = score
            output_json['skeleton'].append(skeleton_info)
            output_json_path = os.path.join(output_video_path, frame)
            with open(output_json_path, 'w+') as of:
                json.dump(of)          

def pack_pathway_output(args, frames):
    """
    Prepare output as a list of tensors. Each tensor corresponding to a
    unique pathway.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
    Returns:
        frame_list (list): list of tensors with the dimension of
            `channel` x `num frames` x `height` x `width`.
    """
    # if cfg.MODEL.ARCH in cfg.MODEL.SINGLE_PATHWAY_ARCH:
    #     frame_list = [frames]
    # elif cfg.MODEL.ARCH in cfg.MODEL.DOUBLE_PATHWAY_ARCH:
    fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
    if isinstance(frames, torch.Tensor):
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // args.alpha
            ).long(),
        )
    else:
        slow_pathway = torch.index_select(
            torch.from_numpy(frames),
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // args.alpha
            ).long(),
        )
    frame_list = [slow_pathway, fast_pathway]
    # elif cfg.MODEL.ARCH in cfg.MODEL.MULTI_PATHWAY_ARCH:
    #     fast_pathway = frames
    #     # Perform temporal sampling from the fast pathway.
    #     slow_pathway = torch.index_select(
    #         frames,
    #         1,
    #         torch.linspace(
    #             0, frames.shape[1] - 1, frames.shape[1] // args.alpha
    #         ).long(),
    #     )
    #     middle_pathway = torch.index_select(
    #         frames,
    #         1,
    #         torch.linspace(
    #             0, frames.shape[1] - 1, (frames.shape[1] // args.alpha) * 2
    #         ).long(),
    #     )
    #     frame_list = [slow_pathway, middle_pathway, fast_pathway]        
    # else:
    #     raise NotImplementedError(
    #         "Model arch {} is not in {}".format(
    #             cfg.MODEL.ARCH,
    #             cfg.MODEL.SINGLE_PATHWAY_ARCH + cfg.MODEL.MULTI_PATHWAY_ARCH,
    #         )
    #     )
    return frame_list