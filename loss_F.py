import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def intensity_loss(gen_frames, gt_frames, l_num=2):
    """
        Calculates the sum of lp losses between the predicted and ground truth frames.

        @param gen_frames: The predicted frames at each scale.
        @param gt_frames: The ground truth frames at each scale
        @param l_num: 1 or 2 for l1 and l2 loss, respectively).

        @return: The lp loss.
        """
    return torch.mean(torch.abs((gen_frames - gt_frames) ** l_num))


def compute_psnr(gen_frames, gt_frames):
    mse = F.mse_loss(gen_frames, gt_frames)
    psnr = 10 * torch.log10(1 / mse)
    return psnr


def cosine_similarity(gen_frames, gt_frames):
    batch_size = gen_frames.shape[0]
    p = gen_frames.view(batch_size, -1)
    x = gt_frames.view(batch_size, -1)
    losses = 1 - F.cosine_similarity(p, x, dim=1, eps=1e-08)
    return torch.mean(losses)

def gdl_loss(gen_frames, gt_frames, local_rank ,alpha=2):
    """
    Calculates the sum of GDL losses between the predicted and ground truth frames.
    @param gen_frames: The predicted frames at each scale.
    @param gt_frames: The ground truth frames at each scale
    @param alpha: The power to which each gradient term is raised.
    @return: The GDL loss.
    """
    filter_x_values = np.array(
        [
            [[[-1, 1, 0, 0]], [[0, 0, 0, 0]], [[0, 0, 0, 0]], [[0, 0, 0, 0]]],
            [[[0, 0, 0, 0]], [[-1, 1, 0, 0]], [[0, 0, 0, 0]], [[0, 0, 0, 0]]],
            [[[0, 0, 0, 0]], [[0, 0, 0, 0]], [[-1, 1, 0, 0]], [[0, 0, 0, 0]]],
            [[[0, 0, 0, 0]], [[0, 0, 0, 0]], [[-1, 1, 0, 0]], [[-1, 1, 0, 0]]],

        ],
        dtype=np.float32,
    )

    filter_x_values = np.repeat(filter_x_values, 64, axis=1) # 256/4 = 64
    filter_x = nn.Conv2d(256, 4, (1, 4), padding=(0, 1))
    filter_x.weight = nn.Parameter(torch.from_numpy(filter_x_values))
    filter_y_values = np.array(
        [
            [[[-1], [1], [0], [0]], [[0], [0], [0], [0]], [[0], [0], [0], [0]],[[0], [0], [0], [0]]],
            [[[0], [0], [0], [0]], [[-1], [1], [0], [0]], [[0], [0], [0], [0]],[[0], [0], [0], [0]]],
            [[[0], [0], [0], [0]], [[0], [0], [0], [0]], [[-1], [1], [0], [0]],[[0], [0], [0], [0]]],
            [[[0], [0], [0], [0]], [[0], [0], [0], [0]],  [[0], [0], [0], [0]],[[-1], [1], [0], [0]],],

        ],
        dtype=np.float32,
    )

    filter_y_values = np.repeat(filter_y_values, 64, axis=1)  # 256/4 = 64
    filter_y = nn.Conv2d(256, 4, (4, 1), padding=(1, 0))
    filter_y.weight = nn.Parameter(torch.from_numpy(filter_y_values))
    filter_x = filter_x.to(local_rank)
    filter_y = filter_y.to(local_rank)

    gen_dx = filter_x(gen_frames)
    gen_dy = filter_y(gen_frames)
    gt_dx = filter_x(gt_frames)
    gt_dy = filter_y(gt_frames)

    grad_diff_x = torch.pow(torch.abs(gt_dx - gen_dx), alpha)
    grad_diff_y = torch.pow(torch.abs(gt_dy - gen_dy), alpha)

    grad_total_x = torch.mean(grad_diff_x)
    grad_total_y = torch.mean(grad_diff_y)
    grad_total = (grad_total_x + grad_total_y) / 2
    return grad_total