import numpy as np
from numba import njit
from typing import Tuple


@njit
def _njit_max_pooling_1d(xs: np.ndarray, result: np.ndarray, kernel_size: Tuple[int], stride: Tuple[int], dilation: int):
    n_samples = xs.shape[0]
    n_channels = xs.shape[1]
    res_len = result.shape[2]
    ks = kernel_size[0]
    st = stride[0]

    for s in range(n_samples):
        for c in range(n_channels):
            for r in range(res_len):
                corner = st * r
                cur_max = -np.inf
                for j in range(corner, corner + ks, dilation):
                    cur_value = xs[s, c, j]
                    if cur_value > cur_max:
                        cur_max = cur_value
                result[s, c, r] = cur_max
    return result


@njit
def _njit_mean_pooling_1d(xs: np.ndarray, result: np.ndarray, kernel_size: Tuple[int], stride: Tuple[int], dilation: int):
    n_samples = xs.shape[0]
    n_channels = xs.shape[1]
    res_len = result.shape[2]
    ks = kernel_size[0]
    st = stride[0]

    for s in range(n_samples):
        for c in range(n_channels):
            for r in range(res_len):
                corner = st * r
                cur_sum = 0.0
                cur_n = 0
                for j in range(corner, corner + ks, dilation):
                    cur_value = xs[s, c, j]
                    cur_sum += cur_value
                    cur_n += 1
                result[s, c, r] = cur_sum / cur_n
    return result


@njit
def _njit_max_pooling_2d(xs: np.ndarray, result: np.ndarray, kernel_size: Tuple[int, int],
                         stride: Tuple[int, int], dilation: int):
    n_samples = xs.shape[0]
    n_channels = xs.shape[1]
    res_height, res_width = result.shape[2], result.shape[3]
    kernel_size_y, kernel_size_x = kernel_size
    stride_y, stride_x = stride

    for s in range(n_samples):
        for c in range(n_channels):
            for ry in range(res_height):
                for rx in range(res_width):
                    corner_y = stride_y * ry
                    corner_x = stride_x * rx
                    cur_max = -np.inf
                    for y in range(corner_y, corner_y + kernel_size_y, dilation):
                        for x in range(corner_x, corner_x + kernel_size_x, dilation):
                            cur_value = xs[s, c, y, x]
                            if cur_value > cur_max:
                                cur_max = cur_value
                    result[s, c, ry, rx] = cur_max
    return result


@njit
def _njit_mean_pooling_2d(xs: np.ndarray, result: np.ndarray, kernel_size: Tuple[int, int],
                          stride: Tuple[int, int], dilation: int):
    n_samples = xs.shape[0]
    n_channels = xs.shape[1]
    res_height, res_width = result.shape[2], result.shape[3]
    kernel_size_y, kernel_size_x = kernel_size
    stride_y, stride_x = stride

    for s in range(n_samples):
        for c in range(n_channels):
            for ry in range(res_height):
                for rx in range(res_width):
                    corner_y = stride_y * ry
                    corner_x = stride_x * rx
                    cur_sum = 0.0
                    cur_n = 0
                    for y in range(corner_y, corner_y + kernel_size_y, dilation):
                        for x in range(corner_x, corner_x + kernel_size_x, dilation):
                            cur_value = xs[s, c, y, x]
                            cur_sum += cur_value
                            cur_n += 1
                    result[s, c, ry, rx] = cur_sum / cur_n
    return result
