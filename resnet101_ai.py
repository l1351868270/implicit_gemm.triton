
import math
import torch
import torch.nn as nn

# import torchvision.models as models
# model = models.resnet101(pretrained=True)
# model.eval()
# print(model)

def conv2d_ai(N, C, H, W, K, R, S, U, V, pad_h, pad_w, dila_h, dila_w, name="", verbose=True):
    # Convolution FLOPS
    KH = R
    KW = S
    P = math.floor((H + 2 * pad_h - dila_h * (R - 1) - 1) / U + 1)
    Q = math.floor((W + 2 * pad_w - dila_w * (S - 1) - 1) / V + 1)
    v_M = N * P * Q
    v_N = K
    v_K = C * R * S

    ops = 2 * v_M * v_N * v_K
    bytes = 2 * (N * C * H * W + K * C * R * S + N * K * P * Q)
    ai = ops / bytes
    v_bytes = 2 * (v_M * v_K + v_N * v_K + v_M * v_N)
    v_ai = ops / v_bytes
    if verbose:
        print(f'[conv2d][{name}] {N}x{C}x{H}x{W} {K}x{C}x{R}x{S} -> {N}x{K}x{P}x{Q} FLOPs: {ops}, Bytes: {bytes}, Arithmetic Intensity: {ai:.5f}--[{v_M}x{v_K} {v_K}x{v_N} {v_M}x{v_N} FLOPs: {ops}, Bytes: {v_bytes}, Arithmetic Intensity: {v_ai:.5f}]')
    MACs = N * C * P * Q * U * V * KH * KW
    return P, Q, ops, bytes, ai

def linear_ai(N, C, K, name="", verbose=True):
    # Linear FLOPS
    ops = 2 * N * C * K
    bytes = 2 * (N * C + N * K + K * C)
    ai = ops / bytes
    if verbose:
        print(f'[linear][{name}] {N}x{C} -> {N}x{K} FLOPs: {ops}, Bytes: {bytes}, Arithmetic Intensity: {ai:.5f}')
    return ai

def resnet101_conv1_ai(N=1, H=224, W=224):
    # (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) 
    return conv2d_ai(N, 3, H, W, 64, 7, 7, 2, 2, 3, 3, 1, 1, name="conv1", verbose=True)
    
def resnet101_ai():
    # (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)
    N = 1
    H = 224
    W = 224
    H, W, _, _, _= resnet101_conv1_ai(N=N, H=H, W=W)
    # (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    pad_h = 1; pad_w = 1; dila_h = 1; dila_w = 1; R = 3; S = 3; U = 2; V = 2
    H = math.floor((H + 2 * pad_h - dila_h * (R - 1) - 1) / U + 1)
    W = math.floor((W + 2 * pad_w - dila_w * (S - 1) - 1) / V + 1)
    H_x = H; W_x = W
    # (layer1): Sequential(
    #     (0): Bottleneck(
    #       (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 64, H, W, 64, 1, 1, 1, 1, 0, 0, 1, 1, name="layer1.0.conv1", verbose=True)
    #       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 64, H, W, 64, 3, 3, 1, 1, 1, 1, 1, 1, name="layer1.0.conv2", verbose=True)
    #       (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 64, H, W, 256, 1, 1, 1, 1, 0, 0, 1, 1, name="layer1.0.conv3", verbose=True)
    #       (downsample): Sequential(
    #         (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H_x, W_x, _, _, _= conv2d_ai(N, 64, H_x, W_x, 256, 1, 1, 1, 1, 0, 0, 1, 1, name="layer1.0.downsample", verbose=True)
    #     (1): Bottleneck(
    #       (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 64, 1, 1, 1, 1, 0, 0, 1, 1, name="layer1.1.conv1", verbose=True)
    #       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 64, H, W, 64, 3, 3, 1, 1, 1, 1, 1, 1, name="layer1.1.conv2", verbose=True)
    #       (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 64, H, W, 256, 1, 1, 1, 1, 0, 0, 1, 1, name="layer1.1.conv3", verbose=True)
    #     (2): Bottleneck(
    #       (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 64, 1, 1, 1, 1, 0, 0, 1, 1, name="layer1.2.conv1", verbose=True)
    #       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 64, H, W, 64, 3, 3, 1, 1, 1, 1, 1, 1, name="layer1.2.conv2", verbose=True)
    #       (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 64, H, W, 256, 1, 1, 1, 1, 0, 0, 1, 1, name="layer1.2.conv3", verbose=True)
    # (layer2): Sequential(
    #     (0): Bottleneck(
    #       (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H_x = H; W_x = W
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 128, 1, 1, 1, 1, 0, 0, 1, 1, name="layer2.0.conv1", verbose=True)
    #       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 128, H, W, 128, 3, 3, 2, 2, 1, 1, 1, 1, name="layer2.0.conv2", verbose=True)
    #       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 128, H, W, 512, 1, 1, 1, 1, 0, 0, 1, 1, name="layer2.0.conv3", verbose=True)
    #       (downsample): Sequential(
    #         (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
    H_x, W_x, _, _, _= conv2d_ai(N, 256, H_x, W_x, 512, 1, 1, 2, 2, 0, 0, 1, 1, name="layer2.0.downsample", verbose=True)
    #     (1): Bottleneck(
    #       (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H_x = H; W_x = W
    H, W, _, _, _= conv2d_ai(N, 512, H, W, 128, 1, 1, 1, 1, 0, 0, 1, 1, name="layer2.1.conv1", verbose=True)
    #       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 128, H, W, 128, 3, 3, 1, 1, 1, 1, 1, 1, name="layer2.1.conv2", verbose=True)
    #       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 128, H, W, 512, 1, 1, 1, 1, 0, 0, 1, 1, name="layer2.1.conv3", verbose=True)
    #     (2): Bottleneck(
    #       (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 512, H, W, 128, 1, 1, 1, 1, 0, 0, 1, 1, name="layer2.2.conv1", verbose=True)
    #       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 128, H, W, 128, 3, 3, 1, 1, 1, 1, 1, 1, name="layer2.2.conv2", verbose=True)
    #       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 128, H, W, 512, 1, 1, 1, 1, 0, 0, 1, 1, name="layer2.2.conv3", verbose=True)
    #     (3): Bottleneck(
    #       (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 512, H, W, 128, 1, 1, 1, 1, 0, 0, 1, 1, name="layer2.3.conv1", verbose=True)
    #       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 128, H, W, 128, 3, 3, 1, 1, 1, 1, 1, 1, name="layer2.3.conv2", verbose=True)
    #       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 128, H, W, 512, 1, 1, 1, 1, 0, 0, 1, 1, name="layer2.3.conv3", verbose=True)
    # (layer3): Sequential(
    #     (0): Bottleneck(
    #       (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H_x = H; W_x = W
    H, W, _, _, _= conv2d_ai(N, 512, H, W, 256, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.0.conv1", verbose=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 256, 3, 3, 2, 2, 1, 1, 1, 1, name="layer3.0.conv2", verbose=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 1024, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.0.conv3", verbose=True)
    #       (downsample): Sequential(
    #         (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
    H_x, W_x, _, _, _= conv2d_ai(N, 512, H_x, W_x, 1024, 1, 1, 2, 2, 0, 0, 1, 1, name="layer3.0.downsample", verbose=True)
    #     (1): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 1024, H, W, 256, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.1.conv1", verbose=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 256, 3, 3, 1, 1, 1, 1, 1, 1, name="layer3.1.conv2", verbose=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 1024, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.1.conv3", verbose=True)
    #     (2): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 1024, H, W, 256, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.2.conv1", verbose=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 256, 3, 3, 1, 1, 1, 1, 1, 1, name="layer3.2.conv2", verbose=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 1024, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.2.conv3", verbose=True)
    #     (3): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 1024, H, W, 256, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.3.conv1", verbose=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 256, 3, 3, 1, 1, 1, 1, 1, 1, name="layer3.3.conv2", verbose=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 1024, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.3.conv3", verbose=True)
    #     (4): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 1024, H, W, 256, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.4.conv1", verbose=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 256, 3, 3, 1, 1, 1, 1, 1, 1, name="layer3.4.conv2", verbose=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 1024, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.4.conv3", verbose=True)
    #     (5): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 1024, H, W, 256, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.5.conv1", verbose=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 256, 3, 3, 1, 1, 1, 1, 1, 1, name="layer3.5.conv2", verbose=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 1024, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.5.conv3", verbose=True)
    #     (6): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 1024, H, W, 256, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.6.conv1", verbose=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 256, 3, 3, 1, 1, 1, 1, 1, 1, name="layer3.6.conv2", verbose=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 1024, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.6.conv3", verbose=True)
    #     (7): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 1024, H, W, 256, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.7.conv1", verbose=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 256, 3, 3, 1, 1, 1, 1, 1, 1, name="layer3.7.conv2", verbose=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 1024, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.7.conv3", verbose=True)
    #     (8): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 1024, H, W, 256, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.8.conv1", verbose=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 256, 3, 3, 1, 1, 1, 1, 1, 1, name="layer3.8.conv2", verbose=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 1024, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.8.conv3", verbose=True)
    #     (9): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 1024, H, W, 256, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.9.conv1", verbose=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 256, 3, 3, 1, 1, 1, 1, 1, 1, name="layer3.9.conv2", verbose=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 1024, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.9.conv3", verbose=True)
    #     (10): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 1024, H, W, 256, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.10.conv1", verbose=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 256, 3, 3, 1, 1, 1, 1, 1, 1, name="layer3.10.conv2", verbose=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 1024, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.10.conv3", verbose=True)
    #     (11): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 1024, H, W, 256, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.11.conv1", verbose=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 256, 3, 3, 1, 1, 1, 1, 1, 1, name="layer3.11.conv2", verbose=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 1024, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.11.conv3", verbose=True)
    #     (12): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 1024, H, W, 256, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.12.conv1", verbose=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 256, 3, 3, 1, 1, 1, 1, 1, 1, name="layer3.12.conv2", verbose=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 1024, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.12.conv3", verbose=True)
    #     (13): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 1024, H, W, 256, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.13.conv1", verbose=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 256, 3, 3, 1, 1, 1, 1, 1, 1, name="layer3.13.conv2", verbose=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 1024, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.13.conv3", verbose=True)
    #     (14): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 1024, H, W, 256, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.14.conv1", verbose=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 256, 3, 3, 1, 1, 1, 1, 1, 1, name="layer3.14.conv2", verbose=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 1024, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.14.conv3", verbose=True)
    #     (15): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 1024, H, W, 256, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.15.conv1", verbose=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 256, 3, 3, 1, 1, 1, 1, 1, 1, name="layer3.15.conv2", verbose=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 1024, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.15.conv3", verbose=True)
    #     (16): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 1024, H, W, 256, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.16.conv1", verbose=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 256, 3, 3, 1, 1, 1, 1, 1, 1, name="layer3.16.conv2", verbose=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 1024, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.16.conv3", verbose=True)
    #     (17): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 1024, H, W, 256, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.17.conv1", verbose=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 256, 3, 3, 1, 1, 1, 1, 1, 1, name="layer3.17.conv2", verbose=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 1024, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.17.conv3", verbose=True)
    #     (18): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 1024, H, W, 256, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.18.conv1", verbose=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 256, 3, 3, 1, 1, 1, 1, 1, 1, name="layer3.18.conv2", verbose=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 1024, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.18.conv3", verbose=True)
    #     (19): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 1024, H, W, 256, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.19.conv1", verbose=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 256, 3, 3, 1, 1, 1, 1, 1, 1, name="layer3.19.conv2", verbose=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 1024, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.19.conv3", verbose=True)
    #     (20): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 1024, H, W, 256, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.20.conv1", verbose=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 256, 3, 3, 1, 1, 1, 1, 1, 1, name="layer3.20.conv2", verbose=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 1024, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.20.conv3", verbose=True)
    #     (21): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 1024, H, W, 256, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.21.conv1", verbose=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 256, 3, 3, 1, 1, 1, 1, 1, 1, name="layer3.21.conv2", verbose=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 1024, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.21.conv3", verbose=True)
    #     (22): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 1024, H, W, 256, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.22.conv1", verbose=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 256, 3, 3, 1, 1, 1, 1, 1, 1, name="layer3.22.conv2", verbose=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 256, H, W, 1024, 1, 1, 1, 1, 0, 0, 1, 1, name="layer3.22.conv3", verbose=True)
    # (layer4): Sequential(
    #     (0): Bottleneck(
    #       (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H_x = H; W_x = W
    H, W, _, _, _= conv2d_ai(N, 1024, H, W, 512, 1, 1, 1, 1, 0, 0, 1, 1, name="layer4.0.conv1", verbose=True)
    #       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 512, H, W, 512, 3, 3, 2, 2, 1, 1, 1, 1, name="layer4.0.conv2", verbose=True)
    #       (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 512, H, W, 2048, 1, 1, 1, 1, 0, 0, 1, 1, name="layer4.0.conv3", verbose=True)
    #       (downsample): Sequential(
    #         (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
    H_x, W_x, _, _, _= conv2d_ai(N, 1024, H_x, W_x, 2048, 1, 1, 2, 2, 0, 0, 1, 1, name="layer4.0.downsample", verbose=True)
    #     (1): Bottleneck(
    #       (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 2048, H, W, 512, 1, 1, 1, 1, 0, 0, 1, 1, name="layer4.1.conv1", verbose=True)
    #       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 512, H, W, 512, 3, 3, 1, 1, 1, 1, 1, 1, name="layer4.1.conv2", verbose=True)
    #       (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 512, H, W, 2048, 1, 1, 1, 1, 0, 0, 1, 1, name="layer4.1.conv3", verbose=True)
    #     (2): Bottleneck(
    #       (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 2048, H, W, 512, 1, 1, 1, 1, 0, 0, 1, 1, name="layer4.2.conv1", verbose=True)
    #       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 512, H, W, 512, 3, 3, 1, 1, 1, 1, 1, 1, name="layer4.2.conv2", verbose=True)
    #       (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
    H, W, _, _, _= conv2d_ai(N, 512, H, W, 2048, 1, 1, 1, 1, 0, 0, 1, 1, name="layer4.2.conv3", verbose=True)
    # (fc): Linear(in_features=2048, out_features=1000, bias=True)
    _ = linear_ai(N, 2048, 1000, name="fc", verbose=True)


if __name__ == '__main__':
    flops_3090 = 1695 * 10 ** 6 * 10496 * 2 * 2 # tensor core accumulate float32
    bandwidth_3090 = 936 * 1024 * 1024 * 1024
    print(f'rtx 3090 ai: TFLOPs:{flops_3090}, bandwidth:{bandwidth_3090}, {flops_3090 / bandwidth_3090:.4f}, FLOAPS/cycle:{10496 * 2 * 2}')
    resnet101_ai()

