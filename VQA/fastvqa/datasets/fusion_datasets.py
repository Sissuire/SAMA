import decord
from decord import VideoReader
from decord import cpu, gpu
import glob
import os.path as osp
import numpy as np
import torch, torchvision
from tqdm import tqdm
import cv2
import os
from functools import lru_cache

import random
import copy

import skvideo.io

random.seed(42)

decord.bridge.set_bridge("torch")


def get_spatial_sama_spm_fragments(
    video,
    is_train=False,
    fragments_h=7,
    fragments_w=7,
    fsize_h=32,
    fsize_w=32,
    aligned=32,
    nfrags=1,
    random=False,
    random_upsample=False,
    fallback_type="upsample",
    **kwargs,
):
    size_h = fragments_h * fsize_h
    size_w = fragments_w * fsize_w
    ## video: [C,T,H,W]
    ## situation for images
    if video.shape[1] == 1:
        aligned = 1

    dur_t, res_h, res_w = video.shape[-3:]
    ratio = min(res_h / size_h, res_w / size_w)
    multiply = 1 if is_train else 4   # 32 for training, 4*32 for testing
    step = aligned // 2   # sample by step 2

    video = video / 255.
    if fallback_type == "upsample" and ratio < 1:
        
        # ovideo = video
        res_h, res_w = round(res_h / ratio), round(res_w / ratio)
        video = torch.nn.functional.interpolate(video, size=(res_h, res_w), mode="bilinear", align_corners=False)
        # video = (video * 255.0).type_as(ovideo)
        factors = [1,] * 16 * multiply
    else:
        factors = list(np.linspace(1, 1 / ratio, 16)) * multiply
    
    assert dur_t % aligned == 0, "Please provide match vclip and align index"
    size = size_h, size_w

    ## make sure that sampling will not run out of the picture     
    img_scale, hgrids, wgrids = [], [], []
    rnd_h, rnd_w = [], []
    if is_train:
        rnd_rh, rnd_rw = torch.rand((fragments_h, fragments_w, dur_t // aligned)), torch.rand((fragments_h, fragments_w, dur_t // aligned))
    else:
        rnd_rh, rnd_rw = torch.ones((fragments_h, fragments_w, dur_t // aligned)) * 0.5, torch.ones((fragments_h, fragments_w, dur_t // aligned)) * 0.5

    for i, scale in enumerate(factors):
        this_h, this_w = round(res_h * scale), round(res_w * scale)
        img_scale.append(255. * torch.nn.functional.interpolate(video[:, 2*i:2*(i+1)], size=(this_h, this_w), mode="bilinear", align_corners=False))

        hgrids.append(torch.LongTensor([min(this_h // fragments_h * i, this_h - fsize_h) for i in range(fragments_h)]))
        wgrids.append(torch.LongTensor([min(this_w // fragments_w * i, this_w - fsize_w) for i in range(fragments_w)]))

        hlength, wlength = this_h // fragments_h, this_w // fragments_w
        rnd_h.append((rnd_rh[:, :, i // step] * (hlength - fsize_h)).int())
        rnd_w.append((rnd_rw[:, :, i // step] * (wlength - fsize_w)).int())

    target_scale_video = torch.zeros(video.shape[:-2] + size).to(video.device)
    target_fullr_video = torch.zeros(video.shape[:-2] + size).to(video.device)
    for k, scale in enumerate(factors):
        for i, hs in enumerate(hgrids[k]):
            for j, ws in enumerate(wgrids[k]):
                h_s, h_e = i * fsize_h, (i + 1) * fsize_h
                w_s, w_e = j * fsize_w, (j + 1) * fsize_w

                h_so = hs + rnd_h[k][i][j]
                h_eo = h_so + fsize_h
                w_so = ws + rnd_w[k][i][j]
                w_eo = w_so + fsize_w
                target_scale_video[:, 2*k:2*(k+1), h_s:h_e, w_s:w_e] = img_scale[k][:, :, h_so:h_eo, w_so:w_eo]  # 32 * 32

    for i, hs in enumerate(hgrids[0]):
        for j, ws in enumerate(wgrids[0]):
            h_s, h_e = i * fsize_h, (i + 1) * fsize_h
            w_s, w_e = j * fsize_w, (j + 1) * fsize_w

            h_so = hs + rnd_h[0][i][j]
            h_eo = h_so + fsize_h
            w_so = ws + rnd_w[0][i][j]
            w_eo = w_so + fsize_w
            target_fullr_video[:, :, h_s:h_e, w_s:w_e] = 255 * video[:, :, h_so:h_eo, w_so:w_eo]

    # patch-based mask [4, 4]
    mask = torch.zeros((1, 1, size_h, size_w))
    for i in range(size_w // 8):  # patch 
        for j in range(size_h // 8):
            mask[:, :, j*8:j*8+4, i*8:i*8+4] = 1
            mask[:, :, j*8+4:j*8+8, i*8+4:i*8+8] = 1

    target_video = mask * target_fullr_video + (1 - mask) * target_scale_video
    return target_video

def get_spatial_sama_swm_fragments(
    video,
    is_train=False,
    fragments_h=7,
    fragments_w=7,
    fsize_h=32,
    fsize_w=32,
    aligned=32,
    nfrags=1,
    random=False,
    random_upsample=False,
    fallback_type="upsample",
    **kwargs,
):
    size_h = fragments_h * fsize_h
    size_w = fragments_w * fsize_w
    ## video: [C,T,H,W]
    ## situation for images
    if video.shape[1] == 1:
        aligned = 1

    dur_t, res_h, res_w = video.shape[-3:]
    ratio = min(res_h / size_h, res_w / size_w)
    multiply = 1 if is_train else 4   # 32 for training, 4*32 for testing
    step = aligned // 2   # sample by step 2

    video = video / 255.
    if fallback_type == "upsample" and ratio < 1:
        
        # ovideo = video
        res_h, res_w = round(res_h / ratio), round(res_w / ratio)
        video = torch.nn.functional.interpolate(video, size=(res_h, res_w), mode="bilinear", align_corners=False)
        # video = (video * 255.0).type_as(ovideo)
        factors = [1,] * 16 * multiply
    else:
        factors = list(np.linspace(1, 1 / ratio, 16)) * multiply
    
    assert dur_t % aligned == 0, "Please provide match vclip and align index"
    size = size_h, size_w

    ## make sure that sampling will not run out of the picture     
    img_scale, hgrids, wgrids = [], [], []
    rnd_h, rnd_w = [], []
    if is_train:
        rnd_rh, rnd_rw = torch.rand((fragments_h, fragments_w, dur_t // aligned)), torch.rand((fragments_h, fragments_w, dur_t // aligned))
    else:
        rnd_rh, rnd_rw = torch.ones((fragments_h, fragments_w, dur_t // aligned)) * 0.5, torch.ones((fragments_h, fragments_w, dur_t // aligned)) * 0.5

    for i, scale in enumerate(factors):
        this_h, this_w = round(res_h * scale), round(res_w * scale)
        img_scale.append(255. * torch.nn.functional.interpolate(video[:, 2*i:2*(i+1)], size=(this_h, this_w), mode="bilinear", align_corners=False))

        hgrids.append(torch.LongTensor([min(this_h // fragments_h * i, this_h - fsize_h) for i in range(fragments_h)]))
        wgrids.append(torch.LongTensor([min(this_w // fragments_w * i, this_w - fsize_w) for i in range(fragments_w)]))

        hlength, wlength = this_h // fragments_h, this_w // fragments_w
        rnd_h.append((rnd_rh[:, :, i // step] * (hlength - fsize_h)).int())
        rnd_w.append((rnd_rw[:, :, i // step] * (wlength - fsize_w)).int())

    target_scale_video = torch.zeros(video.shape[:-2] + size).to(video.device)
    target_fullr_video = torch.zeros(video.shape[:-2] + size).to(video.device)
    for k, scale in enumerate(factors):
        for i, hs in enumerate(hgrids[k]):
            for j, ws in enumerate(wgrids[k]):
                h_s, h_e = i * fsize_h, (i + 1) * fsize_h
                w_s, w_e = j * fsize_w, (j + 1) * fsize_w

                h_so = hs + rnd_h[k][i][j]
                h_eo = h_so + fsize_h
                w_so = ws + rnd_w[k][i][j]
                w_eo = w_so + fsize_w
                target_scale_video[:, 2*k:2*(k+1), h_s:h_e, w_s:w_e] = img_scale[k][:, :, h_so:h_eo, w_so:w_eo]  # 32 * 32

    for i, hs in enumerate(hgrids[0]):
        for j, ws in enumerate(wgrids[0]):
            h_s, h_e = i * fsize_h, (i + 1) * fsize_h
            w_s, w_e = j * fsize_w, (j + 1) * fsize_w

            h_so = hs + rnd_h[0][i][j]
            h_eo = h_so + fsize_h
            w_so = ws + rnd_w[0][i][j]
            w_eo = w_so + fsize_w
            target_fullr_video[:, :, h_s:h_e, w_s:w_e] = 255 * video[:, :, h_so:h_eo, w_so:w_eo]

    # window-based mask [32, 32]
    mask = torch.zeros((1, 1, size_h, size_w))
    for i in range(fragments_h):  # window
        for j in range(fragments_w):
            if (i + j) % 2 == 0:
                mask[:, :, j*32:j*32+32, i*32:i*32+32] = 1

    target_video = mask * target_fullr_video + (1 - mask) * target_scale_video
    return target_video

def get_spatial_sama_fragments(
    video,
    is_train=False,
    fragments_h=7,
    fragments_w=7,
    fsize_h=32,
    fsize_w=32,
    aligned=32,
    nfrags=1,
    random=False,
    random_upsample=False,
    fallback_type="upsample",
    **kwargs,
):
    size_h = fragments_h * fsize_h
    size_w = fragments_w * fsize_w
    ## video: [C,T,H,W]
    ## situation for images
    if video.shape[1] == 1:
        aligned = 1

    dur_t, res_h, res_w = video.shape[-3:]
    ratio = min(res_h / size_h, res_w / size_w)
    multiply = 1 if is_train else 4   # 32 for training, 4*32 for testing
    step = aligned // 2   # sample by step 2

    video = video / 255.
    if fallback_type == "upsample" and ratio < 1:
        
        # ovideo = video
        res_h, res_w = round(res_h / ratio), round(res_w / ratio)
        video = torch.nn.functional.interpolate(video, size=(res_h, res_w), mode="bilinear", align_corners=False)
        # video = (video * 255.0).type_as(ovideo)
        factors = [1,] * 16 * multiply
    else:
        factors = list(np.linspace(1, 1 / ratio, 16)) * multiply
    
    assert dur_t % aligned == 0, "Please provide match vclip and align index"
    size = size_h, size_w

    ## make sure that sampling will not run out of the picture     
    img_scale, hgrids, wgrids = [], [], []
    rnd_h, rnd_w = [], []
    if is_train:
        rnd_rh, rnd_rw = torch.rand((fragments_h, fragments_w, dur_t // aligned)), torch.rand((fragments_h, fragments_w, dur_t // aligned))
    else:
        rnd_rh, rnd_rw = torch.ones((fragments_h, fragments_w, dur_t // aligned)) * 0.5, torch.ones((fragments_h, fragments_w, dur_t // aligned)) * 0.5

    for i, scale in enumerate(factors):
        this_h, this_w = round(res_h * scale), round(res_w * scale)
        img_scale.append(255. * torch.nn.functional.interpolate(video[:, 2*i:2*(i+1)], size=(this_h, this_w), mode="bilinear", align_corners=False))

        hgrids.append(torch.LongTensor([min(this_h // fragments_h * i, this_h - fsize_h) for i in range(fragments_h)]))
        wgrids.append(torch.LongTensor([min(this_w // fragments_w * i, this_w - fsize_w) for i in range(fragments_w)]))

        hlength, wlength = this_h // fragments_h, this_w // fragments_w
        rnd_h.append((rnd_rh[:, :, i // step] * (hlength - fsize_h)).int())
        rnd_w.append((rnd_rw[:, :, i // step] * (wlength - fsize_w)).int())

    target_video = torch.zeros(video.shape[:-2] + size).to(video.device)
    for k, scale in enumerate(factors):
        for i, hs in enumerate(hgrids[k]):
            for j, ws in enumerate(wgrids[k]):
                h_s, h_e = i * fsize_h, (i + 1) * fsize_h
                w_s, w_e = j * fsize_w, (j + 1) * fsize_w

                h_so = hs + rnd_h[k][i][j]
                h_eo = h_so + fsize_h
                w_so = ws + rnd_w[k][i][j]
                w_eo = w_so + fsize_w
                target_video[:, 2*k:2*(k+1), h_s:h_e, w_s:w_e] = img_scale[k][:, :, h_so:h_eo, w_so:w_eo]  # 32 * 32

    return target_video

def get_spatial_sama_mix_fragments(
    video,
    is_train=False,
    fragments_h=7,
    fragments_w=7,
    fsize_h=32,
    fsize_w=32,
    aligned=32,
    nfrags=1,
    random=False,
    random_upsample=False,
    fallback_type="upsample",
    **kwargs,
):
    size_h = fragments_h * fsize_h
    size_w = fragments_w * fsize_w
    ## video: [C,T,H,W]
    ## situation for images
    if video.shape[1] == 1:
        aligned = 1

    dur_t, res_h, res_w = video.shape[-3:]
    ratio = min(res_h / size_h, res_w / size_w)
    multiply = 1 if is_train else 4   # 32 for training, 4*32 for testing
    step = aligned // 2   # sample by step 2

    video = video / 255.
    if fallback_type == "upsample" and ratio < 1:
        
        # ovideo = video
        res_h, res_w = round(res_h / ratio), round(res_w / ratio)
        video = torch.nn.functional.interpolate(video, size=(res_h, res_w), mode="bilinear", align_corners=False)
        # video = (video * 255.0).type_as(ovideo)
        factors = [1,] * 8 * 2 * multiply
    else:
        factors = list(np.linspace(1, 1 / ratio, 8)) * 2 * multiply
    
    assert dur_t % aligned == 0, "Please provide match vclip and align index"
    size = size_h, size_w

    ## make sure that sampling will not run out of the picture     
    img_scale, hgrids, wgrids = [], [], []
    rnd_h, rnd_w = [], []
    if is_train:
        rnd_rh, rnd_rw = torch.rand((fragments_h, fragments_w, dur_t // aligned)), torch.rand((fragments_h, fragments_w, dur_t // aligned))
    else:
        rnd_rh, rnd_rw = torch.ones((fragments_h, fragments_w, dur_t // aligned)) * 0.5, torch.ones((fragments_h, fragments_w, dur_t // aligned)) * 0.5

    for i, scale in enumerate(factors):
        this_h, this_w = round(res_h * scale), round(res_w * scale)
        img_scale.append(255. * torch.nn.functional.interpolate(video[:, 2*i:2*(i+1)], size=(this_h, this_w), mode="bilinear", align_corners=False))

        hgrids.append(torch.LongTensor([min(this_h // fragments_h * i, this_h - fsize_h) for i in range(fragments_h)]))
        wgrids.append(torch.LongTensor([min(this_w // fragments_w * i, this_w - fsize_w) for i in range(fragments_w)]))

        hlength, wlength = this_h // fragments_h, this_w // fragments_w
        rnd_h.append((rnd_rh[:, :, i // step] * (hlength - fsize_h)).int())
        rnd_w.append((rnd_rw[:, :, i // step] * (wlength - fsize_w)).int())

    target_video = torch.zeros(video.shape[:-2] + size).to(video.device)
    for k, scale in enumerate(factors):
        for i, hs in enumerate(hgrids[k]):
            for j, ws in enumerate(wgrids[k]):
                h_s, h_e = i * fsize_h, (i + 1) * fsize_h
                w_s, w_e = j * fsize_w, (j + 1) * fsize_w

                h_so = hs + rnd_h[k][i][j]
                h_eo = h_so + fsize_h
                w_so = ws + rnd_w[k][i][j]
                w_eo = w_so + fsize_w
                target_video[:, 2*k:2*(k+1), h_s:h_e, w_s:w_e] = img_scale[k][:, :, h_so:h_eo, w_so:w_eo]  # 32 * 32

    return target_video

def get_spatial_sama_c_fragments(
    video,
    is_train=False,
    fragments_h=7,
    fragments_w=7,
    fsize_h=32,
    fsize_w=32,
    aligned=32,
    nfrags=1,
    random=False,
    random_upsample=False,
    fallback_type="upsample",
    **kwargs,
):
    size_h = fragments_h * fsize_h
    size_w = fragments_w * fsize_w
    ## video: [C,T,H,W]
    ## situation for images
    if video.shape[1] == 1:
        aligned = 1

    dur_t, res_h, res_w = video.shape[-3:]
    ratio = min(res_h / size_h, res_w / size_w)
    multiply = 1 if is_train else 4   # 32 for training, 4*32 for testing
    step = aligned // 2   # sample by step 2

    video = video / 255.
    if fallback_type == "upsample" and ratio < 1:
        
        # ovideo = video
        res_h, res_w = round(res_h / ratio), round(res_w / ratio)
        video = torch.nn.functional.interpolate(video, size=(res_h, res_w), mode="bilinear", align_corners=False)
        # video = (video * 255.0).type_as(ovideo)
        factors = [1,] * 16 * multiply
    else:
        factors = [1, 1 / ratio] * 8 * multiply
    
    assert dur_t % aligned == 0, "Please provide match vclip and align index"
    size = size_h, size_w

    ## make sure that sampling will not run out of the picture     
    img_scale, hgrids, wgrids = [], [], []
    rnd_h, rnd_w = [], []
    if is_train:
        rnd_rh, rnd_rw = torch.rand((fragments_h, fragments_w, dur_t // aligned)), torch.rand((fragments_h, fragments_w, dur_t // aligned))
    else:
        rnd_rh, rnd_rw = torch.ones((fragments_h, fragments_w, dur_t // aligned)) * 0.5, torch.ones((fragments_h, fragments_w, dur_t // aligned)) * 0.5

    for i, scale in enumerate(factors):
        this_h, this_w = round(res_h * scale), round(res_w * scale)
        img_scale.append(255. * torch.nn.functional.interpolate(video[:, 2*i:2*(i+1)], size=(this_h, this_w), mode="bilinear", align_corners=False))

        hgrids.append(torch.LongTensor([min(this_h // fragments_h * i, this_h - fsize_h) for i in range(fragments_h)]))
        wgrids.append(torch.LongTensor([min(this_w // fragments_w * i, this_w - fsize_w) for i in range(fragments_w)]))

        hlength, wlength = this_h // fragments_h, this_w // fragments_w
        rnd_h.append((rnd_rh[:, :, i // step] * (hlength - fsize_h)).int())
        rnd_w.append((rnd_rw[:, :, i // step] * (wlength - fsize_w)).int())

    target_video = torch.zeros(video.shape[:-2] + size).to(video.device)
    for k, scale in enumerate(factors):
        for i, hs in enumerate(hgrids[k]):
            for j, ws in enumerate(wgrids[k]):
                h_s, h_e = i * fsize_h, (i + 1) * fsize_h
                w_s, w_e = j * fsize_w, (j + 1) * fsize_w

                h_so = hs + rnd_h[k][i][j]
                h_eo = h_so + fsize_h
                w_so = ws + rnd_w[k][i][j]
                w_eo = w_so + fsize_w
                target_video[:, 2*k:2*(k+1), h_s:h_e, w_s:w_e] = img_scale[k][:, :, h_so:h_eo, w_so:w_eo]  # 32 * 32

    return target_video

def get_spatial_fragments(
    video,
    is_train=False,
    fragments_h=7,
    fragments_w=7,
    fsize_h=32,
    fsize_w=32,
    aligned=32,
    nfrags=1,
    random=False,
    random_upsample=False,
    fallback_type="upsample",
    **kwargs,
):
    size_h = fragments_h * fsize_h
    size_w = fragments_w * fsize_w
    ## video: [C,T,H,W]
    ## situation for images
    if video.shape[1] == 1:
        aligned = 1

    dur_t, res_h, res_w = video.shape[-3:]
    ratio = min(res_h / size_h, res_w / size_w)
    if fallback_type == "upsample" and ratio < 1:
        
        ovideo = video
        res_h, res_w = round(res_h / ratio), round(res_w / ratio)
        video = torch.nn.functional.interpolate(video / 255.0, size=(res_h, res_w), mode="bilinear", align_corners=False)
        video = (video * 255.0).type_as(ovideo)
    
    assert dur_t % aligned == 0, "Please provide match vclip and align index"
    size = size_h, size_w

    ## make sure that sampling will not run out of the picture
    hlength, wlength = res_h // fragments_h, res_w // fragments_w
    hgrids = torch.LongTensor([min(hlength * i, res_h - fsize_h) for i in range(fragments_h)])
    wgrids = torch.LongTensor([min(wlength * i, res_w - fsize_w) for i in range(fragments_w)])

    if is_train:
        rnd_h = torch.randint(max(1, hlength - fsize_h), (fragments_h, fragments_w, dur_t // aligned))
        rnd_w = torch.randint(max(1, wlength - fsize_w), (fragments_h, fragments_w, dur_t // aligned))
    else:
        rnd_h = torch.zeros((fragments_h, fragments_w, dur_t // aligned)).int() + (hlength - fsize_h) // 2
        rnd_w = torch.zeros((fragments_h, fragments_w, dur_t // aligned)).int() + (wlength - fsize_w) // 2

    assert rnd_h[0][0][0] > -1 and rnd_w[0][0][0] > -1
    target_video = torch.zeros(video.shape[:-2] + size).to(video.device)
    # target_videos = []

    for i, hs in enumerate(hgrids):
        for j, ws in enumerate(wgrids):
            for t in range(dur_t // aligned):
                t_s, t_e = t * aligned, (t + 1) * aligned
                h_s, h_e = i * fsize_h, (i + 1) * fsize_h
                w_s, w_e = j * fsize_w, (j + 1) * fsize_w

                h_so, h_eo = hs + rnd_h[i][j][t], hs + rnd_h[i][j][t] + fsize_h
                w_so, w_eo = ws + rnd_w[i][j][t], ws + rnd_w[i][j][t] + fsize_w
                target_video[:, t_s:t_e, h_s:h_e, w_s:w_e] = video[:, t_s:t_e, h_so:h_eo, w_so:w_eo]
    return target_video

@lru_cache
def get_resize_function(size_h, size_w, target_ratio=1, random_crop=False):
    if random_crop:
        return torchvision.transforms.RandomResizedCrop((size_h, size_w), scale=(0.40,1.0))
    if target_ratio > 1:
        size_h = int(target_ratio * size_w)
        assert size_h > size_w
    elif target_ratio < 1:
        size_w = int(size_h / target_ratio)
        assert size_w > size_h
    return torchvision.transforms.Resize((size_h, size_w))

def get_resized_video(
    video,
    size_h=224,
    size_w=224,
    random_crop=False,
    arp=False,
    **kwargs,
):
    video = video.permute(1,0,2,3)
    resize_opt = get_resize_function(size_h, size_w, 
                                     video.shape[-2] / video.shape[-1] if arp else 1,
                                     random_crop)
    video = resize_opt(video).permute(1,0,2,3)
    return video

def get_arp_resized_video(
    video,
    short_edge=224,
    train=False,
    **kwargs,
):
    if train: ## if during training, will random crop into square and then resize
        res_h, res_w = video.shape[-2:]
        ori_short_edge = min(video.shape[-2:])
        if res_h > ori_short_edge:
            rnd_h = random.randrange(res_h - ori_short_edge)
            video = video[...,rnd_h:rnd_h+ori_short_edge,:]
        elif res_w > ori_short_edge:
            rnd_w = random.randrange(res_w - ori_short_edge)
            video = video[...,:,rnd_h:rnd_h+ori_short_edge]
    ori_short_edge = min(video.shape[-2:])
    scale_factor = short_edge / ori_short_edge
    ovideo = video
    video = torch.nn.functional.interpolate(
        video / 255.0, scale_factors=scale_factor, mode="bilinear"
    )
    video = (video * 255.0).type_as(ovideo)
    return video

def get_arp_fragment_video(
    video,
    short_fragments=7,
    fsize=32,
    train=False,
    **kwargs,
):
    if train: ## if during training, will random crop into square and then get fragments
        res_h, res_w = video.shape[-2:]
        ori_short_edge = min(video.shape[-2:])
        if res_h > ori_short_edge:
            rnd_h = random.randrange(res_h - ori_short_edge)
            video = video[...,rnd_h:rnd_h+ori_short_edge,:]
        elif res_w > ori_short_edge:
            rnd_w = random.randrange(res_w - ori_short_edge)
            video = video[...,:,rnd_h:rnd_h+ori_short_edge]
    kwargs["fsize_h"], kwargs["fsize_w"] = fsize, fsize
    res_h, res_w = video.shape[-2:]
    if res_h > res_w:
        kwargs["fragments_w"] = short_fragments
        kwargs["fragments_h"] = int(short_fragments * res_h / res_w)
    else:
        kwargs["fragments_h"] = short_fragments
        kwargs["fragments_w"] = int(short_fragments * res_w / res_h)
    return get_spatial_fragments(video, **kwargs)
        
def get_cropped_video(
    video,
    size_h=224,
    size_w=224,
    **kwargs,
):
    kwargs["fragments_h"], kwargs["fragments_w"] = 1, 1
    kwargs["fsize_h"], kwargs["fsize_w"] = size_h, size_w
    return get_spatial_fragments(video, **kwargs)

def get_single_sample(
    video,
    sample_type="resize",
    is_train=False, 
    **kwargs,
):
    if sample_type.startswith("resize"):
        video = get_resized_video(video, **kwargs)
    elif sample_type.startswith("arp_resize"):
        video = get_arp_resized_video(video, **kwargs)
    elif sample_type.startswith("fragments"):
        video = get_spatial_fragments(video, is_train, **kwargs)
    elif sample_type == "sama":
        video = get_spatial_sama_fragments(video, is_train, **kwargs)
    elif sample_type == "sama-spm":
        video = get_spatial_sama_spm_fragments(video, is_train, **kwargs)
    elif sample_type == "sama-swm":
        video = get_spatial_sama_swm_fragments(video, is_train, **kwargs)
    elif sample_type == "sama-mix":
        video = get_spatial_sama_mix_fragments(video, is_train, **kwargs)
    elif sample_type == "sama-c":
        video = get_spatial_sama_c_fragments(video, is_train, **kwargs)
    elif sample_type.startswith("arp_fragments"):
        video = get_arp_fragment_video(video, **kwargs)
    elif sample_type.startswith("crop"):
        video = get_cropped_video(video, **kwargs)
    elif sample_type == "original":
        return video
        
    return video

def get_spatial_samples(
    video,
    random_crop=0, ## 1: ARP-kept Crop; 2: Square-like Crop
    sample_types={"resize": {}, "fragments": {}}, ## resize | arp_resize | crop | fragments
):
    if random_crop == 1:
        print("Alert!")
        ## Random Crop but keep the ARP
        res_h, res_w = video.shape[-2:]
        rnd_ratio = random.random() * 0.2 + 0.8
        new_h, new_w = int(rnd_ratio * res_h), int(rnd_ratio * res_w)
        rnd_h = random.randrange(res_h - new_h)
        rnd_w = random.randrange(res_w - new_w)
        video = video[..., rnd_h:rnd_h+new_h, rnd_w:rnd_w+new_w]
        ovideo = video
        video = torch.nn.functional.interpolate(
            video / 255.0, scale_factor=random.random() * 0.3 + 1.0, mode="bilinear"
        )
        video = (video * 255.0).type_as(ovideo)
        
    if random_crop == 2:
        ## Random Crop into a Size similar to Square
        rnd_ratio = random.random() * 0.2 + 0.8
        res_h, res_w = video.shape[-2:]
        new_h = new_w = int(rnd_ratio * min(res_h, res_w))
        rnd_h = random.randrange(res_h - new_h)
        rnd_w = random.randrange(res_w - new_w)
        video = video[..., rnd_h:rnd_h+new_h, rnd_w:rnd_w+new_w]
    sampled_video = {}
    for sample_type, arg in sample_types.items():
        sampled_video[sample_type] = get_single_sample(video, sample_type, 
                                                       **arg)
    return sampled_video

def get_spatial_and_temporal_samples(
    video_path,
    sample_types,
    samplers,
    stypescale='fragments', 
    is_train=False,
    augment=False,
):
    video = {}
    if video_path.endswith(".yuv"):
        print("This part will be deprecated due to large memory cost.")
        ## This is only an adaptation to LIVE-Qualcomm
        ovideo = skvideo.io.vread(video_path, 1080, 1920, inputdict={'-pix_fmt':'yuvj420p'})
        for stype in samplers:
            frame_inds = samplers[stype](ovideo.shape[0], is_train)
            imgs = [torch.from_numpy(ovideo[idx]) for idx in frame_inds]
            video[stype] = torch.stack(imgs, 0).permute(3, 0, 1, 2)
        del ovideo
    else:
        vreader = VideoReader(video_path)
        ### Avoid duplicated video decoding!!! Important!!!!
        all_frame_inds = []
        frame_inds = {}
        for stype in samplers:
            frame_inds[stype] = samplers[stype](len(vreader), is_train)
            all_frame_inds.append(frame_inds[stype])
            
        ### Each frame is only decoded one time!!!
        all_frame_inds = np.concatenate(all_frame_inds,0)
        frame_dict = {idx: vreader[idx] for idx in np.unique(all_frame_inds)}
        
        for stype in samplers:
            imgs = [frame_dict[idx] for idx in frame_inds[stype]]
            video[stype] = torch.stack(imgs, 0).permute(3, 0, 1, 2)

    sampled_video = {}
    for stype, sopt in sample_types.items():
        sampled_video[stype] = get_single_sample(video[stype], stypescale, is_train, 
                                                       **sopt)
    return sampled_video, frame_inds
        
class SampleFrames:
    def __init__(self, clip_len, frame_interval=1, num_clips=1):

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips

    def _get_train_clips(self, num_frames):
        """Get clip offsets in train mode.

        It will calculate the average interval for selected frames,
        and randomly shift them within offsets between [0, avg_interval].
        If the total number of frames is smaller than clips num or origin
        frames length, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips

        if avg_interval > 0:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = base_offsets + np.random.randint(
                avg_interval, size=self.num_clips
            )
        elif num_frames > max(self.num_clips, ori_clip_len):
            clip_offsets = np.sort(
                np.random.randint(num_frames - ori_clip_len + 1, size=self.num_clips)
            )
        elif avg_interval == 0:
            ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
            clip_offsets = np.around(np.arange(self.num_clips) * ratio)
        else:
            clip_offsets = np.zeros((self.num_clips,), dtype=np.int)
        return clip_offsets

    def _get_test_clips(self, num_frames, start_index=0):
        """Get clip offsets in test mode.

        Calculate the average interval for selected frames, and shift them
        fixedly by avg_interval/2.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) / float(self.num_clips)
        if num_frames > ori_clip_len - 1:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = (base_offsets + avg_interval / 2.0).astype(np.int32)
        else:
            clip_offsets = np.zeros((self.num_clips,), dtype=np.int32)
        return clip_offsets

    def __call__(self, total_frames, train=False, start_index=0):
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if train:
            clip_offsets = self._get_train_clips(total_frames)
        else:
            clip_offsets = self._get_test_clips(total_frames)
        frame_inds = (
            clip_offsets[:, None]
            + np.arange(self.clip_len)[None, :] * self.frame_interval
        )
        frame_inds = np.concatenate(frame_inds)

        frame_inds = frame_inds.reshape((-1, self.clip_len))
        frame_inds = np.mod(frame_inds, total_frames)
        frame_inds = np.concatenate(frame_inds) + start_index
        return frame_inds.astype(np.int32)
    
import numpy as np
import random

class FragmentSampleFrames:
    def __init__(self, fsize_t, fragments_t, frame_interval=1, num_clips=1):

        self.fragments_t = fragments_t
        self.fsize_t = fsize_t
        self.size_t = fragments_t * fsize_t
        self.frame_interval = frame_interval
        self.num_clips = num_clips

    def get_frame_indices(self, num_frames, train=False):

        tgrids = np.array([num_frames // self.fragments_t * i for i in range(self.fragments_t)], dtype=np.int32)
        tlength = num_frames // self.fragments_t

        if train:
            rnd_t = np.random.randint(0, max(1, tlength - self.fsize_t * self.frame_interval), size=len(tgrids))
        else:
            rnd_t = np.zeros(len(tgrids), dtype=np.int32) + max(0, tlength - self.fsize_t * self.frame_interval) // 2
        ranges_t = (np.arange(self.fsize_t)[None, :] * self.frame_interval + rnd_t[:, None] + tgrids[:, None])
        return np.concatenate(ranges_t)

    def __call__(self, total_frames, train=False, start_index=0):
        frame_inds = []

        for i in range(self.num_clips):
            frame_inds += [self.get_frame_indices(total_frames, train=train)]
            
        frame_inds = np.concatenate(frame_inds) + start_index
        frame_inds[frame_inds > total_frames - 1] = total_frames - 1
        return frame_inds.astype(np.int32)


class FineTuneDataset(torch.utils.data.Dataset):
    def __init__(self, video_infos, opt, stype='fragments', is_train=True):
        ## opt is a dictionary that includes options for video sampling
        
        super().__init__()
        
        
        self.video_infos = video_infos
        self.opt = opt

        self.sample_types = opt["sample_types"]
        self.data_backend = opt.get("data_backend", "disk")
        self.augment = opt.get("augment", False)
        if self.data_backend == "petrel":
            from petrel_client import client
            self.client = client.Client(enable_mc=True)

        self.stype = stype
        print('Data form {}'.format(self.stype))

        self.phase = 'train' if is_train else 'test'
        self.crop = opt.get("random_crop", False)
        self.mean = torch.FloatTensor([123.675, 116.28, 103.53])
        self.std = torch.FloatTensor([58.395, 57.12, 57.375])
        self.samplers = {}
        for stype, sopt in opt["sample_types"].items():
            self.samplers[stype] = FragmentSampleFrames(sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"])


    def refresh_hypers(self):
        if not hasattr(self, "initial_sample_types"):
            self.initial_sample_types = copy.deepcopy(self.sample_types)
        
        types = self.sample_types
        
        if "fragments_up" in types:
            ubh, ubw = self.initial_sample_types["fragments_up"]["fragments_h"] + 1, self.initial_sample_types["fragments_up"]["fragments_w"] + 1
            lbh, lbw = self.initial_sample_types["fragments"]["fragments_h"] + 1, self.initial_sample_types["fragments"]["fragments_w"] + 1
            dh, dw = types["fragments_up"]["fragments_h"], types["fragments_up"]["fragments_w"]

            types["fragments_up"]["fragments_h"] = random.randrange(max(lbh, dh-1), min(ubh, dh+2))
            types["fragments_up"]["fragments_w"] = random.randrange(max(lbw, dw-1), min(ubw, dw+2))
            
        if "resize_up" in types:
        
            types["resize_up"]["size_h"] = types["fragments_up"]["fragments_h"] * types["fragments_up"]["fsize_h"]
            types["resize_up"]["size_w"] = types["fragments_up"]["fragments_w"] * types["fragments_up"]["fsize_w"]
        
        self.sample_types.update(types)

        #print("Refreshed sample hyper-paremeters:", self.sample_types)

        
    def __getitem__(self, index):
        video_info = self.video_infos[index]
        filename = video_info["filename"]
        label = video_info["label"]
        
        data, frame_inds = get_spatial_and_temporal_samples(filename, self.sample_types, self.samplers, self.stype, self.phase == "train", self.augment and (self.phase == "train"))

        for k, v in data.items():
            data[k] = ((v.permute(1, 2, 3, 0) - self.mean) / self.std).permute(3, 0, 1, 2)

        data["num_clips"] = {}
        for stype, sopt in self.sample_types.items():
            data["num_clips"][stype] = sopt["num_clips"]
        data["frame_inds"] = frame_inds
        data["gt_label"] = label
        data["name"] = osp.basename(video_info["filename"])
        
        return data
    
    def __len__(self):
        return len(self.video_infos)
    

class FusionDataset(torch.utils.data.Dataset):
    def __init__(self, opt, stype='fragments'):
        ## opt is a dictionary that includes options for video sampling
        
        super().__init__()
        
        
        self.video_infos = []
        self.ann_file = opt["anno_file"]
        self.data_prefix = opt["data_prefix"]
        self.opt = opt
        self.sample_types = opt["sample_types"]
        self.data_backend = opt.get("data_backend", "disk")
        self.augment = opt.get("augment", False)
        if self.data_backend == "petrel":
            from petrel_client import client
            self.client = client.Client(enable_mc=True)

        self.stype = stype
        print('Data form {}'.format(self.stype))

        self.phase = opt["phase"]
        self.crop = opt.get("random_crop", False)
        self.mean = torch.FloatTensor([123.675, 116.28, 103.53])
        self.std = torch.FloatTensor([58.395, 57.12, 57.375])
        self.samplers = {}
        for stype, sopt in opt["sample_types"].items():
            self.samplers[stype] = FragmentSampleFrames(sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"])


        if isinstance(self.ann_file, list):
            self.video_infos = self.ann_file
        else:
            try:
                with open(self.ann_file, "r") as fin:
                    for line in fin:
                        line_split = line.strip().split(",")
                        fileid, _, _, label = line_split
                        label = float(label)
                        filename = osp.join(self.data_prefix, fileid)
                        self.video_infos.append(dict(filename=filename, label=label, fileid=fileid))
            except:
                #### No Label Testing
                video_filenames = sorted(glob.glob(self.data_prefix+"/*.mp4"))
                print(video_filenames)
                for filename in video_filenames:
                    self.video_infos.append(dict(filename=filename, label=-1))


    def refresh_hypers(self):
        if not hasattr(self, "initial_sample_types"):
            self.initial_sample_types = copy.deepcopy(self.sample_types)
        
        types = self.sample_types
        
        if "fragments_up" in types:
            ubh, ubw = self.initial_sample_types["fragments_up"]["fragments_h"] + 1, self.initial_sample_types["fragments_up"]["fragments_w"] + 1
            lbh, lbw = self.initial_sample_types["fragments"]["fragments_h"] + 1, self.initial_sample_types["fragments"]["fragments_w"] + 1
            dh, dw = types["fragments_up"]["fragments_h"], types["fragments_up"]["fragments_w"]

            types["fragments_up"]["fragments_h"] = random.randrange(max(lbh, dh-1), min(ubh, dh+2))
            types["fragments_up"]["fragments_w"] = random.randrange(max(lbw, dw-1), min(ubw, dw+2))
            
        if "resize_up" in types:
        
            types["resize_up"]["size_h"] = types["fragments_up"]["fragments_h"] * types["fragments_up"]["fsize_h"]
            types["resize_up"]["size_w"] = types["fragments_up"]["fragments_w"] * types["fragments_up"]["fsize_w"]
        
        self.sample_types.update(types)

        #print("Refreshed sample hyper-paremeters:", self.sample_types)

        
    def __getitem__(self, index):
        video_info = self.video_infos[index]
        filename = video_info["filename"]
        label = video_info["label"]
        
        data, frame_inds = get_spatial_and_temporal_samples(filename, self.sample_types, self.samplers, self.stype, self.phase == "train", self.augment and (self.phase == "train"))

        for k, v in data.items():
            data[k] = ((v.permute(1, 2, 3, 0) - self.mean) / self.std).permute(3, 0, 1, 2)

        data["num_clips"] = {}
        for stype, sopt in self.sample_types.items():
            data["num_clips"][stype] = sopt["num_clips"]
        data["frame_inds"] = frame_inds
        data["gt_label"] = label
        data["name"] = osp.basename(video_info["filename"])
        
        return data
    
    def __len__(self):
        return len(self.video_infos)
    
