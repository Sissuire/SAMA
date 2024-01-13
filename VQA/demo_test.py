# -------------------------------------------------
# SAMA, AAAI 2024
# Testing code for VQA. 
# This code is modified from FAST-VQA [ECCV, 2022]
# -------------------------------------------------
import torch
import random
import os.path as osp
import fastvqa.models as models
import fastvqa.datasets as datasets
import os
import argparse
import sys

from scipy.stats import spearmanr, pearsonr
from scipy.stats.stats import kendalltau as kendallr
import numpy as np

import timeit
import math

import yaml

from functools import reduce
from thop import profile
import warnings

warnings.filterwarnings("ignore")

from torch.utils.tensorboard import SummaryWriter  


def rescale(pr, gt=None):
    if gt is None:
        pr = (pr - np.mean(pr)) / np.std(pr)
    else:
        pr = ((pr - np.mean(pr)) / np.std(pr)) * np.std(gt) + np.mean(gt)
    return pr

sample_types=["resize", "fragments", "crop", "arp_resize", "arp_fragments"]



def inference_set(inf_loader, model, device):

    results = []

    tic = timeit.default_timer()
    gt_labels, pr_labels = [], []
 
    for i, data in enumerate(inf_loader):
        result = dict()
        video, video_up = {}, {}
        for key in sample_types:
            if key in data:
                video[key] = data[key].to(device)
                ## Reshape into clips
                b, c, t, h, w = video[key].shape
                video[key] = video[key].reshape(b, c, data["num_clips"][key], t // data["num_clips"][key], h, w).permute(0,2,1,3,4,5).reshape(b * data["num_clips"][key], c, t // data["num_clips"][key], h, w) 

        with torch.no_grad():
            result["pr_labels"] = model(video).cpu().numpy()
                
        result["gt_label"] = data["gt_label"].item()

        results.append(result)
        
    ## generate the demo video for video quality localization
    gt_labels = [r["gt_label"] for r in results]
    pr_labels = [np.mean(r["pr_labels"][:]) for r in results]
    pr_labels = rescale(pr_labels, gt_labels)
    
    s = spearmanr(gt_labels, pr_labels)[0]
    p = pearsonr(gt_labels, pr_labels)[0]
    k = kendallr(gt_labels, pr_labels)[0]
    r = np.sqrt(((gt_labels - pr_labels) ** 2).mean())
    
    torch.cuda.empty_cache()

    toc = timeit.default_timer()
    minutes = int((toc - tic) / 60)
    seconds = int((toc - tic) % 60)

    print(
        f"For {len(gt_labels)} videos, \nthe accuracy of the model is as follows:\n  SROCC: {s:.4f} best: {best_s:.4f} \n  PLCC:  {p:.4f} best: {best_p:.4f}  \n  KROCC: {k:.4f} best: {best_k:.4f} \n  RMSE:  {r:.4f} best: {best_r:.4f}."
    )
    print('time elapsed {:02d}m {:02d}s.'.format(minutes, seconds))

    return s, p, k, r



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--opt", type=str, 
                        default="./options/fast-SAMA-test.yml", help="the option file")

    args = parser.parse_args()
    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)
    print(opt)
    
    ## adaptively choose the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if sys.gettrace():
        print('in DEBUGE mode.')
        opt["name"] = "DEBUG"
        opt['test_num_workers']=0

    ## defining model and loading checkpoint

    print('using device: {}'.format(device))
    model = getattr(models, opt["model"]["type"])(**opt["model"]["args"]).to(device)
    
        
    stype = opt['stype'] if opt['stype'] in ['sama', 'sama-c', 'sama-mix', 'sama+spm', 'sama+swm'] else 'fragments'
        
    val_datasets = {}
    for key in opt["data"]:
        if key.startswith("val"):
            val_datasets[key] = getattr(datasets, opt["data"][key]["type"])(opt["data"][key]["args"], stype=stype)
            print('dataset=[{}], with {} samples.'.format(key, len(val_datasets[key])))

    val_loaders = {}
    for key, val_dataset in val_datasets.items():
        val_loaders[key] = torch.utils.data.DataLoader(val_dataset, 
                                                        batch_size=opt["test_batch_size"], 
                                                        num_workers=opt["test_num_workers"], 
                                                        pin_memory=False,
                                                        shuffle=False,
                                                        drop_last=False)

    if "load_path" in opt:
        state_dict = torch.load(opt["load_path"], map_location=device)           
        print(model.load_state_dict(state_dict['state_dict'] , strict=False))
            

    print(f"evaluation ..")

    bests = {}
    for key in val_loaders:
        bests[key] = inference_set(
            val_loaders[key],
            model,
            device
        )                  

    for key in val_loaders:
        print(
            f"""For the finetuning process on {key} with {len(val_datasets[key])} videos,
            the best validation accuracy of the model-s is as follows:
            SROCC: {bests[key][0]:.4f}
            PLCC:  {bests[key][1]:.4f}
            KROCC: {bests[key][2]:.4f}
            RMSE:  {bests[key][3]:.4f}."""
        )



if __name__ == "__main__":
    main()
