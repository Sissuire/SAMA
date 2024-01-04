# -------------------------------------------------
# Submission 6299, AAAI 2024
# Finetuning code for VQA. This code is modified from FAST-VQA [ECCV, 2022]
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


def train_test_split(dataset_path, ann_file, ratio=0.8, seed=42):
    random.seed(seed)
    video_infos = []
    with open(ann_file, "r") as fin:
        for line in fin.readlines():
            line_split = line.strip().split(",")
            filename, _, _, label = line_split
            label = float(label)
            filename = osp.join(dataset_path, filename)
            video_infos.append(dict(filename=filename, label=label))
    random.shuffle(video_infos)
    return (
        video_infos[: int(ratio * len(video_infos))],
        video_infos[int(ratio * len(video_infos)) :],
    )


def rank_loss(y_pred, y):
    ranking_loss = torch.nn.functional.relu(
        (y_pred - y_pred.t()) * torch.sign((y.t() - y))
    )
    scale = 1 + torch.max(ranking_loss)
    return (
        torch.sum(ranking_loss) / y_pred.shape[0] / (y_pred.shape[0] - 1) / scale
    ).float()

def plcc_loss(y_pred, y):
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
    sigma, m = torch.std_mean(y, unbiased=False)
    y = (y - m) / (sigma + 1e-8)
    loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
    rho = torch.mean(y_pred * y)
    loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
    return ((loss0 + loss1) / 2).float()

def rescaled_l2_loss(y_pred, y):
    y_pred_rs = (y_pred - y_pred.mean()) / y_pred.std()
    y_rs = (y - y.mean()) / (y.std() + eps)
    return torch.nn.functional.mse_loss(y_pred_rs, y_rs)

def rplcc_loss(y_pred, y, eps=1e-8):
    ## Literally (1 - PLCC) / 2
    cov = torch.cov(y_pred, y)
    std = (torch.std(y_pred) + eps) * (torch.std(y) + eps)
    return (1 - cov / std) / 2

def self_similarity_loss(f, f_hat, f_hat_detach=False):
    if f_hat_detach:
        f_hat = f_hat.detach()
    return 1 - torch.nn.functional.cosine_similarity(f, f_hat, dim=1).mean()

def contrastive_similarity_loss(f, f_hat, f_hat_detach=False, eps=1e-8):
    if f_hat_detach:
        f_hat = f_hat.detach()
    intra_similarity = torch.nn.functional.cosine_similarity(f, f_hat, dim=1).mean()
    cross_similarity = torch.nn.functional.cosine_similarity(f, f_hat, dim=0).mean()
    return (1 - intra_similarity) / (1 - cross_similarity + eps)

def rescale(pr, gt=None):
    if gt is None:
        pr = (pr - np.mean(pr)) / np.std(pr)
    else:
        pr = ((pr - np.mean(pr)) / np.std(pr)) * np.std(gt) + np.mean(gt)
    return pr

sample_types=["resize", "fragments", "crop", "arp_resize", "arp_fragments"]




def finetune_epoch(ft_loader, model, model_ema, optimizer, scheduler, device, epoch=-1, writer=None, 
                   need_upsampled=True, need_feat=True, need_fused=False, need_separate_sup=False):
    model.train()
    tic = timeit.default_timer()
    train_labels, pred_labels = [], []
    epoch_loss = 0

    for i, data in enumerate(ft_loader):
        optimizer.zero_grad()
        video = {}
        for key in sample_types:
            if key in data:
                video[key] = data[key].to(device)
        
        y = data["gt_label"].float().detach().to(device).unsqueeze(-1)
        scores = model(video, inference=False, reduce_scores=False) 
        if len(scores) > 1:
            y_pred = reduce(lambda x,y:x+y, scores)
        else:
            y_pred = scores[0]
        y_pred = y_pred.mean((-2, -1)).sum(-1)

        frame_inds = data["frame_inds"]
        
        # Plain Supervised Loss
        p_loss, r_loss = plcc_loss(y_pred, y), rank_loss(y_pred, y)
        
        loss = p_loss + 0.3 * r_loss
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()
        
        pred_labels.extend(list(y_pred.view(-1).detach().cpu().numpy()))
        train_labels.extend(list(y.view(-1).detach().cpu().numpy()))

        #ft_loader.dataset.refresh_hypers()

        
        if model_ema is not None:
            model_params = dict(model.named_parameters())
            model_ema_params = dict(model_ema.named_parameters())
            for k in model_params.keys():
                model_ema_params[k].data.mul_(0.999).add_(
                    model_params[k].data, alpha=1 - 0.999)
                
    
    train_srcc = spearmanr(train_labels, pred_labels)[0]

    writer.add_scalar('train_srcc', train_srcc, epoch)
    writer.add_scalar('train_total_loss', epoch_loss, epoch)

    toc = timeit.default_timer()

    minutes = int((toc - tic) / 60)
    seconds = int((toc - tic) % 60)
    print('Epoch-{:02d}, training SRCC={:.4f}, time elapsed {:02d}m {:02d}s.'.format(epoch, train_srcc, minutes, seconds))
    print('backbone_lr = {:.2e}, head_lr = {:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'],
                                                          optimizer.state_dict()['param_groups'][-1]['lr']))
    
    model.eval()

    
def profile_inference(inf_set, model, device):
    video = {}
    data = inf_set[0]
    for key in sample_types:
        if key in data:
            video[key] = data[key].to(device).unsqueeze(0)
    with torch.no_grad():
        flops, params = profile(model, (video, ))
    print(f"The FLOps of the Variant is {flops/1e9:.1f}G, with Params {params/1e6:.2f}M.")

def inference_set(inf_loader, model, device, best_, epoch, writer=None, save_model=False, suffix='s', save_name="divide"):

    results = []

    tic = timeit.default_timer()
    gt_labels, pr_labels = [], []

    best_s, best_p, best_k, best_r = best_
 
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

    writer.add_scalar('val_{}_srcc'.format(suffix), s, epoch)
    writer.add_scalar('val_{}_plcc'.format(suffix), p, epoch)
    writer.add_scalar('val_{}_krcc'.format(suffix), k, epoch)
    writer.add_scalar('val_{}_rmse'.format(suffix), r, epoch)
    
    torch.cuda.empty_cache()

    if s + p > best_s + best_p and save_model:
        state_dict = model.state_dict()
        torch.save(
            {"state_dict": state_dict, 
             "validation_results": best_},
            f"pretrained_weights/{save_name}_{suffix}_dev_v0.0.pth")

    best_s, best_p, best_k, best_r = (
        max(best_s, s),
        max(best_p, p),
        max(best_k, k),
        min(best_r, r),
    )


    writer.add_scalar('val_{}_best_srcc'.format(suffix), best_s, epoch)
    writer.add_scalar('val_{}_best_plcc'.format(suffix), best_p, epoch)
    writer.add_scalar('val_{}_best_krcc'.format(suffix), best_k, epoch)
    writer.add_scalar('val_{}_best_rmse'.format(suffix), best_r, epoch)

    toc = timeit.default_timer()
    minutes = int((toc - tic) / 60)
    seconds = int((toc - tic) % 60)

    print(
        f"For {len(gt_labels)} videos, \nthe accuracy of the model: [{suffix}] is as follows:\n  SROCC: {s:.4f} best: {best_s:.4f} \n  PLCC:  {p:.4f} best: {best_p:.4f}  \n  KROCC: {k:.4f} best: {best_k:.4f} \n  RMSE:  {r:.4f} best: {best_r:.4f}."
    )
    print('time elapsed {:02d}m {:02d}s.'.format(minutes, seconds))

    return best_s, best_p, best_k, best_r

    # torch.save(results, f'{args.save_dir}/results_{dataset.lower()}_s{32}*{32}_ens{args.famount}.pkl')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--opt", type=str, default="./options/fast-SAMA-finetune.yml", help="the option file"
    )

    args = parser.parse_args()
    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)
    print(opt)
    

    ## adaptively choose the device

    # os.environ['CUDA_VISIBLE_DEVICES']='6'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if sys.gettrace():
        print('in DEBUGE mode.')
        opt["name"] = "DEBUG"
        opt['train_num_workers']=0
        opt['test_num_workers']=0


    if opt.get("split_seed", -1) > 0:
        num_splits = 10
    else:
        num_splits = 1
        
    stype = opt['stype'] if opt['stype'] in ['sama', 'sama-c', 'sama-mix', 'sama+spm', 'sama+swm'] else 'fragments'
    
    for split in range(num_splits):
        print(f"""\n==================== SPLIT-{split:02d} ====================""")
        
        key = opt["data"]["database"]
        ann_file = opt["data"]["anno_file"]
        data_prefix = opt["data"]["data_prefix"]
        video_infos = []
        with open(ann_file, "r") as fin:
            for line in fin:
                line_split = line.strip().split(",")
                fileid, _, _, label = line_split
                label = float(label)
                filename = osp.join(data_prefix, fileid)
                video_infos.append(dict(filename=filename, label=label, fileid=fileid))
        video_infos = np.asarray(video_infos)

        index_current = np.arange(len(video_infos))
        random.Random(split * 123).shuffle(index_current)   # shuffle with certain seed
        pos_train_end = int(0.8 * len(video_infos))
        trainindex = index_current[:pos_train_end]
        evalindex = index_current[pos_train_end:]

        train_datasets, train_loaders, val_datasets, val_loaders = {}, {}, {}, {}
                
        val_datasets[key] = getattr(datasets, opt["data"]["type"])(video_infos[evalindex], 
                                                                   opt["data"]["test"], 
                                                                   stype=stype,
                                                                   is_train=False)
        val_loaders[key] = torch.utils.data.DataLoader(val_datasets[key], 
                                                       batch_size=opt["test_batch_size"], 
                                                       num_workers=opt["test_num_workers"], 
                                                       pin_memory=False,
                                                       shuffle=False,
                                                       drop_last=False)
        
        train_datasets[key] = getattr(datasets, opt["data"]["type"])(video_infos[trainindex], 
                                                                     opt["data"]["train"], 
                                                                     stype=stype, 
                                                                     is_train=True)
        train_loaders[key] = torch.utils.data.DataLoader(train_datasets[key], 
                                                         batch_size=opt["train_batch_size"], 
                                                         num_workers=opt["train_num_workers"], 
                                                         shuffle=True)
        print('dataset=[{}], with {} samples.'.format(key, len(train_datasets[key])))
        print('dataset=[{}], with {} samples.'.format(key, len(val_datasets[key])))

        ## defining model and loading checkpoint
        print('using device: {}'.format(device))
        model = getattr(models, opt["model"]["type"])(**opt["model"]["args"]).to(device)
        if "load_path" in opt:
            state_dict = torch.load(opt["load_path"], map_location=device)["state_dict"]
            print(model.load_state_dict(state_dict, strict=True))
            
        if opt["ema"]:
            from copy import deepcopy
            model_ema = deepcopy(model)
        else:
            model_ema = None

        #profile_inference(val_dataset, model, device)    

        # finetune the model
        param_groups=[]

        for key, value in dict(model.named_children()).items():
            if "backbone" in key:
                param_groups += [{"params": value.parameters(), "lr": opt["optimizer"]["lr"] * opt["optimizer"]["backbone_lr_mult"]}]
            else:
                param_groups += [{"params": value.parameters(), "lr": opt["optimizer"]["lr"]}]

        optimizer = torch.optim.AdamW(lr=opt["optimizer"]["lr"], 
                                      params=param_groups,
                                      weight_decay=opt["optimizer"]["wd"])
        warmup_iter = 0
        for train_loader in train_loaders.values():
            warmup_iter += int(opt["warmup_epochs"] * len(train_loader))
        max_iter = int((opt["num_epochs"] + opt["l_num_epochs"]) * len(train_loader))
        lr_lambda = (
            lambda cur_iter: cur_iter / warmup_iter
            if cur_iter <= warmup_iter
            else 0.5 * (1 + math.cos(math.pi * (cur_iter - warmup_iter) / max_iter))
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        bests = {}
        # bests_n = {}
        for key in val_loaders:
            bests[key] = -1,-1,-1,1000
            # bests_n[key] = -1,-1,-1,1000
        
        os.makedirs('./tensorboard/', exist_ok=True)
        os.makedirs('./pretrained_weights/', exist_ok=True)
        writer = SummaryWriter('./tensorboard/{}'.format(opt['name']))
        
        for epoch in range(opt["num_epochs"]):
            print(f"Finetune Epoch {epoch}:")

            for key, train_loader in train_loaders.items():
                finetune_epoch(
                    train_loader, model, model_ema, optimizer, scheduler, device, epoch, writer, 
                    opt.get("need_upsampled", False), opt.get("need_feat", False), opt.get("need_fused", False),
                )


            print(f"evaluation ..")

            for key in val_loaders:
                bests[key] = inference_set(
                    val_loaders[key],
                    model_ema if model_ema is not None else model,
                    device, bests[key], epoch, writer, 
                    save_model=opt["save_model"], save_name=opt["name"],
                    suffix=key+"_s",
                )
        if opt["num_epochs"] > 0:
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
