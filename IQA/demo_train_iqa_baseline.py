# -------------------------------------------------
# SAMA, AAAI 2024
# Training code for IQA. 
# -------------------------------------------------
import torch
import random
import os
import os.path as osp
import fastvqa.models as models
import sys
import argparse
import torch.nn as nn

from scipy.stats import spearmanr, pearsonr
from scipy.stats import kendalltau as kendallr
import numpy as np
from torchvision import transforms

import yaml
import timeit 
from PIL import Image

from thop import profile
import warnings

warnings.filterwarnings("ignore")

from torch.utils.tensorboard import SummaryWriter  


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, files, labels, 
                 data_args={"fwin_h": 8, "fwin_w": 8, "fsize_h": 32, "fsize_w": 32}, 
                 stype="fragment",
                 is_train=True):
        
        super().__init__()
        
        self.files = files 
        self.labels = labels
        self.is_train = is_train
        self.length = len(files)

        self.fwin_h = data_args['fwin_h']
        self.fwin_w = data_args['fwin_w']
        self.fsize_h = data_args['fsize_h']
        self.fsize_w = data_args['fsize_w']

        self.minh = self.fwin_h * self.fsize_h
        self.minw = self.fwin_w * self.fsize_w
        self.minsize = max(self.minh, self.minw)

        self.stype = stype if stype in ["sama", "sama-spm"] else "fragment"
        print("processing data with [{}]".format(self.stype))

        if self.is_train:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(45),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

    def get_spatial_fragments(self, img, fragments_h=8, fragments_w=8, fsize_h=32, fsize_w=32):
        size_h = fragments_h * fsize_h
        size_w = fragments_w * fsize_w

        res_h, res_w = img.shape[-2:]
        ratio = min(res_h / size_h, res_w / size_w)
        if ratio < 1:
            img = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=1 / ratio, mode="bilinear", align_corners=False)
            img = img[0]
        size = size_h, size_w

        ## make sure that sampling will not run out of the picture
        hgrids = torch.LongTensor([min(res_h // fragments_h * i, res_h - fsize_h) for i in range(fragments_h)])
        wgrids = torch.LongTensor([min(res_w // fragments_w * i, res_w - fsize_w) for i in range(fragments_w)])
        hlength, wlength = res_h // fragments_h, res_w // fragments_w

        if self.is_train:
            if hlength > fsize_h:
                rnd_h = torch.randint(hlength - fsize_h, (len(hgrids), len(wgrids)))
            else:
                rnd_h = torch.zeros((len(hgrids), len(wgrids))).int()
            if wlength > fsize_w:
                rnd_w = torch.randint(wlength - fsize_w, (len(hgrids), len(wgrids)))
            else:
                rnd_w = torch.zeros((len(hgrids), len(wgrids))).int()
        else:
            rnd_h = torch.ones((len(hgrids), len(wgrids))).int() * int((hlength - fsize_h) / 2)
            rnd_w = torch.ones((len(hgrids), len(wgrids))).int() * int((wlength - fsize_w) / 2) 

        t_img = torch.zeros(img.shape[:-2] + size).to(img.device)

        for i, hs in enumerate(hgrids):
            for j, ws in enumerate(wgrids):
                h_s, h_e = i * fsize_h, (i + 1) * fsize_h
                w_s, w_e = j * fsize_w, (j + 1) * fsize_w

                h_so, h_eo = hs + rnd_h[i][j], hs + rnd_h[i][j] + fsize_h
                w_so, w_eo = ws + rnd_w[i][j], ws + rnd_w[i][j] + fsize_w
                t_img[:, h_s:h_e, w_s:w_e] = img[:, h_so:h_eo, w_so:w_eo]
        return t_img


    def get_spatial_fragments_spm(self, img, fragments_h=8, fragments_w=8, fsize_h=32, fsize_w=32):
        size_h = fragments_h * fsize_h
        size_w = fragments_w * fsize_w

        res_h, res_w = img.shape[-2:]
        ratio = min(res_h / size_h, res_w / size_w)
        if ratio < 1:
            res_h, res_w = round(res_h / ratio), round(res_w / ratio)
            img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(res_h, res_w), mode="bilinear", align_corners=False)
            img = img[0]
            ratio = min(res_h / size_h, res_w / size_w)
        size = size_h, size_w

        img_scale, hgrids, wgrids = [], [], []
        rnd_h, rnd_w = [], []
        if self.is_train:
            rnd_rh, rnd_rw = torch.rand((fragments_h, fragments_w)), torch.rand((fragments_h, fragments_w))
        else:
            rnd_rh, rnd_rw = torch.ones((fragments_h, fragments_w)) * 0.5, torch.ones((fragments_h, fragments_w)) * 0.5

        factors = [1, 1 / ratio]
        for scale in factors:
            this_h, this_w = round(res_h * scale), round(res_w * scale)
            img_scale.append(torch.nn.functional.interpolate(img.unsqueeze(0), size=(this_h, this_w), mode="bilinear", align_corners=False)[0])

            hgrids.append(torch.LongTensor([min(this_h // fragments_h * i, this_h - fsize_h) for i in range(fragments_h)]))
            wgrids.append(torch.LongTensor([min(this_w // fragments_w * i, this_w - fsize_w) for i in range(fragments_w)]))

            hlength, wlength = this_h // fragments_h, this_w // fragments_w
            rnd_h.append((rnd_rh[:, :] * (hlength - fsize_h)).int())
            rnd_w.append((rnd_rw[:, :] * (wlength - fsize_w)).int())

        target_imgs = torch.zeros((2, ) + img.shape[:-2] + size).to(img.device)
        for k, scale in enumerate(factors):
            for i, hs in enumerate(hgrids[k]):
                for j, ws in enumerate(wgrids[k]):
                    h_s, h_e = i * fsize_h, (i + 1) * fsize_h
                    w_s, w_e = j * fsize_w, (j + 1) * fsize_w

                    h_so = hs + rnd_h[k][i][j]
                    h_eo = h_so + fsize_h
                    w_so = ws + rnd_w[k][i][j]
                    w_eo = w_so + fsize_w
                    target_imgs[k, :, h_s:h_e, w_s:w_e] = img_scale[k][:, h_so:h_eo, w_so:w_eo]  # 32 * 32

        # patch-based mask [4, 4]
        mask = torch.zeros((1, size_h, size_w))
        for i in range(size_w // 8):  # patch为4
            for j in range(size_h // 8):
                mask[:, j*8:j*8+4, i*8:i*8+4] = 1
                mask[:, j*8+4:j*8+8, i*8+4:i*8+8] = 1

        out_img = mask * target_imgs[0] + (1 - mask) * target_imgs[1]
        return out_img

    def get_spatial_fragments_swm(self, img, fragments_h=8, fragments_w=8, fsize_h=32, fsize_w=32):
        size_h = fragments_h * fsize_h
        size_w = fragments_w * fsize_w

        res_h, res_w = img.shape[-2:]
        ratio = min(res_h / size_h, res_w / size_w)
        if ratio < 1:
            res_h, res_w = round(res_h / ratio), round(res_w / ratio)
            img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(res_h, res_w), mode="bilinear", align_corners=False)
            img = img[0]
            ratio = min(res_h / size_h, res_w / size_w)
        size = size_h, size_w

        img_scale, hgrids, wgrids = [], [], []
        rnd_h, rnd_w = [], []
        if self.is_train:
            rnd_rh, rnd_rw = torch.rand((fragments_h, fragments_w)), torch.rand((fragments_h, fragments_w))
        else:
            rnd_rh, rnd_rw = torch.ones((fragments_h, fragments_w)) * 0.5, torch.ones((fragments_h, fragments_w)) * 0.5

        factors = [1, 1 / ratio]
        for scale in factors:
            this_h, this_w = round(res_h * scale), round(res_w * scale)
            img_scale.append(torch.nn.functional.interpolate(img.unsqueeze(0), size=(this_h, this_w), mode="bilinear", align_corners=False)[0])

            hgrids.append(torch.LongTensor([min(this_h // fragments_h * i, this_h - fsize_h) for i in range(fragments_h)]))
            wgrids.append(torch.LongTensor([min(this_w // fragments_w * i, this_w - fsize_w) for i in range(fragments_w)]))

            hlength, wlength = this_h // fragments_h, this_w // fragments_w
            rnd_h.append((rnd_rh[:, :] * (hlength - fsize_h)).int())
            rnd_w.append((rnd_rw[:, :] * (wlength - fsize_w)).int())

        target_imgs = torch.zeros((2, ) + img.shape[:-2] + size).to(img.device)
        for k, scale in enumerate(factors):
            for i, hs in enumerate(hgrids[k]):
                for j, ws in enumerate(wgrids[k]):
                    h_s, h_e = i * fsize_h, (i + 1) * fsize_h
                    w_s, w_e = j * fsize_w, (j + 1) * fsize_w

                    h_so = hs + rnd_h[k][i][j]
                    h_eo = h_so + fsize_h
                    w_so = ws + rnd_w[k][i][j]
                    w_eo = w_so + fsize_w
                    target_imgs[k, :, h_s:h_e, w_s:w_e] = img_scale[k][:, h_so:h_eo, w_so:w_eo]  # 32 * 32

        # window-based mask [32, 32]
        mask = torch.zeros((1, size_h, size_w))
        for i in range(fragments_h):  # window
            for j in range(fragments_w):
                if (i + j) % 2 == 0:
                    mask[:, j*32:j*32+32, i*32:i*32+32] = 1

        out_img = mask * target_imgs[0] + (1 - mask) * target_imgs[1]
        return out_img
    
    def __getitem__(self, index):
        filename = self.files[index]
        label = float(self.labels[index])
        
        img = Image.open(filename).convert('RGB')
        width, height = img.size

        if min(width, height) < self.minsize:
            scale_factor = self.minsize / min(width, height)
            img = img.resize((int(width * scale_factor), int(height * scale_factor)), Image.BILINEAR)

        img = self.transform(img)

        if self.stype == "fragment":
            data = self.get_spatial_fragments(img, self.fwin_h, self.fwin_w, self.fsize_h, self.fsize_w)
        elif self.stype == "sama-spm":
            data = self.get_spatial_fragments_spm(img, self.fwin_h, self.fwin_w, self.fsize_h, self.fsize_w)
        elif self.stype == "sama":
            data = self.get_spatial_fragments_swm(img, self.fwin_h, self.fwin_w, self.fsize_h, self.fsize_w)
        else:
            raise NotImplementedError

        return data, label
    
    def __len__(self):
        return self.length



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
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)

    sigma, m = torch.std_mean(y, unbiased=False)
    y = (y - m) / (sigma + 1e-8)
    
    rho = torch.mean(y_pred.reshape(y.shape) * y)
    return 1 - rho

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

sample_types=["resize", "diamond_fragments", "fragments", "crop", "arp_resize", "arp_fragments"]


def finetune_epoch(ft_loader, model, model_ema, optimizer, scheduler, device, epoch=-1, split=-1, writer=None):
    
    model.train()

    tic = timeit.default_timer()

    criterion = nn.SmoothL1Loss()
    train_labels, pred_labels = [], []
    plcc_loss_total, rank_loss_total, loss_total = 0, 0, 0
    for i, (data, label) in enumerate(ft_loader):
        optimizer.zero_grad()
        
        data = data.to(device)
        label = label.to(device).float()
        
        scores = model(data) 
        scores = scores.view(label.shape)
        
        # Plain Supervised Loss
        # p_loss, r_loss = plcc_loss(scores, label), rank_loss(scores, label)
        
        loss = criterion(scores, label) # + 0.5 * rplcc_loss(scores, label)
        # loss = p_loss + 0.3 * r_loss + 0.3 * criterion(scores, label)


        # plcc_loss_total += p_loss.item()
        # rank_loss_total += r_loss.item()
        loss_total += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()
        
        #ft_loader.dataset.refresh_hypers()

        pred_labels.extend(list(scores.view(-1).detach().cpu().numpy()))
        train_labels.extend(list(label.view(-1).detach().cpu().numpy()))
        
        if model_ema is not None:
            model_params = dict(model.named_parameters())
            model_ema_params = dict(model_ema.named_parameters())
            for k in model_params.keys():
                model_ema_params[k].data.mul_(0.999).add_(
                    model_params[k].data, alpha=1 - 0.999)
    
    train_srcc = spearmanr(train_labels, pred_labels)[0]
    writer.add_scalar('train_srcc', train_srcc, epoch)

    writer.add_scalar('train_plcc_loss', plcc_loss_total, epoch)
    writer.add_scalar('train_rank_loss', rank_loss_total, epoch)
    writer.add_scalar('train_total_loss', loss_total, epoch)

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


def inference_set(inf_loader, model, device, best_, epoch, split=-1, save_model=False, suffix='s', save_name="divide", writer=None):

    model.eval()

    tic = timeit.default_timer()
    gt_labels, pr_labels = [], []

    best_s, best_p, best_k, best_r = best_
    
    with torch.no_grad():
        for i, (data, label) in enumerate(inf_loader):

            data = data.to(device)
            label = label.to(device)
            
            scores = model(data) 
            scores = scores.view(label.shape)

            pr_labels.extend(list(scores.cpu().numpy()))
            gt_labels.extend(list(label.cpu().numpy()))

    pr_labels = rescale(pr_labels, gt_labels)
    
    s = spearmanr(gt_labels, pr_labels)[0]
    p = pearsonr(gt_labels, pr_labels)[0]
    k = kendallr(gt_labels, pr_labels)[0]
    r = np.sqrt(((gt_labels - pr_labels) ** 2).mean())

    writer.add_scalar('val_{}_srcc'.format(suffix), s, epoch)
    writer.add_scalar('val_{}_plcc'.format(suffix), p, epoch)
    writer.add_scalar('val_{}_krcc'.format(suffix), k, epoch)
    writer.add_scalar('val_{}_rmse'.format(suffix), r, epoch)
    
    # del results, result #, video, video_up
    torch.cuda.empty_cache()

    if s + p > best_s + best_p and save_model:
        state_dict = model.state_dict()
        torch.save(
            {
                "state_dict": state_dict,
                "validation_results": best_,
            },
            f"pretrained_weights/{save_name}_{suffix}_dev.pth",
        )

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
        f"For {len(gt_labels)} images, \nthe accuracy of the model: [{suffix}] is as follows:\n  SROCC: {s:.4f} best: {best_s:.4f} \n  PLCC:  {p:.4f} best: {best_p:.4f}  \n  KROCC: {k:.4f} best: {best_k:.4f} \n  RMSE:  {r:.4f} best: {best_r:.4f}."
    )
    print('time elapsed {:02d}m {:02d}s.'.format(minutes, seconds))

    return best_s, best_p, best_k, best_r


def read_info(info_file, prefix):
    name, mos = [], []
    import os.path as osp
    if info_file[-3:] == "txt":
        with open(info_file, 'r') as f:
            for line in f:
                dis, score = line.split()
                name.append(osp.join(prefix, dis))
                mos.append(float(score))
        name = np.stack(name)
        mos = np.stack(mos).astype(np.float32)

    elif info_file[-3:] == "csv":
        import pandas as pd
        d = pd.read_csv(info_file)
        mos = np.asarray(d['MOS_zscore'].to_list()).astype(np.float32)
        name = d['image_name'].to_list()
        for i in range(len(name)):
            name[i] = osp.join(prefix, name[i])
        name = np.asarray(name)
    elif info_file[-3:] == "pkl":
        import pickle
        with open(info_file, 'rb') as f:
            d = pickle.load(f)
        for i, ifile in enumerate(d['files']):
            name.append(osp.join(prefix, ifile))
        name = np.asarray(name)
        mos = np.asarray(d['labels'])
    else:
        raise NotImplementedError
    
    return name, mos


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--opt", type=str, default="./options/fast-sama-iqa.yml", help="the option file"
    )

    args = parser.parse_args()

    if sys.gettrace():
        print('in DEBUG mode.')
        args.opt = './options/fast-sama-iqa.yml'
        
    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)
    
    if sys.gettrace():
        opt['num_workers'] = 0
        opt['test_num_workers'] = 0
        opt['name'] = 'DEBUG'

    print(opt)

    database = opt["data"]["database"]
    files, labels = read_info(opt["data"]["data_info"], opt["data"]["data_prefix"])

    num_samples = len(files)
    num_repeat = opt["num_splits"]
    if opt["data"]["database"] == "kadid":
        ref_idx = np.arange(81).repeat(5*25).reshape(-1)
        index_all = np.zeros((num_repeat, 81), dtype=np.int)
        for ii in range(num_repeat):
            index_current = np.arange(81)
            random.Random(ii * 123).shuffle(index_current)
            index_all[ii] = index_current
    else:
        index_all = np.zeros((num_repeat, num_samples), dtype=np.int)
        for ii in range(num_repeat):
            index_current = np.asarray(range(num_samples))
            random.Random(ii * 123).shuffle(index_current)   # shuffle with certain seed
            index_all[ii] = index_current
    np.savetxt('rand_index_{}.txt'.format(database), index_all, fmt='%d')

    # ------------------ fix seed -----------------------
    seed = 44442
    torch.manual_seed(seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    # ---------------------------------------------------

    os.makedirs('./pretrained_weights/', exist_ok=True)
    os.makedirs('./tf-logs/', exist_ok=True)
    torch.utils.backcompat.broadcast_warning.enabled = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # if sys.gettrace():
    #     device = 'cpu'
    
    # best_eval = {'koniq': [], 'livec': []}
    best_eval = {database: []}
    for split in range(num_repeat):
        print(f"""\n==================== SPLIT-{split:02d} ====================""")
        writer = SummaryWriter('./tf-logs/{}-split-{:02d}'.format(opt['name'], split))

        index = index_all[split]
        
        pos_train_end = int(0.8 * num_samples)
        if opt["data"]["database"] == "kadid":
            eval_ref_idx = index[:int(0.2 * 81)]
            trainindex, evalindex = [], []
            for iii in range(len(files)):
                if ref_idx[iii] in eval_ref_idx:
                    evalindex.append(iii)
                else:
                    trainindex.append(iii)
            trainindex = np.asarray(trainindex)
            evalindex = np.asarray(evalindex)
        else:
            trainindex = index[:pos_train_end]                 # the first 80%
            evalindex = index[pos_train_end:]

        trainindex.sort()
        evalindex.sort()

        train_dataset = ImageDataset(files[trainindex], labels[trainindex], data_args=opt["data"], stype=opt["stype"], is_train=True)
        eval_datasets = {}
        eval_datasets[database] = ImageDataset(files[evalindex], labels[evalindex], data_args=opt["data"], stype=opt["stype"], is_train=False)
        # eval_datasets['livec'] = ImageDataset(files_livec, labels_livec, data_args=opt["data"], is_train=False)

        train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=opt["batch_size"], num_workers=opt["num_workers"], shuffle=True)
        eval_loaders = {}
        for key, idataset in eval_datasets.items():
            eval_loaders[key] = torch.utils.data.DataLoader(
                idataset, batch_size=opt["test_batch_size"], num_workers=opt["test_num_workers"], 
                pin_memory=True, shuffle=False, drop_last=False)
        
        model = getattr(models, "IQAModel")().to(device)
        
        if "load_path" in opt:
            state_dict = torch.load(opt["load_path"], map_location=device)
            if 'pretrained_weights' in opt["load_path"] and "state_dict" in state_dict:
                i_state_dict = state_dict['state_dict']

            elif "model" in state_dict:
                ### migrate training weights from swin-transformer-v1
                state_dict = state_dict["model"]
                from collections import OrderedDict

                i_state_dict = OrderedDict()
                for key in state_dict.keys():
                    tkey = 'backbone.' + key
                    i_state_dict[tkey] = state_dict[key]

            elif "state_dict" in state_dict:
                ### migrate training weights from mmaction
                state_dict = state_dict["state_dict"]
                from collections import OrderedDict

                i_state_dict = OrderedDict()
                for key in state_dict.keys():
                    if "head" in key:
                        continue
                    if "cls" in key:
                        tkey = key.replace("cls", "vqa")
                    elif "backbone" in key:
                        i_state_dict[key] = state_dict[key]
                        i_state_dict["fragments_"+key] = state_dict[key]
                        i_state_dict["resize_"+key] = state_dict[key]
                    else:
                        i_state_dict[key] = state_dict[key]
            t_state_dict = model.state_dict()
            for key, value in t_state_dict.items():
                if key in i_state_dict and i_state_dict[key].shape != value.shape:
                    i_state_dict.pop(key)
            
            print(model.load_state_dict(i_state_dict, strict=False))
            
        #print(model)

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
        
        warmup_iter = int(opt["warmup_epochs"] * len(train_loader))
        max_iter = int((opt["num_epochs"] + opt["l_num_epochs"]) * len(train_loader))
        warmup_iter = max(1, warmup_iter)
        lr_lambda = (
            lambda cur_iter: max(1e-2, cur_iter / warmup_iter)
            # lambda cur_iter: cur_iter / warmup_iter
            if cur_iter <= warmup_iter
            else 1
            # else max(1e-1, min(1, 1 - 0.9 * (cur_iter / len(train_loader) - opt["constant_epochs"]) / (opt["num_epochs"] - 20 - opt["constant_epochs"]))) 
            # else 0.5 * (1 + math.cos(math.pi * (cur_iter - warmup_iter) / max_iter))
        )
        # lr_lambda = (lambda x: x)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda,
        )

        bests = {}
        bests_n = {}
        for key in eval_loaders.keys():
            bests[key] = -1,-1,-1,1000
            bests_n[key] = -1,-1,-1,1000
        
        for epoch in range(opt["num_epochs"]):
            print(f"Finetune Epoch {epoch}:")
            finetune_epoch(
                train_loader, model, model_ema, optimizer, scheduler, device, epoch, split,
                writer=writer)
            
            print(f"evaluation ..")
            # ----------------------------- reduce time consumption 
            for key in eval_loaders:
                bests[key] = inference_set(
                    eval_loaders[key],
                    model_ema if model_ema is not None else model,
                    device, bests[key], epoch, split, 
                    save_model=opt["save_model"], save_name=opt["name"],
                    suffix=key+"_s",
                    writer=writer
                )
        

        if opt["num_epochs"] > 0:
            for key in eval_loaders:
                print(
                    f"""SPLIT-{split:02d}, for the finetuning process on {key} with {len(eval_datasets[key])} images,
                    the best validation accuracy of the model-s is as follows:
                    SROCC: {bests[key][0]:.4f}
                    PLCC:  {bests[key][1]:.4f}
                    KROCC: {bests[key][2]:.4f}
                    RMSE:  {bests[key][3]:.4f}."""
                )


                best_eval[key].append([bests[key][0], bests[key][1], bests[key][2], bests[key][3]])
            
    print('\n ============================================== ')
    print(np.median(best_eval[database], 0))



if __name__ == "__main__":
    main()
