import torch
import torch.nn as nn

from .swin_v1 import SwinTransformer as ImageEncoder_v1
from .swin_v2 import SwinTransformerV2 as ImageEncoder
from .head import VQAHead, IQAHead, VARHead, VQAHeadMLP, HyperHead


class IQAModel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.backbone = ImageEncoder_v1()
        self.backbone = ImageEncoder()
        self.vqa_head = VQAHeadMLP()

    def forward(self, x):
        f = self.backbone(x)
        scores = self.vqa_head(f)
        return scores.flatten(1).mean(1)

