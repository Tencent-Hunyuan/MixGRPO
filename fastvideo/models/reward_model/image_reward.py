# Image-Reward: Copyied from https://github.com/THUDM/ImageReward
import os
from typing import Union, List
from PIL import Image

import torch
try:
    import ImageReward as RM
except ImportError:
    raise Warning("ImageReward is required to be installed (`pip install image-reward`) when using ImageReward for post-training.")


class ImageRewardModel(object):
    def __init__(self, model_name, device, http_proxy=None, https_proxy=None, med_config=None):
        if http_proxy:
            os.environ["http_proxy"] = http_proxy
        if https_proxy:
            os.environ["https_proxy"] = https_proxy
        self.model_name = model_name if model_name else "ImageReward-v1.0"
        self.device = device
        self.med_config = med_config
        self.build_reward_model()

    def build_reward_model(self):
        self.model = RM.load(self.model_name, device=self.device, med_config=self.med_config)

    @torch.no_grad()
    def __call__(
            self,
            images,
            texts,
    ):
        if isinstance(texts, str):
            texts = [texts] * len(images)
        
        rewards = []
        for image, text in zip(images, texts):
            ranking, reward = self.model.inference_rank(text, [image])
            rewards.append(reward)
        return rewards
