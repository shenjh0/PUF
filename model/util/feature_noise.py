import torch
import random

def uniform_noise(feat, min_val=0.0, max_val=0.2):
    noise = torch.empty(feat.size()).uniform_(min_val, max_val).to(feat.device)
    noisy_feat = feat + noise
    return noisy_feat

def salt_pepper_noise(feat, salt_prob=0.05, pepper_prob=0.05):
    noisy_feat = feat.clone()
    salt_mask = (torch.rand(feat.size()) < salt_prob).to(noisy_feat.device)
    pepper_mask = (torch.rand(feat.size()) < pepper_prob).to(noisy_feat.device)
    noisy_feat[salt_mask] = 1.0
    noisy_feat[pepper_mask] = 0.0
    return noisy_feat

def dropout_noise(feat, drop_ratio=0.3):
    return torch.nn.Dropout2d(drop_ratio)(feat)

def gaussian_noise(feat, mean=0.0, std=0.1):
    noise = (torch.randn(feat.size()) * std + mean).to(feat.device)
    noisy_feat = feat + noise
    return noisy_feat

def gaussian_mixture_noise(feat, num_components=2, mean_range=(0.0, 0.1), std_range=(0.05, 0.2)):
    noisy_feat = feat.clone()
    for _ in range(num_components):
        mean = torch.empty(feat.size()).uniform_(*mean_range)
        std = torch.empty(feat.size()).uniform_(*std_range)
        noise = torch.randn(feat.size()) * std + mean
        noisy_feat += noise
    return noisy_feat

def exponential_noise(feat, scale=0.1):
    noise = torch.empty(feat.size()).exponential_(scale=scale)
    noisy_feat = feat + noise
    return noisy_feat


if __name__ == "__main__":
    feat = torch.zeros(4, 2048, 16, 16)
    noisy_feat_uniform = uniform_noise(feat, min_val=0.0, max_val=0.2)
    noisy_feat_salt_pepper = salt_pepper_noise(feat, salt_prob=0.05, pepper_prob=0.05)