import torch
import torch.nn.functional as F
import pdb


from sklearn import metrics as mr

@ torch.no_grad()
def compute_uncertainty(feature, mode='entropy', axis=0):
    if mode == 'var':
        flat_tensor = feature.view(-1)
        top_values = torch.topk(flat_tensor, int(flat_tensor.numel() * 0.1)).values
        var_v = 0.5*(torch.var(flat_tensor)) + (1-top_values.mean())*0.5
        return var_v #0
    elif mode == 'entropy':
        feature = F.softmax(feature, dim=axis) # 1
        flat_tensor = feature.view(-1)
        top_values = torch.topk(flat_tensor, int(flat_tensor.numel() * 0.01)).values
        entropy = -0.5*(torch.sum(feature * torch.log(feature + 1e-8)) / feature.numel()) + (1-top_values.mean())*0.5
        return entropy
        # feat1 = F.softmax(feature, dim=axis) # 1
        # entropy = torch.sum(feat1 * torch.log(feat1 + 1e-8)) / feat1.numel()
        # feat2 = F.softmax(feature, dim=axis) # 0/1
        # # feat2 = feature # 0/1
        # flat_tensor = feat2.view(-1)
        # top_values = torch.topk(flat_tensor, int(flat_tensor.numel() * 0.01)).values
        # uncertainty = -0.5*entropy + (1-top_values.mean())*0.5
        # pdb.set_trace()
        # return uncertainty
    elif mode == 'alea':
        soft_feat  = feature.softmax(dim=axis) # 0 or 1
        var = torch.var(soft_feat)
        aleatoric_var = (torch.sum(torch.mul(torch.exp(-var), soft_feat)) / soft_feat.numel()) + var
        aleatoric_var = (torch.sum(torch.mul(torch.exp(-var), (1-soft_feat))) / soft_feat.numel()) + var
        return aleatoric_var
    
@ torch.no_grad()
def compute_entropy(feature, mode='entropy',axis=0):
    if mode == 'mean':
        feature = F.softmax(feature, dim=axis) #1
        # entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

        # 不考虑mean的熵计算
        entropy = -torch.sum(feature * torch.log(feature + 1e-8)) / feature.numel()
        return entropy

        # # 考虑mean的熵计算
        # flat_tensor = feature.view(-1)
        # top_values = torch.topk(flat_tensor, int(flat_tensor.numel() * 0.01)).values
        # entropy = -0.5*(torch.sum(feature * torch.log(feature + 1e-8)) / feature.numel()) + (1-top_values.mean())*0.5
        # return entropy
    else:
        # 不考虑mean的熵计算
        entropy = -torch.sum(feature * torch.log(feature + 1e-8)) / feature.numel()
        return entropy


def sklearn_mi(img1, img2):
    nmi = mr.normalized_mutual_info_score(img1.reshape(-1), img2.reshape(-1))
    # nmi = mr.adjusted_mutual_info_score(img1.reshape(-1), img2.reshape(-1))
    return nmi

def _bs_mi(imgs1, imgs2):
    mis = []
    for img1, img2 in zip(imgs1, imgs2):
        nmi = sklearn_mi(img1, img2)
        mis.append(nmi)
    
    if len(mis) != 0:
        mis_m = sum(mis) / len(mis)
    return mis_m

def _per_change(val1, val2):
    per_change = ((val2 - val1) / val1) * 100
    return per_change

if __name__ == "__main__":
    img1 = torch.randn(1, 3, 256, 256)
    img2 = torch.randn(1, 3, 256, 256)