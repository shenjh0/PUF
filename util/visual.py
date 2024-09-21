import torch
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pdb
from torchvision import transforms
from sklearn.manifold import TSNE
import seaborn as sns


def add_alpha(image, alpha=64):
    # 打开背景图像和前景图像
    background = Image.fromarray(np.zeros_like(image))
    image = Image.fromarray(image)
    
    # 调整前景图像的透明度
    alpha = 0.5  # 设置透明度为50%
    image.putalpha(int(alpha * 255))
    
    # 使用Image.blend()方法混合图像
    result = Image.blend(background, image, alpha)
    return result

def bi_visual_conf(bi_path=r'/data/sjh/semi_dataset/CDD/label/train_01010.jpg'):
    mask = np.array(Image.open(bi_path).convert('RGB'))
    mask = torch.from_numpy(mask).unsqueeze(0).permute(0, 3, 1, 2)
    feature_map = mask.detach()
    feature_map = feature_map[:,:,::5,::5].sigmoid()
    # heatmap = feature_map[:,0,:,:]*0    #
    heatmap = feature_map[:1, 0, :, :] * 0 #取一张图片,初始化为0
    for c in range(feature_map.shape[1]):   # 按通道
        heatmap+=feature_map[:1,c,:,:]      # 像素值相加[1,H,W]
    heatmap = heatmap.cpu().numpy()    #因为数据原来是在GPU上的
    heatmap = np.mean(heatmap, axis=0) #计算像素点的平均值,会下降一维度[H,W]

    heatmap = np.maximum(heatmap, 0)  #返回大于0的数[H,W]
    heatmap /= np.max(heatmap)      #/最大值来设置透明度0-1,[H,W]

    # pdb.set_trace() 
    heatmap[heatmap==0.5] = 0.1
    heatmap = cv2.resize(np.float32(heatmap), (256, 256))  # 将热力图的大小调整为与原始图像相同
    heatmap0 = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式,0-255,heatmap0显示红色为关注区域，如果用heatmap则蓝色是关注区域
    heatmap = cv2.applyColorMap(heatmap0, cv2.COLORMAP_JET)  # cv2.COLORMAP_RAINBOW cv2.COLORMAP_JET
    blurred_image = cv2.GaussianBlur(heatmap, (0, 0), 10)
    heatmap = cv2.addWeighted(heatmap, 0.8, blurred_image, 0.2, 0)

    cv2.imwrite('heatmap_01010.png', heatmap)

@torch.no_grad()
def tsne_visualization(features, labels, outpath, hex="#ff8b8b",stride=1, down_scale=4):
    """
        features:(b,c,h,w)
        label:(h,w)
    """
    import pandas as pd

    # 将特征张量变形为二维矩阵
    num_samples = features.size(0)
    num_channels = features.size(1)
    height = features.size(2)
    width = features.size(3)
    reshaped_features = features.permute(0,2,3,1).view(num_samples*height*width, -1)
    # reshaped_features = features.view(num_samples*num_channels, height * width)

    # labels
    labels = torch.nn.functional.interpolate(labels, size=(int(height/down_scale), int(width/down_scale)), mode='nearest')
    reshaped_labels = labels.flatten().numpy().astype(np.uint8)
    labels = np.zeros_like(tsne_features[::stride, 0])

    # 将特征张量从 GPU 转移到 CPU
    reshaped_features = reshaped_features.cpu()

    # 使用 t-SNE 进行降维
    tsne = TSNE(n_components=2)
    tsne_features = tsne.fit_transform(reshaped_features.detach().numpy())

    #一个类似于表格的数据结构
    df = pd.DataFrame()
    df["y"] = labels[::stride, 0]
    df["comp1"] = tsne_features[::stride, 0]
    df["comp2"] = tsne_features[::stride, 1]
 
    # hex = ["#ff8b8b", "#6e85b7"]
    class_num = len(hex)
    data_label = []
    for v in df.y.tolist():
        if v == 1:
            data_label.append("one")
        else:
            data_label.append("zero")
    df["value"] = data_label
    
    # hue:根据y列上的数据种类，来生成不同的颜色；
    # style:根据y列上的数据种类，来生成不同的形状点；
    plt.figure()
    sns.scatterplot(x= df.comp1.tolist(), y= df.comp2.tolist(),hue=df.value.tolist(),style = df.value.tolist(),
                    palette=sns.color_palette(hex,class_num),markers= {"one":"v","zero":"o"},
                    data=df).set(title="T-SNE projection")
   
    plt.savefig(outpath)
    plt.clf()




if __name__ == "__main__":
    bi_visual_conf()
