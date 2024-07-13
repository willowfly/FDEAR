import torch.nn as nn 
import torch 
import torchvision.transforms as transforms 
import pandas as pd
from PIL import Image
import os 
import matplotlib.pyplot as plt
from torchvision.models import resnet50, densenet121, regnet_x_8gf, efficientnet_b3
from torchvision.transforms import ToPILImage
import numpy as np
import argparse

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model(model_name='resnet50'):
    if model_name == 'resnet50':
        filename = './model/resnet50/best_checkpoint.pth'
        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 6)
    if model_name == 'densenet121':
        filename = './model/densenet121/best_checkpoint.pth'
        model = densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 6)
    if model_name == 'efficientnet_b3':
        filename = './model/efficientnet_b3/best_checkpoint.pth'
        model = efficientnet_b3()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features,6)
    if model_name == 'regnet_x_8gf':
        filename = './model/regnet_x_8gf/best_checkpoint.pth'
        model = regnet_x_8gf()
        model.fc = nn.Linear(model.fc.in_features,6)
    tmp = torch.load(filename,map_location='cpu')
    model.load_state_dict(tmp)
    return model

class Dataset120(torch.utils.data.Dataset):
    def __init__(self,transform=None):
        self.image_folder = './data/dataset120/'
        self.labels = self.get_result(result_file='./data/dataset120/true_result.csv')
        self.transform = transform
    def __len__(self):
        return len(self.labels)
    def __getitem__(self,index):
        path = self.image_folder+f'{index+1}.jpg'
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[index]
        filename = path
        return img,label,filename
    def get_result(self,result_file):
        res = pd.read_csv(result_file)
        label = res['type']-1
        return torch.from_numpy(label.values)
class RealWorldData(torch.utils.data.Dataset):
    def __init__(self,data_label,transform=None):
        if data_label == 'hfz':
            self.image_folder = './data/hfz/'
        if data_label == 'wrmq':
            self.image_folder = './data/wrmq/'
        if data_label == 'fj':
            self.image_folder = './data/fj/'
        if data_label == 'zy':
            self.image_folder = './data/zy/'
        if data_label == 'test':
            self.image_folder = './data/test/'
        if data_label == 'be4k':
            self.image_folder = './data/be4k/'
        if data_label == 'correction':
            self.image_folder = './data/correction/'
        self.labels,self.image_files = self.get_result()
        self.transform = transform 
    def __len__(self):
        return len(self.labels)
    def __getitem__(self,index):
        path = self.image_folder+self.image_files[index]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[index]
        return img,label,path
    def get_result(self):
        res = pd.read_csv(self.image_folder+'true_result.csv')
        label = res['type']
        files = res['filename']
        return torch.from_numpy(label.values),files.values


if __name__ == '__main__':
    # argparse 
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=str,
                        help='data')
    parser.add_argument('--model',
                        type=str, default=None,
                        help='model')
    parser.add_argument('--index', type=int)
    parser.add_argument('--mask', type=int)
    args = parser.parse_args()
    
    if args.data == 'data120':
        d = Dataset120(transform=val_transform)
        dd = Dataset120(transform=None)
    else: 
        d = RealWorldData(data_label=args.data,transform=val_transform)
    m = load_model(model_name=args.model)
    m.eval()
    
    x,y,filename = d[args.index]
    X,y,filename = dd[args.index]
    yhat = m(x.unsqueeze(dim=0))
    ypred = torch.argmax(yhat,axis=1).numpy()[0]
    # yscore = yhat[ypred]
    y = y.numpy()
    
    n = int(x.size()[1]/args.mask)
    res = torch.zeros(size=(n,n))
    d_result = torch.zeros(size=(n,n))
    d_score = torch.zeros(size=(n,n))
    softmax = nn.Softmax(dim=1)
    yhat = softmax(yhat).detach().squeeze(dim=0)
    score = yhat[ypred]
    for i in range(n):
        for j in range(n):
            xx = x.clone()
            xx[:,i*args.mask:(i+1)*args.mask,j*args.mask:(j+1)*args.mask] = xx.min()
            yhat2 = m(xx.unsqueeze(dim=0))
            ypred2 = torch.argmax(yhat2,axis=1).numpy()[0]
            yhat2 = softmax(yhat2).detach().squeeze(dim=0)
            res[i,j] = ypred2
            score2 = yhat2[ypred]
            if ypred != ypred2:
                d_result[i,j] = 1
            d_score[i,j] = score - score2

    print(yhat)
    print(ypred)
    print(res)
    print(d_result)
    print(d_score)
    
    plt.subplot(1,3,1)
    X = X.resize((224,224),Image.BILINEAR)
    plt.imshow(X)
    # heatmap = Image.fromarray(d_result.numpy(), mode='RGB')  # 'RGB'模式是彩色图像
    d_score[d_score<=0] = 0
    d_score = (d_score-d_score.min())/(d_score.max()-d_score.min())
    #d_score_mean = d_score.mean()
    #d_score_std = d_score.std()
    #d_score = (d_score-d_score_mean)/d_score_std
    heatmap = Image.fromarray(255*d_score.numpy())
    heatmap = heatmap.convert("RGB")
    heatmap = heatmap.resize(X.size, Image.BICUBIC) 
    width,height = heatmap.size 
    for ii in range(width):
       for jj in range(height):
           r,g,b = heatmap.getpixel((ii,jj))
           heatmap.putpixel((ii,jj),(r,0,0)) 
    # heatmap = heatmap.resize(X.size, Image.BILINEAR)  # 将热力图调整为原图的大小
     # 将热力图调整为原图的大小
    blend_img = Image.blend(X, heatmap, alpha=0.7)  # alpha 设置叠加的透明度
    #Xt = X.convert("RGB")
    #width,height = X.size 
    #for ii in range(width):
    #   for jj in range(height):
    #       r,g,b = Xt.getpixel((ii,jj))
    #       newa,newb,newc = heatmap.getpixel((ii,jj))
    #       c = float(newa)/255
    #       filt = 0.1+np.power(c,1/3)
    #       # filt = 1/(1+np.exp(-(c-0.2)*20.0))
    #       Xt.putpixel((ii,jj),(int(r*filt),int(g*filt),int(b*filt))) 
    #blend_img = Xt

    X.save('x.png')
    heatmap.save('h.png')
    blend_img.save('xh.png')

    plt.subplot(1,3,2)
    plt.axis('off')
    plt.imshow(heatmap)
    plt.subplot(1,3,3)
    plt.imshow(blend_img)
    plt.show()