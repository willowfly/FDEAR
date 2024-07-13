import torch.nn as nn 
import torch 
import torchvision.transforms as transforms 
import pandas as pd
from PIL import Image
import os 
import matplotlib.pyplot as plt
from torchvision.models import resnet50, densenet121, regnet_x_8gf, efficientnet_b3
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
    args = parser.parse_args()

    print(f'copying with {args.data} using {args.model}\n')

    if args.data == 'data120':
        d = Dataset120(transform=val_transform)
    else: 
        d = RealWorldData(data_label=args.data,transform=val_transform)
    m = load_model(model_name=args.model)
    m.eval()
    
    outputfile = f'output_{args.data}_{args.model}.csv'
    with open(outputfile,'w') as f:
        f.write(f'filename,ytrue,ypred,score0,score1,score2,score3,score4,score5\n')
        c = 0
        for idx in range(len(d)):
            x,y,filename = d[idx]
            yhat = m(x.unsqueeze(dim=0))
            y = y.numpy()
            score = yhat.detach().numpy()[0]
            ypred = torch.argmax(yhat,axis=1).numpy()[0]
            f.write(f'{filename},{y},{ypred},')
            f.write(f'{score[0]},{score[1]},{score[2]},{score[3]},{score[4]},{score[5]}\n')
            if y==ypred:
                c+=1.0
        print(c/len(d))