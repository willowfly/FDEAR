# this code cope with the wrmq data 
import argparse
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import *

def read_result(filename):
    tmp = pd.read_csv(filename,delimiter=',',header=0)
    yt6 = tmp['ytrue'].values 
    yp6 = tmp['ypred'].values
    
    if yt6.max()==5 or yp6.max()==5:
        class6 = ['normal','lop ear','Stahl','helical','cup','microtia']
    else:
        class6 = ['normal','lop ear','Stahl','helical','cup']
    yt2,yp2,class2 = transfer2(yt6,yp6)    
    return yt6,yp6,class6,yt2,yp2,class2   

def transfer2(y_true,y_pred):
    y_t = y_true.copy()
    y_p = y_pred.copy()
    y_t[y_true>0] = 1
    y_p[y_pred>0] = 1
    classes = ['normal','abnormal']
    return y_t,y_p,classes

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
def get_hist(input):
    hist = torch.zeros( (6,1), dtype=torch.long )
    for ii in range( input.size()[0] ):
        d = input[ii]
        hist[d] = hist[d]+1
    return hist 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=str,
                        help='data')
    parser.add_argument('--model',
                        type=str, default=None,
                        help='model')
    args = parser.parse_args()
    outputfile = f'./result/output_{args.data}_{args.model}.csv'
    yt6,yp6,class6,yt2,yp2,class2 = read_result(filename=outputfile)
    cm6 = confusion_matrix(yt6,yp6)
    rpt6 = classification_report(yt6, yp6, labels=None, 
        target_names=class6, sample_weight=None, digits=4, output_dict=False,
        zero_division=1)
    cm2 = confusion_matrix(yt2,yp2)
    rpt2 = classification_report(yt2, yp2, labels=None, 
        target_names=class2, sample_weight=None, digits=4, output_dict=False,
        zero_division=1)
    file = f'./result/output_{args.data}_{args.model}'
    np.savetxt(fname=file+'.cm6',X=cm6)
    np.savetxt(fname=file+'.cm2',X=cm2)
    with open(file+'.rpt6','w') as f:
        f.write(rpt6)
    with open(file+'.rpt2','w') as f:
        f.write(rpt2)   
    print(outputfile)
    print(rpt6)
    print(rpt2)
