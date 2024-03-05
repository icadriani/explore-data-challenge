import copy
import numpy as np
from PIL import Image
from random import random

class AbstractDataset():
    def __init__(self,hp):
        self._hp=hp
        self.get_labels_dicts()
    def normalize(self,im,max_value=255):
        if len(np.unique(im))==1: return im # im*0+max_value//2 # None
        im=np.array(im)
        im=copy.deepcopy(im)
        # im=im-np.nanmin(im)
        # # if np.nanmax(im)==0: return None
        # im=im/np.nanmax(im)
        im=im/max_value
        # im=im.tolist()
        return im
    def _preprocess(self,imgs,max_value=255,norm=True,figsize=None):
        if figsize is None:
            figsize=self._hp.figsize
        ims=[]
        for im in imgs:
            # im=im.resize((self._hp.figsize,self._hp.figsize))
            if im.shape[0]<figsize:
                base=np.zeros((figsize,figsize))
                x=int((figsize/2)-(im.shape[0]/2))
                y=int((figsize/2)-(im.shape[1]/2))
                # x+=round(random()*figsize/4-figsize/2)
                # y+=round(random()*figsize/4-figsize/2)
                # print(x,x+im.shape[0],y,y+im.shape[1],im.shape,base[x:x+im.shape[0],y:y+im.shape[1]].shape)
                base[x:x+im.shape[0],y:y+im.shape[1]]=np.array(im)
                im=copy.deepcopy(base)
            elif im.shape[0]<figsize:
                im=np.array(im)
                im=Image.fromarray(im)
                im=im.resize((figsize,figsize))
                im=np.array(im)
            else:
                im=np.array(im)
            # if isinstance(im,np.ndarray):
            #     im=Image.fromarray(im)
            # if not isinstance(im,np.ndarray):
            if norm:
                im=self.normalize(im)
            if im is not None:
                ims.append(im)
        return ims
    def get_labels_dicts(self):
        self.class2enc={'terrain':0,'boulder':1,'crater':2}
        self.enc2class={v:k for k,v in self.class2enc.items()}
        self._hp.set_n_classes(len(self.class2enc.keys()))
    def encode_label(self,label):
        return self.class2enc[label]
    def decode_label(self,label):
        return self.enc2class[label] if label in self.enc2class else 'nan'
    def encode_labels(self,labels):
        return [self.encodelabel(x) for x in labels]
    def decode_labels(self,labels):
        return [self.decode_label(x) for x in labels]
    def one_hot_encoding(self,lab):
        l=[0]*self._hp.n_classes
        l[lab]=1
        return l

    def get_obs_size(self,im,row,col,points,min_p):
        if im[row][col]>0 and [row,col] not in points:
            points+=[[row,col]]
            for r in range(max(0,row-1),min(row+2,len(im))):
                for c in range(max(0,col-1),min(col+2,len(im[0]))):
                    points+=self.get_obs_size(im,r,c,points,min_p)
                    points=np.unique(points,axis=0).tolist()
                    if len(points)>min_p:
                        return points
        return points

    def clean(self,im):
        cleaned=copy.deepcopy(im)
        min_p=5
        if type(cleaned)!=list:
            cleaned=cleaned.tolist()
        for r in range(len(cleaned)):
            for c in range(len(cleaned[0])):
                if cleaned[r][c]>0:
                    points=self.get_obs_size(cleaned,r,c,[],min_p)
                    if len(points)>0:
                        points=np.unique(points,axis=0).tolist()
                    if len(points)<min_p:
                        for p in points:
                            cleaned[p[0]][p[1]]*=0
        return np.array(cleaned)        