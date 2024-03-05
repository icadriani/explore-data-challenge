import os
import shutil
from AbstractDataset import AbstractDataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from pycocotools.coco import COCO
from colorama import Fore
from random import shuffle, random, randint, choice
from math import ceil
import copy
from functools import reduce
import albumentations as A
import cv2
import gc

class BCDataset(AbstractDataset):
    def __init__(self,hp,compute_segmented=True):
        super(BCDataset,self).__init__(hp)
        self.indices={}
        self.dataset_paths={}
        self.datasets={}
        self.define_paths()
        self.lens={}
        self.datasets=self.load()
        self.labels=None if compute_segmented or not os.path.exists(self.dataset_paths['segmented_boulder']['base']) or not os.path.exists(self.dataset_paths['segmented_crater']['base'])  else self.load(True)
        self.filter()
        self.get_data()
        self.get_sizes()
        # self.save_segmented()
        self.augment()
        if self.indices is None:
            self.indices={s:list(range(len(self.data[s]['images']))) for s in self.data}
        else:
            for s in self.data:
                if s not in self.indices:
                    self.indices[s]=list(range(len(self.data[s]['images'])))
    def save_segmented(self):
        for k in self.dataset_paths:
            if 'segmented' in k:
                og=k.replace('segmented_','')
                for y in self.dataset_paths[k]:
                    if y!='base':
                        path=self.dataset_paths[k][y]
                        if not os.path.exists(path):
                            os.makedirs(path)
                        l=[x for x in os.listdir(self.dataset_paths[og][y]) if '.json' in x]
                        for x in l:
                            shutil.copy(os.path.join(self.dataset_paths[og][y],x),path)
        for s in self.keys:
            for i,idx in enumerate(self.keys[s]):
                name=self.datasets[idx[0]][s].imgs[idx[1]]['file_name']
                plt.imsave(os.path.join(self.dataset_paths['segmented_'+idx[0]][s],name),self.data[s]['labels'][i],format='tiff',cmap='gray')
    def get_data(self):
        print()
        self.data={}
        # self.augmentable={}
        for s in ['train','valid','test']:
            # self.keys[s]=self.keys[s][:100]
            ids=self.keys[s]
            # ids=ids[:200]
            # if s=='train':
            #     obs=[x[0] for x in ids]
            #     obs,ns=np.unique(obs,return_counts=True)
            #     # counts={obs[i]:ns[i] for i in range(len(obs))}
            #     min_counts=np.min(ns)
            #     ids_={k:[x for x in ids if x[0]==k] for k in obs}
            #     for k in ids_:
            #         shuffle(ids_[k])
            #     ids_={k:v[:min_counts] for k,v in ids_.items()}
            #     ids=reduce(lambda x,y: x+y, list(ids_.values()))
            #     self.keys[s]=ids
            self.data[s]={'images':[],'labels':[],'encoded_labels':[]}
            # bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Fore.RESET)
            t=tqdm(range(len(ids)),unit='image',dynamic_ncols=True,colour='cyan',desc=s[0].upper()+s[1:].lower())
            delete_keys=[]
            # self.augmentable[s]=[]
            cutims={}
            cutlabs={}
            cutencs={}
            add_keys={}
            for i in t:
                k=ids[i][0]
                idx=ids[i][1]
                im=Image.open(os.path.join(self.dataset_paths[k][s],self.datasets[k][s].imgs[idx]['file_name']))# for k,i in ids]
                # im=self._preprocess([im])
                # if len(im)>0:
                # im=im[0]
                if len(np.unique(np.array(im)))>2:
                    if self.labels is None:
                        # label=self.get_seg_label_(im,ids[i])
                        label=self.get_seg_label(ids[i],s)
                    else:
                        label=Image.open(os.path.join(self.dataset_paths['segmented_'+k][s],self.labels['segmented_'+k][s].imgs[idx]['file_name']))
                        label=np.array(label)
                        label=np.min(label,-1)
                        # label=label/255
                        # label=int(label*self._hp.n_classes)
                        # label*=self.encode_label(k)
                    im=np.array(im)
                    # print(im.shape,label.shape,self.datasets[k][s].imgs[idx]['height'])
                    # images,labels=self.cut(im,label,figsize=self._hp.image_size)
                    images=[im]
                    labels=[label]
                    # n=len(images)
                    # if n==1:
                    #     self.augmentable[s].append(i)
                    # else:
                    #     add_keys[i]=[self.keys[s][i]]*n
                    # plt.figure()
                    # plt.imshow(label,cmap='gray')
                    first=True
                    labels=self._preprocess(labels,figsize=self._hp.image_size)
                    labels=np.where(np.array(labels)>0,1,0)
                    images=self._preprocess(images,figsize=self._hp.image_size)
                    # print('-------------')
                    # print(len(labels))
                    for im,label in zip(images,labels):
                        # plt.figure()
                        # plt.imshow(label,cmap='gray')
                        # if True in np.isnan(im) or True in np.isnan(label):
                        #     print('hellooooooooooooooooooooooooooooooooooo')
                        if len(im)>0:
                            # print(len(np.unique(im))==1 or 1 not in label)
                            if len(np.unique(im))==1 or 1 not in label: 
                                label=label*0
                                enc=self.encode_label('terrain')
                            else:
                                enc=self.encode_label(k)
                            if first:
                                self.data[s]['images'].append(im)
                                self.data[s]['labels'].append(label)
                                self.data[s]['encoded_labels'].append(enc)
                                cutims[i]=[]
                                cutlabs[i]=[]
                                cutencs[i]=[]
                                add_keys[i]=[]
                                first=False
                            else:
                                cutims[i].append(im)
                                cutlabs[i].append(label)
                                cutencs[i].append(enc)
                                add_keys[i].append(self.keys[s][i])
                        else:
                            delete_keys.append(i)
                else:
                    delete_keys.append(i)
                # break
            # self.augmentable[s]=len()
            # print(len(cutims))
            for k in cutims:
                # print(len(cutims[k]))
                if k not in delete_keys:
                    if len(cutims[k])>0:
                        # print(len(cutims[k]),len(cutlabs[k]),len(cutencs[k]))
                        self.data[s]['images']+=cutims[k]
                        self.data[s]['labels']+=cutlabs[k]
                        self.data[s]['encoded_labels']+=cutencs[k]
            self.keys[s]=[x for j,x in enumerate(self.keys[s]) if j not in delete_keys]
            for k in add_keys:
                if k not in delete_keys:
                    # # print('add_keys',{k1:type(v) for k1,v in add_keys.items()})
                    # print('cutims',{k1:type(v) for k1,v in cutims.items()})
                    # print('cutlabs',{k1:type(v) for k1,v in cutlabs.items()})
                    # print('cutencs',{k1:type(v) for k1,v in cutencs.items()})
                    # # print('add_keys',{k1:np.isnan(np.array(v)) for k1,v in add_keys.items()})
                    # print('cutims',{k1:np.isnan(np.array(v)) for k1,v in cutims.items()})
                    # print('cutlabs',{k1:np.isnan(np.array(v)) for k1,v in cutlabs.items()})
                    # print('cutencs',{k1:np.isnan(np.array(v)) for k1,v in cutencs.items()})
                    self.keys[s]+=add_keys[k]
            self.keys[s]=list(range(len(self.keys[s])))
            t.bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.RESET)
        # self.data,self.keys=self.add_surface(self.data,self._hp.surface_samples)
    def add_obstacle(self,terrain,set_name):
        i=randint(0,len(self.data[set_name]['images'])-1)
        im=self.data[set_name]['images'][i]
        lab=self.data[set_name]['labels'][i]
        new_shape=(terrain.shape[1],terrain.shape[0])
        im=Image.fromarray(im)
        im=im.resize(new_shape)
        im=np.array(im)
        lab=Image.fromarray(lab)
        lab=lab.resize(new_shape)
        lab=np.array(lab)
        terrain=np.where(lab>0,im,terrain)
        return terrain,lab
    def add_surface(self,data,max_samples=None):
        if max_samples==0:
            return data,self.keys
        folder=os.path.join(self._hp.data_path,'surface')
        files=[os.path.join(folder,x) for x in os.listdir(folder)]#[:10]
        shuffle(files)
        files=files[:400]
        Image.MAX_IMAGE_PIXELS=None
        images=[]
        # labels=[]
        # encs=[]
        for f in tqdm(files):
            im=Image.open(f)
            im=np.array(im)
            im=cv2.pyrDown(im)
            # lab=np.zeros(im.shape)
            ims,labs=self.cut(im,figsize=self._hp.image_size)
            images+=ims
            # labels+=labs
            # encs+=[self.encode_label('terrain')]*len(ims)
        i=0
        size=len(images)
        t=tqdm(range(max_samples-len(images)))
        while len(images)<max_samples:
            im=self.augment_image(image=images[i],label=np.zeros(images[i].shape))[0]
            images.append(im)
            i=(i+1)%size
            t.update(1)
        idx=list(range(len(images)))
        shuffle(idx)
        if max_samples is not None:
            idx=idx[:max_samples]
        images=[images[i] for i in idx]
        gc.collect()
        # labels=[labels[i] for i in idx]
        # encs=[encs[i] for i in idx]
            # labels=labels[:max_samples]
            # encs=encs[:max_samples]
        train_size=ceil(len(images)*0.7)
        test_size=ceil(len(images)*0.1)
        train,valid,test={},{},{}
        train['images']=images[:train_size]
        valid['images']=images[train_size:-test_size]
        test['images']=images[-test_size:]
        # train['labels']=[np.zeros(images[0])]*len(train['images'])
        # train['encoded_labels']=[self.encode_label('terrain')]*len(train['images'])
        # valid['labels']=[np.zeros(images[0])]*len(valid['images'])
        # valid['encoded_labels']=[self.encode_label('terrain')]*len(valid['images'])
        # test['labels']=[np.zeros(images[0])]*len(test['images'])
        # test['encoded_labels']=[self.encode_label('terrain')]*len(test['images'])
        surf={'train':train,'valid':valid,'test':test}
        for s in surf:
            surf[s]['labels']=[np.zeros(images[0].shape)]*len(surf[s]['images'])
            surf[s]['encoded_labels']=[self.encode_label('terrain')]*len(surf[s]['images'])
        # for s in surf:
        #     for i in range(len(surf[s]['images'])):
        #         if random()<0.75:
        #             im,lab=self.add_obstacle(surf[s]['images'][i],s)
        #             surf[s]['images'][i]=im
        #             surf[s]['labels'][i]=lab
        keys={}
        for s in surf:
            for k in surf[s]:
                if isinstance(data[s][k],np.ndarray):
                    data[s][k]=data[s][k].tolist()
                data[s][k]+=surf[s][k]
            idx=list(range(len(data[s]['images'])))
            shuffle(idx)
            for k in data[s]:
                data[s][k]=[data[s][k][i] for i in idx]
            keys[s]=list(range(len(data[s]['images'])))
        return data,keys
    def define_paths(self):
        self.dataset_paths['boulder']={'base':os.path.join(self._hp.data_path,'Boulders_split_dataset')}
        self.dataset_paths['boulder'].update({s:os.path.join(self.dataset_paths['boulder']['base'],s+'_data') for s in ['train','valid','test']})
        self.dataset_paths['crater']={'base':os.path.join(self._hp.data_path,'Craters_split_dataset')}
        self.dataset_paths['crater'].update({s:os.path.join(self.dataset_paths['crater']['base'],s+'_data') for s in ['train','valid','test']})
        self.dataset_paths['segmented_boulder']={'base':os.path.join(self._hp.data_path,'segmented','Boulders_split_dataset')}
        self.dataset_paths['segmented_boulder'].update({s:os.path.join(self.dataset_paths['segmented_boulder']['base'],s+'_data') for s in ['train','valid','test']})
        self.dataset_paths['segmented_crater']={'base':os.path.join(self._hp.data_path,'segmented','Craters_split_dataset')}
        self.dataset_paths['segmented_crater'].update({s:os.path.join(self.dataset_paths['segmented_crater']['base'],s+'_data') for s in ['train','valid','test']})
    def load(self,segmentations=False):
        datasets={}
        # t='crater'
        for t in ['boulder','crater']:
            if segmentations:
                t='segmented_'+t
            datasets[t]={}
            for k in self.dataset_paths[t]:
                if k!='base':
                    datasets[t][k]=COCO(os.path.join(self.dataset_paths[t][k],'dataset.json'))
        return datasets
    def filter(self):
        for t in self.datasets:
            if self.datasets[t] is not None:
                for k in self.datasets[t]:
                    ims={}
                    for i in self.datasets[t][k].imgs:
                        im=self.datasets[t][k].imgs[i]
                        # 1000
                        if im['height']>20 and im['height']<500 and os.path.exists(os.path.join(self.dataset_paths[t][k],im['file_name'])):
                            ims[i]=im
                    self.datasets[t][k].imgs=ims
        self._define_key_list()
    def shuffle(self,s='train'):
        if s not in self.indices:
            self.indices[s]=list(range(len(self.data[s]['images'])))
        # if s in self.lens:
        #     self.restore(s,self.lens[s])
        indices=self.indices[s]
        shuffle(indices)
        # print(self.keys[s])
        self.indices[s]=indices
        # print(self.train_indeces)
        ims=[self.data[s]['images'][x] for x in indices]
        labs=[self.data[s]['labels'][x] for x in indices]
        encs=[self.data[s]['encoded_labels'][x] for x in indices]
        pos=[self.data[s]['positions'][x] for x in indices]
        self.data[s]['images']=ims
        self.data[s]['labels']=labs
        self.data[s]['encoded_labels']=encs
        self.data[s]['positions']=pos
        # self.lens[s]=self.flat(s)
    def _define_key_list(self):
        self.keys={s:[[k,e] for k in self.datasets if self.datasets[k] is not None for e in self.datasets[k][s].imgs] for s in self.datasets['crater']}
    def get_imgs(self,ids,s='train'):
        ims=[Image.open(os.path.join(self.dataset_paths[k][s],self.datasets[k][s].imgs[i]['file_name'])) for k,i in ids]
        ims=self._preprocess(ims)
        ims+=[np.zeros((self._hp.figsize,self._hp.figsize))]*(self._hp.batch_size-len(ims))
        ims=torch.FloatTensor(np.array(ims)).to(self._hp.device)
        ims=ims.unsqueeze(1)
        return ims
    def get_seg_label_(self,im,id_):
        min_value=np.min(im)
        median_value=np.median(im)
        max_value=np.max(im)
        eps=int(0.1*max_value)
        l=np.where(im<min_value+eps,self.encode_label(id_[0]),0)
        l=np.where(im>max_value-eps,self.encode_label(id_[0]),l)
        l=l.astype(np.uint8)
        l=Image.fromarray(l)
        l=l.resize((self._hp.figsize,self._hp.figsize))
        l=np.array(l)
        return l
    def get_seg_label(self,id_,s='train'):
        xs,ys=self.get_xy_segmentation(self.datasets[id_[0]][s].anns[id_[1]]['segmentation'])
        # xy=[[xs[i],ys[i]] for i in range(len(xs)s)]
        im_info=self.datasets[id_[0]][s].imgs[id_[1]]
        in_area=self.in_area((im_info['width'],im_info['height']),xs,ys)
        l=np.where(in_area,self.encode_label(id_[0]),0)
        l=l.astype(np.uint8)
        # l=Image.fromarray(l)
        # l=l.resize((self._hp.figsize,self._hp.figsize))
        # l=np.array(l)
        return l
    def get_labels(self,ids,s='train'):
        labels=[self.get_seg_label(x,s) for x in ids]
        labels+=[np.zeros((self._hp.figsize,self._hp.figsize))]*(self._hp.batch_size-len(labels))
        labels=torch.LongTensor(np.array(labels)).to(self._hp.device).flatten()
        return labels.flatten()
    def get_batch(self,i,s='train'):
        # lens=self.flat(s)
        start=i*self._hp.batch_size
        finish=start+self._hp.batch_size
        imgs=self.data[s]['images'][start:finish]
        unpadded_len=len(imgs)
        left=self._hp.batch_size-unpadded_len
        # imgs+=[(self._hp.ground_color+((random()*0.04)-0.02))*np.ones((self._hp.figsize,self._hp.figsize))]*(self._hp.batch_size-unpadded_len)
        segmetation=self.data[s]['labels'][start:finish]
        # segmetation+=[np.zeros(np.array(segmetation[0]).shape)]*(self._hp.batch_size-unpadded_len)
        labels=self.data[s]['encoded_labels'][start:finish]
        # labels+=[np.zeros(np.array(labels[0]).shape)]*(self._hp.batch_size-unpadded_len)
        # print((self._hp.batch_size-unpadded_len))
        positions=self.data[s]['positions'][start:finish]
        # print(positions)
        # positions=[p%(self._hp.batch_size+1) for p in positions]
        # print(positions)
        # positions+=[0]*(self._hp.batch_size-unpadded_len)
        # imgs=np.array(imgs)
        imgs=self.padding(imgs,left)
        segmetation=self.padding(segmetation,left)
        labels=self.padding(labels,left)
        positions=self.padding(positions,left)
        # print(np.min(imgs,0).shape)
        # tmp=np.reshape(imgs,(self._hp.batch_size,self._hp.figsize**2))
        # print(np.min(imgs),np.max(imgs),np.min(np.min(imgs,1,keepdims=True),-1,keepdims=True))
        # imgs=imgs-np.min(np.min(imgs,1,keepdims=True),-1,keepdims=True)
        # # print(np.min(imgs),np.max(imgs),np.max(np.max(imgs,1,keepdims=True),-1,keepdims=True))
        # max_im=np.max(np.max(imgs,1,keepdims=True),-1,keepdims=True)
        # imgs=np.where(max_im>0,imgs/max_im,imgs)
        # print(np.min(imgs),np.max(imgs))
        imgs=torch.FloatTensor(np.array(imgs)).to(self._hp.device)#.unsqueeze(1)
        # print(imgs.min(),imgs.max())
        # imgs=(imgs-imgs.min(1))/imgs.max(1)
        segmetation=torch.LongTensor(np.array(segmetation)).to(self._hp.device)#.flatten(1)
        labels=torch.LongTensor(np.array(labels)).to(self._hp.device).squeeze()
        # positions=torch.LongTensor(np.array(reduce(lambda x,y:x+y,[list(range(1,l+1) for l in lens)]))).to(self._hp.device)
        positions=torch.LongTensor(np.array(positions)).to(self._hp.device)
        batch={'input':imgs,'target_segmentations':segmetation,'target_labels':labels,'unpadded_len':unpadded_len*self._hp.figsize**2,'positions':positions}
        # for k in batch:
        #     print(k,True in np.isnan(batch[k].cpu().numpy()) if type(batch[k])!=int else np.isnan(batch[k]))
        # self.restore(s,lens)
        return batch
    # def get_positions(self,pos):
    #     pass
    def padding(self,l,left):
        shape=[left]+list(np.array(l).shape)[1:]
        return l+np.zeros(shape).tolist()
    def get_sizes(self):
        self.sizes={s:ceil(len(self.data[s]['images'])/self._hp.batch_size) for s in ['train','valid','test']}
    def get_all_batches(self):
        print('Dividing the dataset into batches ...')
        self.batches={}
        for s in self.sizes:
            if s!='train':
                self.batches[s]=[]
                bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Fore.RESET)
                t=tqdm(range(self.sizes[s]),unit='batch',dynamic_ncols=True,bar_format=bar_format,desc=s[0].upper()+s[1:].lower())
                for i in t:
                    batch=self.get_batch(i,s)
                    self.batches[s].append(batch)
                t.bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.RESET)
    def train(self):
        self.curr_set='train'
    def eval(self,test_type='test'):
        self.curr_set=test_type
    def __getitem__(self,i):
        return self.batches[self.curr_set][i]
    def get_xy_segmentation(self,s):
        return s[0][0::2], s[0][1::2]
    def x_on_segments(self,x,y,xs,ys):
        x_on_segments=[]
        for i in range(len(xs)):
            s=i
            e=i+1 if i<len(xs)-1 else 0
            x1=xs[s]
            x2=xs[e]
            y1=ys[s]
            y2=ys[e]
            x=((y-y1)*(x2-x1))/(y2-y1)+x1 if y2-y1!=0 else x
            x=round(x)
            x_on_segments.append(x)
        return x_on_segments
    def y_on_segments(self,x,y,xs,ys):
        y_on_segments=[]
        for i in range(len(xs)):
            s=i
            e=i+1 if i<len(xs)-1 else 0
            x1=xs[s]
            x2=xs[e]
            y1=ys[s]
            y2=ys[e]
            y=((x-x1)*(y2-y1))/(x2-x1)+y1 if x2-x1!=0 else y
            y=round(y)
            y_on_segments.append(y)
        return y_on_segments
    def x_on_segment(self,x,y,x1,y1,x2,y2):
        x=((y-y1)*(x2-x1))/(y2-y1)+x1 if y2-y1!=0 else x
        x=round(x)
        return ((x1<=x2 and x>=x1 and x<=x2) or (x2<=x1 and x<=x1 and x>=x2)) and ((y1<=y2 and y>=y1 and y<=y2) or (y2<=y1 and y<=y1 and y>=y2)), x
    def y_on_segment(self,x,y,x1,y1,x2,y2):
        y=((x-x1)*(y2-y1))/(x2-x1)+y1 if x2-x1!=0 else y
        y=round(y)
        return ((x1<=x2 and x>=x1 and x<=x2) or (x2<=x1 and x<=x1 and x>=x2)) and ((y1<=y2 and y>=y1 and y<=y2) or (y2<=y1 and y<=y1 and y>=y2)), y
    def in_area_(self,x,y,xs,ys):
        x4=[]
        y4=[]
        for i in range(len(xs)):
            s=i
            e=i+1 if i<len(xs)-1 else 0
            x_on_segment=self.x_on_segment(x,y,xs[s],ys[s],xs[e],ys[e])
            y_on_segment=self.y_on_segment(x,y,xs[s],ys[s],xs[e],ys[e])
            if x_on_segment[0] and y_on_segment[0]:
                return True
            if x_on_segment[0]:
                x4.append(x_on_segment[1])
                y4.append(y)
            elif y_on_segment[0]:
                x4.append(x)
                y4.append(y_on_segment[1])
        lx=[xi for xi in x4 if int(xi)<x]
        gx=[xi for xi in x4 if ceil(xi)>x]
        ly=[yi for yi in y4 if int(yi)<y]
        gy=[yi for yi in y4 if ceil(yi)>y]
        if len(lx)>0 and len(gx)>0 and len(ly)>0 and len(gy)>0:
            lx=max(lx)
            gx=min(gx)
            ly=max(ly)
            gy=min(gy)
            return x>=lx and x<=gx and y>=ly and y<=gy
        return False
    def in_area(self,shape,xs,ys):
        np.seterr(divide='ignore', invalid='ignore')

        x1=np.array(xs)
        x2=np.array(xs[1:]+[xs[0]])
        y1=np.array(ys)
        y2=np.array(ys[1:]+[ys[0]])
        x=np.array([[x]*len(xs) for x in range(shape[0])])
        y=np.array([[y]*len(ys) for y in range(shape[1])])

        x1=np.expand_dims(x1,(0,1))
        x1=np.repeat(x1,shape[1],axis=1)
        x1=np.repeat(x1,shape[0],axis=0)
        x2=np.expand_dims(x2,(0,1))
        x2=np.repeat(x2,shape[1],axis=1)
        x2=np.repeat(x2,shape[0],axis=0)
        y1=np.expand_dims(y1,(0,1))
        y1=np.repeat(y1,shape[1],axis=1)
        y1=np.repeat(y1,shape[0],axis=0)
        y2=np.expand_dims(y2,(0,1))
        y2=np.repeat(y2,shape[1],axis=1)
        y2=np.repeat(y2,shape[0],axis=0)
        x=np.expand_dims(x,1)
        x=np.repeat(x,shape[1],axis=1)
        y=np.expand_dims(y,0)
        y=np.repeat(y,shape[0],axis=0)

        x_on_segments=np.where(y2-y1!=0,((y-y1)*(x2-x1))/(y2-y1)+x1,x)
        y_on_segments=np.where(x2-x1!=0,((x-x1)*(y2-y1))/(x2-x1)+y1,y)

        x_on_segments=np.round(x_on_segments)
        y_on_segments=np.round(y_on_segments)

        is_x_on_segments_and1=np.where(np.where(x1<=x2,x_on_segments>=x1,False),x_on_segments<=x2,False)
        is_x_on_segments_and2=np.where(np.where(x1>=x2,x_on_segments<=x1,False),x_on_segments>=x2,False)
        is_x_on_segments_or1=np.where(is_x_on_segments_and1,True,is_x_on_segments_and2)
        is_x_on_segments_and3=np.where(np.where(y1<=y2,y>=y1,False),y<=y2,False)
        is_x_on_segments_and4=np.where(np.where(y1>=y2,y<=y1,False),y>=y2,False)
        is_x_on_segments_or2=np.where(is_x_on_segments_and3,True,is_x_on_segments_and4)
        is_x_on_segments=np.where(is_x_on_segments_or1,is_x_on_segments_or2,False)

        is_y_on_segments_and1=np.where(np.where(x1<=x2,x>=x1,False),x<=x2,False)
        is_y_on_segments_and2=np.where(np.where(x1>=x2,x<=x1,False),x>=x2,False)
        is_y_on_segments_or1=np.where(is_y_on_segments_and1,True,is_y_on_segments_and2)
        is_y_on_segments_and3=np.where(np.where(y1<=y2,y_on_segments>=y1,False),y_on_segments<=y2,False)
        is_y_on_segments_and4=np.where(np.where(y1>=y2,y_on_segments<=y1,False),y_on_segments>=y2,False)
        is_y_on_segments_or2=np.where(is_y_on_segments_and3,True,is_y_on_segments_and4)
        is_y_on_segments=np.where(is_y_on_segments_or1,is_y_on_segments_or2,False)

        first=np.where(is_x_on_segments,is_y_on_segments,False)
        first=np.any(first,axis=-1)

        # return first

        x4=np.where(is_x_on_segments,x_on_segments,np.where(is_y_on_segments,x,np.nan))
        y4=np.where(is_x_on_segments,y,np.where(is_y_on_segments,y_on_segments,np.nan))

        lx=np.where(x4<x,x4,-1)
        lx=np.nanmax(lx,axis=-1)
        lx=np.where(lx!=-1,lx,np.nan)
        gx=np.where(x4>x,x4,np.inf)
        gx=np.nanmin(gx,axis=-1)
        gx=np.where(gx!=np.inf,gx,np.nan)
        ly=np.where(y4<y,y4,-1)
        ly=np.nanmax(ly,axis=-1)
        ly=np.where(ly!=-1,ly,np.nan)
        gy=np.where(y4>y,y4,np.inf)
        gy=np.nanmin(gy,axis=-1)
        gy=np.where(gy!=np.inf,gy,np.nan)

        x=np.max(x,-1)
        y=np.max(y,-1)
        inside=np.where(x>=lx,x<=gx,False)
        inside=np.where(inside,y>=ly,False)
        inside=np.where(inside,y<=gy,False)

        ok=np.where(first,True,inside)

        return ok
    def resize(self,image,label,newsize,norm=True):
            # ground_color=np.mean(image)
            ground_color=image[0][0]
            im=Image.fromarray(image)
            im=im.resize((newsize,newsize))
            im=np.array(im)
            lab=Image.fromarray(label)
            lab=lab.resize((newsize,newsize))
            lab=np.array(lab)
            sx=max(0,(self._hp.figsize-abs(newsize))//2)
            sy=max(0,(self._hp.figsize-abs(newsize))//2)
            fx=sx+newsize
            fy=sy+newsize
            sx=max(0,sx)
            sy=max(0,sy)
            fx=min(self._hp.figsize,fx)
            fy=min(self._hp.figsize,fy)
            start=max(0,(abs(newsize)-self._hp.figsize)//2)
            finish=min(start+newsize,start+self._hp.figsize)
            im=im[start:finish,start:finish]
            lab=lab[start:finish,start:finish]
            # image=np.pad(im,((sx,self._hp.figsize-fx),(sy,self._hp.figsize-fy)),mode=pad_mode)
            # label=np.pad(lab,((sx,self._hp.figsize-fx),(sy,self._hp.figsize-fy)))      
            if norm:      
                im=self.normalize(im)
            if im is not None:
                image=np.ones((self._hp.figsize,self._hp.figsize))
                image=image*ground_color
                label=np.zeros((self._hp.figsize,self._hp.figsize))
                image[sx:fx,sy:fy]=im
                label[sx:fx,sy:fy]=lab
                return image,label
            return None, None
    def augment_image(self,s=None,i=None,image=None,label=None,enc=None):
        # # probs=[-1,-0.5,0.5,1]
        # # probs_size=[0.5,1,1.5,2]
        # # both=random()<0.85
        # # size=random()<0.5
        if s is not None:
            image=copy.deepcopy(self.data[s]['images'][i])
            label=copy.deepcopy(self.data[s]['labels'][i])
        figsize=max(image.shape[0],self._hp.figsize)

        # scale=[0.25,0.5,0.75,1]
        # translate=[0,0.2,0.35,0.5]
        # rotate=[0,45,90,140,180,270]
        # shear=[-45,0,45]

        transform=A.Compose([
            # A.RandomSizedCrop([int(0.3*figsize),int(0.8*figsize)],figsize,figsize,p=0.5),
            #                  A.RandomScale(0.5),
            #                  A.Rotate(359),
                            #  A.Affine(scale=random()*(2 if s is None or self.data[s]['encoded_labels'][i]!=self.encode_label('boulder') else 5),translate_percent=random()*0.5,rotate=random()*359,shear=random()*90-45,p=1),
                            #  A.Affine(scale=random()*(1 if s is None or self.data[s]['encoded_labels'][i]!=self.encode_label('boulder') else 3),translate_percent=random()*0.5,rotate=random()*359,shear=random()*90-45,p=1),
                             A.Affine(scale=random()*(10 if s is None or self.data[s]['encoded_labels'][i]!=self.encode_label('boulder') else 200),translate_percent=random()*0.5,rotate=random()*359,shear=random()*90-45,p=1),
                             A.Flip(),
                             A.PadIfNeeded(figsize,figsize)])

        transformed = transform(image=image,mask=label)
        image = transformed["image"]
        label = transformed["mask"]

        # if random()<0.5:
        #     image=np.max(image)-image

        # if image.shape[0]!=figsize or image.shape[1]!=figsize:
        #     print('helpppppppppppppppp')
        # # if np.argmax(self.data[s]['encoded_labels'][i])!=1: return None, None
        # # if i not in self.augmentable[s]: return image, label
        # isaug=False
        # ground_color=np.mean(image)
        # # if size:
        # newsize=round(((random()*2)-1)*(0.99*figsize)+figsize)
        # while newsize<=0:
        #     newsize=round(((random()*2)-1)*(0.99*figsize)+figsize)
        # image,label=self.resize(image,label,newsize,norm=False)
        # isaug=True
        # if image is None:
        #     return None, None
        # # if both or not size:
        # rows=round(((random()*1)-0.5)*figsize)
        # cols=round(((random()*1)-0.5)*figsize)
        # sizex=figsize-abs(rows)
        # sizey=figsize-abs(cols)
        # xs=max(0,-rows)
        # ys=max(0,-cols)
        # im=image[xs:xs+sizex,ys:ys+sizey]
        # lab=label[xs:xs+sizex,ys:ys+sizey]
        # sx=max(0,rows)
        # sy=max(0,cols)
        # # image=np.pad(im,((sx,self._hp.figsize-fx),(sy,self._hp.figsize-fy)),mode=pad_mode)
        # # label=np.pad(lab,((sx,self._hp.figsize-fx),(sy,self._hp.figsize-fy)))
        # # im=self.normalize(im) 
        # if im is not None:
        #     fx=sx+im.shape[0]
        #     fy=sy+im.shape[1]
        #     image=np.ones((figsize,figsize))
        #     image=image*ground_color #(np.median(self.data[s]['images'][i]))
        #     label=np.zeros((figsize,figsize))
        #     image[sx:fx,sy:fy]=im
        #     label[sx:fx,sy:fy]=lab
        #     isaug=True
        # # print(image.shape)
        # # image=self.normalize(image)
        # if not isaug:
        #     return None, None
        if 1 not in label:
            enc=self.encode_label('terrain')
        elif enc is None:
            enc=copy.deepcopy(self.data[s]['encoded_labels'][i])
        return image, label, enc
    def mix_ims(self,s,i1,i2):
        rnd=random()
        dominant=i1 if rnd>0.5 else i2
        recessivo=i1 if rnd<=0.5 else i2
        # print(i1,i2,dominant,recessivo)
        imd=self.data[s]['images'][dominant]
        imr=self.data[s]['images'][recessivo]
        segd=self.data[s]['labels'][dominant]
        segr=self.data[s]['labels'][recessivo]
        encd=self.data[s]['encoded_labels'][dominant]
        encr=self.data[s]['encoded_labels'][recessivo]
        # plt.figure()
        # plt.imshow(np.argmax(encd,-1),'gray')
        # plt.figure()
        # plt.imshow(np.argmax(encr,-1),'gray')        
        imd=copy.deepcopy(imd)
        imr=copy.deepcopy(imr)
        segd=copy.deepcopy(segd)
        segr=copy.deepcopy(segr)
        encd=copy.deepcopy(encd)
        encr=copy.deepcopy(encr)
        im=np.where(segr>segd,imr,imd)
        median=np.median(imd)
        seg=np.where(segr>segd,segr,segd)
        esegr=copy.deepcopy(segr)
        esegr=np.expand_dims(esegr,axis=-1)
        esegd=np.expand_dims(segd,axis=-1)
        enc=np.where(esegr>esegd,encr,encd)
        imd=copy.deepcopy(imd)
        imd=np.where(seg>0,0,imd)
        encim=np.expand_dims(imd,axis=-1)
        enc=np.where(encim==median,encr,enc)
        seg=np.where(imd==median,segr,seg)
        im=np.where(imd==median,imr,im)
        # im=np.mean([imd,imr],axis=0)
        # seg=np.mean([segd,segr],axis=0)
        # enc=np.mean([encd,encr],axis=0)
        # plt.figure()
        # plt.imshow(imd,'gray')
        # plt.figure()
        # plt.imshow(imr,'gray')
        # plt.figure()
        # plt.imshow(im,'gray')
        # plt.figure()
        # plt.imshow(segd,'gray')
        # plt.figure()
        # plt.imshow(segr,'gray')
        # plt.figure()
        # plt.imshow(seg,'gray')
        # plt.figure()
        # plt.imshow(np.argmax(enc,-1),'gray')
        # return
        # print(im.shape,seg.shape,enc.shape,dominant)
        # im=self.normalize(im)
        # print(im)
        # enc=enc.tolist()
        return im,seg,enc,dominant
    def n_aug(self,n_augs,sizes):
        if n_augs>0:
            print()
            t=tqdm(range(n_augs),unit='augs',dynamic_ncols=True,colour='cyan',desc='Adding augmentations')
            for n in t:
                for s in self.data:
                    for i in range(sizes[s]):
                        # if np.argmax(self.data[s]['labels'][i])==1:
                            image, label,enc=self.augment_image(s,i)
                            if image is not None:
                                self.data[s]['images'].append(image)
                                self.data[s]['labels'].append(label)
                                self.data[s]['encoded_labels'].append(enc)
                                self.keys[s].append(self.keys[s][i])
            t.close()
    def balance(self,original):
        print()
        most={}
        counts={s:{} for s in self.data}
        # indices={}
        for s in self.data:
            # labs=np.unique(np.argmax(self.data[s]['encoded_labels'],-1))
            # lab_shapes,counts_=np.unique([[np.argmax(enc),im.shape] for im,enc in zip(self.data[s]['images'],self.data[s]['encoded_labels'])],return_counts=True)
            # labs=[]
            # counts={}
            for lab,enc in zip(original[s]['labels'],original[s]['encoded_labels']):
                # shape=np.array(im).shape
                # lab=np.argmax(enc)
                if enc!=self.encode_label('terrain'):
                    if enc not in counts[s]:
                        counts[s][enc]=0
                    # curr=(shape[0]//3+1)*(shape[1]//3+1)
                    lab=np.array(lab).ravel()
                    curr=len(lab[lab>0])
                    counts[s][enc]+=curr
            # print(labs)
            # print(counts_)
            most[s]=sorted([[k,v] for k,v in counts[s].items() if k!=self.encode_label('terrain')],key=lambda x: x[1])[-1][0]
            # counts_=list(counts_)
            # counts[s]={k:v for k,v in zip(labs,counts_)}
            # most[s]=labs[counts_.index(max(counts_))]
            # indices[s]=idxes
            # print(counts[s])
            # print(most[s])
        # m=min(list(counts['train'].values()))
        # print(counts)
        # if m>10000:
        #     indices={k:[] for k in counts['train']}
        #     t=tqdm(len(self.data['train']['encoded_labels']))
        #     i=0
        #     while True in [len(indices[k])<m for k in indices]:
        #         k=np.argmax(self.data['train']['encoded_labels'][i])
        #         if len(indices[k])<m:
        #             indices[k].append(i)
        #         i+=1
        #         t.update(1)
        #     t.close()
        #     # for s in self.data:
        #     indices=reduce(lambda x,y: x+y,list(indices.values()))
        #     self.data['train']['images']=self.data['train']['images'][indices]
        #     self.data['train']['labels']=self.data['train']['labels'][indices]
        #     self.data['train']['encoded_labels']=self.data['train']['encoded_labels'][indices]
        #     self.keys['train']=self.keys['train'][indices]
        # else:
        balanced={s:False for s in original}
        # balanced={s:True for s in self.data}
        for s in original:
            to_do=sum([counts[s][most[s]]-counts[s][x] for x in counts[s] if x!=most[s]])
            t=tqdm(range(to_do),unit='image',dynamic_ncols=True,colour='cyan',desc='Balancing '+s)
            nothing_added=True
            left=to_do
            while not balanced[s]:
                for i in range(len(original[s])):
                    # lab=list(original[s]['encoded_labels'][i]).index(1)
                    lab=original[s]['encoded_labels'][i]
                    # print(counts,lab,most[s])
                    if lab!=most[s] and counts[s][lab]<counts[s][most[s]]:
                            # print('hello 2')
                        # if np.argmax(self.data[s]['labels'][i])!=1:
                            image, label, lab=self.augment_image(s,i)
                            # lab=list(enc).index(1)
                            if lab in counts[s] and image is not None and lab!=most[s] and counts[s][lab]<counts[s][most[s]]:
                                # print('hello 3')
                                self.data[s]['images'].append(image)
                                self.data[s]['labels'].append(label)
                                self.data[s]['encoded_labels'].append(lab)
                                self.keys[s].append(self.keys[s][i])
                                # curr=(image.shape[0]//3+1)*(image.shape[1]//3+1)
                                label=np.array(label).ravel()
                                curr=len(label[label>0])
                                counts[s][lab]+=curr
                                left-=curr
                                t.update(curr)
                                nothing_added=False
                # print('ciao')
                # print(left)
                balanced[s]=left<=0 or nothing_added
                # print(nothing_added,balanced[s])
            t.close()
            print(left)
    def add_full_obstacles(self,sizes):
        print('\nAdding full obstacles')
        for s in self.data:
            obsts=np.unique(self.data[s]['encoded_labels'])
            obsts={k:0 for k in obsts}
            tot=int(len(self.data[s]['images'])*0.5)
            i=0
            t=tqdm(range(tot*len([k for k in obsts if k!=self.encode_label('terrain')])),unit='image',dynamic_ncols=True,colour='cyan',desc=s[0].upper()+s[1:].lower())
            while True in [obsts[k]<tot for k in obsts if k!=self.encode_label('terrain')]:
                lab=list(self.data[s]['encoded_labels'][i]).index(1)
                if lab!=self.encode_label('terrain') and obsts[lab]<tot:
                    image=copy.deepcopy(self.data[s]['images'][i])
                    label=copy.deepcopy(self.data[s]['labels'][i])
                    times=4
                    newsize=image.shape[0]*times
                    im,labim=self.resize(image,label,newsize)
                    # print(im,labim)
                    # print(i,n,tot,im is None)
                    if im is not None:
                        while labim is not None and 0 in labim and 1 in labim:
                            times+=1
                            newsize=image.shape[0]*times
                            im,labim=self.resize(image,label,newsize,norm=False)
                        if im is not None and 1 in labim:
                            image=im
                            label=labim
                            # if 0 in np.unique(label):
                            #     print('hello')
                            self.data[s]['images'].append(image)
                            self.data[s]['labels'].append(label)
                            self.data[s]['encoded_labels'].append(self.data[s]['encoded_labels'][i])
                            self.keys[s].append(self.keys[s][i])
                            # n+=1
                            obsts[lab]+=1
                            t.update(1)
                # print(obsts,end='\r')
                i=(i+1)%sizes[s]
            t.close()
    def augment_original(self,sizes):
        print()
        for s in self.data:
            delete=[]
            t=tqdm(range(sizes[s]),unit='image',dynamic_ncols=True,colour='cyan',desc=s[0].upper()+s[1:].lower())
            for i in t:
                # if random()<0.95:# and np.argmax(self.data[s]['encoded_labels'][i])==1:
                    image, label,enc=self.augment_image(s,i)
                    if image is not None:
                        # if np.argmax(self.data[s]['labels'][i])!=1:
                            self.data[s]['images'][i]=image
                            self.data[s]['labels'][i]=label
                            self.data[s]['encoded_labels'][i]=enc
                    else:
                        delete.append(i)
            t.close()
            self.data[s]['images']=[self.data[s]['images'][i] for i in range(len(self.data[s]['images'])) if i not in delete]
            self.data[s]['labels']=[self.data[s]['labels'][i] for i in range(len(self.data[s]['labels'])) if i not in delete]
            self.data[s]['encoded_labels']=[self.data[s]['encoded_labels'][i] for i in range(len(self.data[s]['encoded_labels'])) if i not in delete]
            self.keys[s]=[self.keys[s][i] for i in range(len(self.keys[s])) if i not in delete]
    def mix(self):
        sizes={s:len(self.data[s]['images']) for s in self.data}
        mixed_samples={k:sizes[k] for k in self.data}
        # mixed_samples={k:0 for k in mixed_samples}
        for s in self.data:
            # print(np.array(self.data[s]['images']).shape)
            # print(np.array(self.data[s]['labels']).shape)
            # print(np.array(self.data[s]['encoded_labels']).shape)
            t=tqdm(range(mixed_samples[s]),desc='Mixing '+s.lower()+' data',colour='cyan',unit='image')
            for n in t:
                i=randint(0,sizes[s])
                j=randint(0,sizes[s])
                while j==i or self.keys[s][i][0]==self.keys[s][j][0]:
                    j=randint(0,sizes[s])
                im,lab,enc,dom=self.mix_ims(s,i,j)
                # if enc.shape!=(50,50,3):
                #     print(enc.shape)
                if im is not None:
                    self.data[s]['images'].append(im)
                    self.data[s]['labels'].append(lab)
                    self.data[s]['encoded_labels'].append(enc)
                    self.keys[s].append(self.keys[s][dom])
            # print(np.array(self.data[s]['encoded_labels']).shape)
    def add_ground(self):
        print('\nAdding ground images ...')
        for s in self.data:
            ground=len(self.data[s]['images'])//2
            shape=[ground]+list(np.array(self.data[s]['images'][0]).shape)
            im=np.ones(shape)*(self._hp.ground_color+((random()*0.04)-0.02))
            im=im.tolist()
            labels=np.zeros(shape).tolist()
            enc=[self.encode_label('terrain')]*reduce(lambda x,y: x*y,shape)
            enc=np.reshape(enc,shape+[len(enc[0])]).tolist()
            ks=copy.deepcopy(self.keys[s][:ground]) #list(range(ground))
            self.data[s]['images']+=im
            self.data[s]['labels']+=labels
            self.data[s]['encoded_labels']+=enc
            self.keys[s]+=ks
    def check_sides(self,label):
        for n in range(label.shape[0]):
            if label[n][0]!=0 or label[n][label.shape[1]-1]!=0:
                return False
        for n in range(label.shape[1]):
            if label[0][n]!=0 or label[label.shape[0]-1][n]!=0:
                return False
        return True
    def augment(self):
        # self.data={k:{k1:self.data[k][k1][-10000:] for k1 in self.data[k]} for k in self.data}
        # sizes={s:1000 for s in self.data}
        # print(self._hp.ground_color)
        for s in self.data:
            # labs,counts_=np.unique(np.argmax(self.data[s]['encoded_labels'],-1),return_counts=True)
            # # labs,counts=np.unique(np.a)
            # counts_=list(counts_)
            # counts={k:v for k,v in zip(labs,counts_)}
            # print(counts)
            # m=max([counts[k] for k in counts if k!=self.encode_label('terrain')])
            # terrain=0
            indices=[]
            for i in range(len(self.data[s]['images'])):
                k=self.data[s]['encoded_labels'][i]
                if k!=self.encode_label('terrain'):# or terrain<m:
                    indices.append(i)
                # if k==self.encode_label('terrain'):
                #     terrain+=1
            # print(indices)
            self.data[s]['images']=[self.data[s]['images'][i] for i in indices]
            self.data[s]['labels']=[self.data[s]['labels'][i] for i in indices]
            self.data[s]['encoded_labels']=[self.data[s]['encoded_labels'][i] for i in indices]
            self.keys[s]=[self.keys[s][i] for i in indices]
        # self.data,self.keys=self.add_surface(self.data,self._hp.surface_samples)
        sizes={s:len(self.data[s]['images']) for s in self.data}
        # n_augs=10
        original=copy.deepcopy(self.data)
        self.n_aug(self._hp.n_augs,sizes)
        # self.balance(sizes)
        # self.add_full_obstacles(sizes)
        self.augment_original(sizes)
        self.balance(original)
        # self.mix()
        # self.add_ground()
        self.data,self.keys=self.add_surface(self.data,self._hp.surface_samples)

        print()
        for s in self.data:
            t=tqdm(range(len(self.data[s]['images'])),unit='image',dynamic_ncols=True,colour='cyan',desc='Normalizing '+s.capitalize())
            for i in t:
                self.data[s]['images'][i]=np.array(self.data[s]['images'][i])
                self.data[s]['images'][i]=self.data[s]['images'][i]-np.min(np.min(self.data[s]['images'][i],0,keepdims=True),-1,keepdims=True)
                # print(np.min(imgs),np.max(imgs),np.max(np.max(imgs,1,keepdims=True),-1,keepdims=True))
                max_im=np.max(np.max(self.data[s]['images'][i],0,keepdims=True),-1,keepdims=True)
                # print(max_im.shape)
                self.data[s]['images'][i]=np.where(max_im>0,self.data[s]['images'][i]/max_im,self.data[s]['images'][i])
                # print(np.min(self.data[s]['images'][i]),np.max(self.data[s]['images'][i]))
        
        # print('hi')
        print()
        for s in self.data:
            images=[]
            labels=[]
            positions=[]
            encs=[]
            keys=[]
            t=tqdm(range(len(self.data[s]['images'])),unit='image',dynamic_ncols=True,colour='cyan',desc='Cutting '+s.capitalize())
            for i in t:
                # if len(np.unique(self.data[s]['labels'][i]))>1:
                #     print('hello')
                    ims_imsize,labs_imsize=self.cut(self.data[s]['images'][i],self.data[s]['labels'][i],self._hp.image_size)
                # if self.check_sides(self.data[s]['labels'][i]):
                    for im,lab in zip(ims_imsize,labs_imsize):
                        ims,labs=self.cut(im,lab)
                        # vals=[np.unique(lab) for lab in labs]
                        pos=list(range(1,1+len(ims)))
                        pos=[[x]*self._hp.figsize**2 for x in pos]
                        # print(np.array(pos).shape,np.array(ims).shape)
                        pos=np.reshape(np.array(pos),np.array(ims).shape).tolist()

                        # keep=[len(vals[j])!=1 or vals[j][0]!=0 for j in range(len(ims))]
                        # ims=[x for j,x in enumerate(ims) if keep[j]]
                        # pos=[x for j,x in enumerate(pos) if keep[j]]
                        # labs=[x for j,x in enumerate(labs) if keep[j]]

                        enc=[self.data[s]['encoded_labels'][i]]*len(ims)
                        ks=[self.keys[s][i]]*len(ims)

                        images.append(ims)
                        labels.append(copy.deepcopy(labs))
                        positions.append(pos)
                        # # print(self.data[s]['encoded_labels'][i],[self.data[s]['encoded_labels'][i]]*len(ims))
                        # # labs=np.array(labs)
                        # # print(vals)
                        # # enc=[self.data[s]['encoded_labels'][i] if len(vals[j])!=1 or vals[j][0]!=0 else self.one_hot_encoding(self.encode_label('terrain')) for j in range(len(ims))]
                        # # print(np.array(labs).shape,vals.shape,np.array(enc).shape)                
                        encs.append(enc)
                        keys.append(ks)
                        # images+=ims
                        # labels+=labs
                        # positions+=pos
                        # encs+=enc
                        # keys+=ks
            self.data[s]['images']=images
            self.data[s]['labels']=labels
            self.data[s]['positions']=positions
            # print(encs)
            self.data[s]['encoded_labels']=encs
            # print(keys)
            self.keys[s]=keys
            self.indices[s]=list(range(len(self.data[s]['images'])))
            # lens=[len(x) for x in self.data[s]['images']]
            # self.data[s]['positions']=[list(range(1,l+1)) for l in lens] 
        
        # for s in self.data:
        #     encs=[]
        #     t=tqdm(range(len(self.data[s]['images'])),unit='image',dynamic_ncols=True,colour='cyan',desc='Cutting '+s.capitalize())
        #     for i in range(len(self.data[s]['encoded_labels'])):
        #         encs.append([])
        #         for j in range(len(self.data[s]['encoded_labels'][i])):
        #             vals=np.unique(self.data[s]['labels'][i][j])
        #             enc=np.array(self.data[s]['encoded_labels'][i][j])*(len(vals)!=1 or vals[0]!=0)
        #             encs[i].append(enc)
        #     self.data[s]['encoded_labels']=encs

        # print()
        # for s in self.data:
        #     lens=self.flat(s)
        #     t=tqdm(range(len(self.data[s]['images'])),unit='image',dynamic_ncols=True,colour='cyan',desc='Deleteting some terrain from '+s.capitalize())
        #     # labs,counts_=np.unique(np.argmax(self.data[s]['encoded_labels'],-1),return_counts=True)
        #     # # print('---------------')
        #     # # labs,counts=np.unique(np.a)
        #     # counts_=list(counts_)
        #     # counts={k:v for k,v in zip(labs,counts_)}
        #     # print(counts)
        #     l,c=np.unique(self.data[s]['labels'],return_counts=True)
        #     print(l,c)
        #     m=c[l.tolist().index(1)]//(self._hp.figsize**2)
        #     m*=0
        #     # m=0.5*(min([counts[k] for k in counts if k!=self.encode_label('terrain')])/(self._hp.figsize**2))+1
        #     # print(m,m*0.1)
        #     # m=0
        #     indices=[]
        #     # terrain=0
        #     keep=[]
        #     # print(lens)
        #     for i in t:
        #         k=np.argmax(self.data[s]['encoded_labels'][i])
        #         # print(np.array(self.data[s]['encoded_labels'][i]).shape,k)
        #         if k==self.encode_label('terrain'):
        #             indices.append(i)
        #         else:
        #             keep.append([i,True])
        #             # keep.append(True)
        #         # if k==self.encode_label('terrain'):
        #         #     terrain+=1
        #             # keep.append(False)
        #     # print(indices)
        #     shuffle(indices)
        #     indices=[[indices[i],True if i*(self._hp.figsize**2)<m else False] for i in range(len(indices))]
        #     # print(len([x for x in indices if x[1]]),m)
        #     keep+=indices
        #     keep.sort(key=lambda x: x[0])
        #     # print(keep)
        #     keep=[x[1] for x in keep]
        #     self.restore(s,lens)
        #     to_keep=[]
        #     start=0
        #     for l in lens:
        #         to_keep.append(keep[start:start+l])
        #         start+=l
        #     # print(to_keep)
        #     # print(self.keys[s])
        #     self.data[s]['images']=[[self.data[s]['images'][i][j] for j in range(len(self.data[s]['images'][i])) if to_keep[i][j]] for i in range(len(self.data[s]['images']))]
        #     self.data[s]['labels']=[[self.data[s]['labels'][i][j] for j in range(len(self.data[s]['labels'][i])) if to_keep[i][j]] for i in range(len(self.data[s]['labels']))]
        #     self.data[s]['encoded_labels']=[[self.data[s]['encoded_labels'][i][j] for j in range(len(self.data[s]['encoded_labels'][i])) if to_keep[i][j]] for i in range(len(self.data[s]['encoded_labels']))]
        #     self.data[s]['positions']=[[self.data[s]['positions'][i][j] for j in range(len(self.data[s]['positions'][i])) if to_keep[i][j]] for i in range(len(self.data[s]['positions']))]
        #     self.keys[s]=[[self.keys[s][i][j] for j in range(len(self.keys[s][i])) if to_keep[i][j]] for i in range(len(self.keys[s]))]
        #     # labs,counts_=np.unique(np.argmax(reduce(lambda x,y:x+y,self.data[s]['encoded_labels']),-1),return_counts=True)
        #     # print(counts_)

        print()
        # encs={}
        for s in self.data:
            # encs=copy.deepcopy(self.data[s]['encoded_labels'])
            # lens=self.flat(s)
            encs=[]
            t=tqdm(range(len(self.data[s]['images'])),unit='image',dynamic_ncols=True,colour='cyan',desc=s.capitalize())
            for i in t:
                # enc=copy.deepcopy(np.array(self.data[s]['labels'][i]))*np.argmax(copy.deepcopy(self.data[s]['encoded_labels'][i]))
                # encs.append(enc)
                encs.append([])
                for j in range(len(self.data[s]['images'][i])):
                #     # if 'valid' in s.lower():
                #     #     # print(lab,type(lab))
                #     #     print(self.data[s]['encoded_labels'][i][j],type(self.data[s]['encoded_labels'][i][j]))
                #     #     print(np.array(self.data[s]['labels'][i][j]))
                #     # print('-----------')
                #     # print(self.data[s]['encoded_labels'][i][j],np.max(self.data[s]['encoded_labels'][i][j]),np.argmax(self.data[s]['encoded_labels'][i][j]))
                #     # tmp=np.array(self.data[s]['labels'][i][j])*np.argmax(self.data[s]['encoded_labels'][i][j])
                #     # print(tmp)
                    # print(self.data[s]['encoded_labels'][i][j])
                    enc=copy.deepcopy(np.array(self.data[s]['labels'][i][j]))*self.data[s]['encoded_labels'][i][j]
                    enc=enc.tolist()
                    # for r in range(len(enc)):
                    #     for c in range(len(enc[r])):
                    #         enc[r][c]=self.one_hot_encoding(int(enc[r][c]))
                #     # print(enc.shape)
                    encs[i].append(enc)
                    # print(np.max(self.data[s]['encoded_labels'][i][j]))
            self.data[s]['encoded_labels']=encs
            # encoded=copy.deepcopy(self.data[s]['encoded_labels'])
            # encoded=np.array(encoded)
            # encoded=np.reshape(encoded,(encoded.shape[0],1,1,encoded.shape[-1]))
            # encoded=np.repeat(encoded,self._hp.figsize,axis=1)
            # encoded=np.repeat(encoded,self._hp.figsize,axis=2)
            # labels=np.expand_dims(np.array(self.data[s]['labels']),-1)
            # t=self.encode_label('terrain')
            # encoded=np.where(labels==t,self.one_hot_encoding(t),encoded)
            # encoded=encoded.tolist()
            # self.data[s]['encoded_labels']=encoded
            # self.restore(s,lens)            
        # for s in self.data:

        # mixed_samples={'train':5000,'valid':1000,'test':500}
        # shape=[ground]+list(np.array(self.data[s]['labels'][0]).shape)
        # self.lens={s:self.flat(s) for s in self.data}
        # print(self.keys)
        print('\nShuffling data ...')
        # for s in self.data: 
        for s in self.data:
            # self.lens[s]=self.flat(s)
            self.shuffle(s)
        # for s in self.data:
        #     for k in self.data[s]:
        #         print(len(self.data[s][k]))
        self.get_sizes()
        # self.restore('train',lens['train'])
        print('\nDone.')
    def flat(self,s):
        for k in self.data[s]:
            if isinstance(self.data[s][k],np.ndarray): 
                self.data[s][k]=self.data[s][k].tolist()
        if isinstance(self.keys[s],np.ndarray): 
            self.keys[s]=self.keys[s].tolist()
        lens=[len(x) for x in self.data[s]['images']]
        self.data[s]['images']=reduce(lambda x,y: x+y,self.data[s]['images'])
        self.data[s]['labels']=reduce(lambda x,y: x+y,self.data[s]['labels'])
        self.data[s]['encoded_labels']=reduce(lambda x,y: x+y,self.data[s]['encoded_labels'])
        if 'positions' in self.data[s]:
            self.data[s]['positions']=reduce(lambda x,y: x+y,self.data[s]['positions'])
        # print(self.keys[s])
        self.keys[s]=reduce(lambda x,y: x+y,self.keys[s])
        return lens
    def restore(self,s,lens):
        # self.data[s]['images']=[self.data[s]['images'][:lens[0]]]
        images=[]
        labels=[]
        encs=[]
        keys=[]
        positions=[]
        start=0
        for l in lens:
            images.append(self.data[s]['images'][start:start+l])
            labels.append(self.data[s]['labels'][start:start+l])
            encs.append(self.data[s]['encoded_labels'][start:start+l])
            if 'positions' in self.data[s]:
                positions.append(self.data[s]['positions'][start:start+l])
                # print(keys)
                # print(self.keys[s])
            keys.append(self.keys[s][start:start+l])
            start+=l
        self.data[s]['images']=images
        self.data[s]['labels']=labels
        self.data[s]['encoded_labels']=encs
        if 'positions' in self.data[s]:
            self.data[s]['positions']=positions
        self.keys[s]=keys        
    def cut(self,image,label=None,figsize=None):
        if figsize is None:
            figsize=self._hp.figsize
        if image.shape[0]<=figsize: return [image],[label]
        images=[]
        labels=[]
        shape=(figsize,figsize)
        for r in range(0,image.shape[0],figsize):
            for c in range(0,image.shape[1],figsize):
                im=np.ones(shape)*(self._hp.ground_color+((random()*0.04)-0.02))
                lab=np.zeros(shape)
                im_=copy.deepcopy(image[r:r+figsize,c:c+figsize])
                # print(im_.shape,im.shape)
                im[:min(figsize,im_.shape[0]),:min(figsize,im_.shape[1])]=copy.deepcopy(im_[:figsize,:figsize])
                if label is not None:
                    lab_=copy.deepcopy(label[r:r+figsize,c:c+figsize])
                    lab[:min(figsize,lab_.shape[0]),:min(figsize,lab_.shape[1])]=copy.deepcopy(lab_[:figsize,:figsize])
                images.append(copy.deepcopy(im))
                labels.append(copy.deepcopy(lab))
        return images, labels

