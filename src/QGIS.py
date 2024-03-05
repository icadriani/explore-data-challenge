from AbstractDataset import AbstractDataset
import os
import numpy as np
from math import ceil, sqrt
import torch
from tqdm import tqdm
# import imageio as iio
from colorama import Fore
import matplotlib.pyplot as plt
from functools import reduce
from PIL import Image
import copy
import cv2

class QGIS(AbstractDataset):
    def __init__(self,hp):
        super(QGIS,self).__init__(hp)
        # self._hp=hp
        self.path=os.path.join(self._hp.data_path,'EXPLORE_ML_DataChallenge_QGIS_project','QGIS')
        self.nac_path=os.path.join(self.path,'NAC_Archytas.tif')
        self.output_images=os.path.join(self._hp.eval_path,'step_2_output')
        if not os.path.exists(self.output_images):
            os.mkdir(self.output_images)
        self.table_path=os.path.join(self.output_images,'detections.tsv')
        self.load()
        self.cut_pipeline(self._hp.image_size,self._hp.figsize)
    def cut_pipeline(self,image_size,figsize,nac=None):
        ims=[]
        centers=[]
        positions=[]
        imsize_ims,imsize_centers=self.cut(image_size,image_size,nac=nac)
        bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.RESET)
        t=tqdm(total=len(imsize_ims),unit='image',dynamic_ncols=True,bar_format=bar_format,desc='Cutting')
        # i=0
        self.small_centers=None
        for im,c in zip(imsize_ims,imsize_centers):
            curr_ims,curr_c=self.cut(figsize,figsize,nac=im,verbose=False)
            pos=list(range(1,len(curr_ims)+1))
            pos=[[x]*figsize**2 for x in pos]
            pos=np.reshape(np.array(pos),np.array(curr_ims).shape).tolist()
            # print(c)
            # print(curr_c)
            if self.small_centers is None:
                self.small_centers=curr_c
            curr_c=[[c[0]-im.shape[0]//2+cc[0],c[1]-im.shape[1]//2+cc[1]] for cc in curr_c]
            # print(curr_c)
            # if i==2:
            #     exit(0)
            # i+=1
            # print(curr_c)
            # ims+=curr_ims
            # centers+=curr_c
            # positions+=pos
            ims.append(curr_ims)
            centers.append(curr_c)
            positions.append(pos)
            t.update(1)
        t.close()
        self.images=ims
        self.centers=centers
        self.positions=positions
        self.size=ceil(len(self.images)/self._hp.batch_size)
        # print(np.array(self.images).shape)
    def get_batch(self,i):
        start=i*self._hp.batch_size
        finish=start+self._hp.batch_size
        ims=self.images[start:finish]
        ims+=[np.zeros(np.array(ims[0]).shape)]*(self._hp.batch_size-len(ims))
        ims=torch.FloatTensor(np.array(ims)).to(self._hp.device)#.unsqueeze(1)
        pos=self.positions[start:finish]
        pos+=[np.zeros(np.array(pos[0]).shape)]*(self._hp.batch_size-len(pos))
        pos=torch.LongTensor(np.array(pos)).to(self._hp.device)
        batch={'images':ims,'positions':pos}
        return batch
    def load(self):
        Image.MAX_IMAGE_PIXELS=None
        self.nac = Image.open(self.nac_path)
        self.nac=np.array(self.nac)
        # self.nac_og=copy.deepcopy(self.nac)
        self.eps=1e10
        # self.nanvalue=self.nac[0][0]
        self.nac=np.where(self.nac<-self.eps,np.nan,self.nac)
        self.nac=np.where(self.nac>self.eps,np.nan,self.nac)
        minnac=np.nanmin(self.nac)
        self.nac=np.where(np.isnan(self.nac),minnac,self.nac)
        self.nac=self.nac-np.nanmin(self.nac)
        self.nac=self.nac/np.nanmax(self.nac)
        # self.nac_norm=copy.deepcopy(self.nac)
        # self.nac=Image.fromarray(self.nac)
        # perc=0.25
        # self.nac=self.nac.resize((round(perc*self.nac.size[0]),round(perc*self.nac.size[1])))
        # self.nac=np.array(self.nac)
        # self.nac=self.nac-np.nanmin(self.nac)
        # self.nac=self.nac/np.nanmax(self.nac)
    def cut(self,figsize,stride,nac=None,verbose=True):
        self.stride=stride
        if nac is None: nac=copy.deepcopy(self.nac)
        ims=[]
        centers=[]
        if verbose:
            bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.RESET)
            t=tqdm(total=ceil(nac.shape[0]/stride)*ceil(nac.shape[1]/stride),unit='image',dynamic_ncols=True,bar_format=bar_format,desc='Cutting')
        for r in range(figsize//2,nac.shape[0],stride):
            for c in range(figsize//2,nac.shape[1],stride):
                im=nac[max(0,r-figsize//2):r+figsize//2,max(0,c-figsize//2):c+figsize//2]
                # if im.shape[0]==figsize:
                    # if im.shape[1]<figsize:
                if im.shape[0]<figsize or im.shape[1]<figsize:
                    pad=np.zeros((figsize,figsize))
                    sr=0 if r>figsize//2 else -im.shape[0]
                    er=im.shape[0] if r>figsize//2 else figsize
                    sc=0 if c>figsize//2 else -im.shape[1]
                    ec=im.shape[1] if c>figsize//2 else figsize
                    if c<figsize//2:
                        # im=np.concatenate((pad,im),axis=-1)
                        pad[sr:er,sc:ec]=im
                    else:
                        # im=np.concatenate((im,pad),axis=-1)
                        pad[sr:er,sc:ec]=im
                    im=pad
                ims.append(im)
                center=[r,c]
                centers.append(center)
                if verbose:
                    t.update(1)
        if verbose:
            t.close()
        # self.images=ims
        # self.centers=centers
        # self.size=ceil(len(self.images)/self._hp.batch_size)
        return ims, centers
    def get_segmentation(self,label):
        xy=[]
        for r in range(label.shape[0]):
            for c in range(label.shape[1]):
                if label[r,c]!=0:
                    zero=False
                    for x in range(min(0,r-1),max(r+1,label.shape[0])):
                        for y in range(min(0,c-1),max(c+1,label.shape[1])):
                            if x!=r and y!=c and label[x,y]==0:
                                zero=True
                                xy.append([r,c])
                                break
                        if zero:
                            break
        segmentation=[xy[0]]
        xy=xy[1:]
        while len(xy)>0:
            last=segmentation[-1]
            d=[sqrt((e[0]-last[0])**2+(e[1]-last[1])**2) for e in xy]
            m=d.index(min(d))
            segmentation.append(xy[m])
            xy=xy[:m]+xy[m+1:]
        return segmentation
    def get_obstacle_type(self,label):
        ls={}
        for l in self.enc2class:
            if l!=0:
                ls[l]=np.count_nonzero(label==l)
        values=list(ls.values())
        enc=list(ls.keys())[values.index(max(values))]
        dec=self.enc2class[enc]
        return dec
    def filter(self,images,labels,centers):
        labels=labels[:len(images)]
        ls=[]
        ims=[]
        cs=[]
        for i,label in enumerate(labels):
            label=np.array(label)
            if np.any(label>0):
                ls.append(label)
                ims.append(images[i])
                cs.append(centers[i])
        return ls,ims,cs
    def process_pred(self,preds):
        preds,ims,centers=self.filter(self.images,preds,self.centers)
        bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Fore.RESET)
        t=tqdm(preds,unit='image',dynamic_ncols=True,bar_format=bar_format,desc='Process')
        segs=[]
        obs=[]
        for l in t:
            segs.append(self.get_segmentation(l))
            obs.append(self.get_obstacle_type(l))
        idxs=list(range(1,len(preds)+1))
        filenames=[os.path.join(self.output_images,'NAC_Archytas_'+str(i)+'.jpeg') for i in idxs]
        segmented_filenames=[os.path.join(self.output_images,'segmented_NAC_Archytas_'+str(i)+'.jpeg') for i in idxs]
        return ims,centers,segs,obs,idxs,filenames,segmented_filenames
    def save_table(self,centers,segs,obs,idxs,filenames,segmented_filenames):
        t=[['id','filename','segmented filename','center x','center y','obstacle type','segmentation']]
        segs=[reduce(lambda x,y: x+y,s) for s in segs]
        t+=[[str(idxs[i]),filenames[i].replace(self._hp.src_path,''),segmented_filenames[i].replace(self._hp.src_path,''),str(centers[i][0]),str(centers[i][1]),obs[i],str(segs[i])] for i in range(len(idxs))]
        t='\n'.join(['\t'.join(ti) for ti in t])
        with open(self.table_path,'w+') as f:
            f.write(t)
    def read_table(self):
        with open(self.table_path,'r') as f:
            t=f.read().strip().split('\n')
        t=[ti.split('\t') for ti in t]
        idxs=[int(x[0]) for x in t]
        filenames=[x[1] for x in t]
        segmented_filenames=[x[2] for x in t]
        centers=[[int(x[3]),int(x[4])] for x in t]
        obs=[x[5] for x in t]
        segs=[[int(y) for y in x[6][1:-1]] for x in t]
        segs=[segs[i:i+2] for i in range(0,len(segs)-2,2)]
        return centers,segs,obs,idxs,filenames,segmented_filenames
    def save_images(self,images,preds,filenames,segmented_filenames):
        for i in range(len(images)):
            plt.imsave(filenames[i],images[i],cmap='gray')
            pred=np.array(preds[i])
            plt.imsave(segmented_filenames[i],pred)
    def cut_tr_size(self,left,top,shape):
        # nac=copy.deepcopy(self.nac)
        nac=np.zeros((self.nac.shape[0]+left,self.nac.shape[1]+top))
        nac[left:,top:]=copy.deepcopy(self.nac)
        # print()
        # print(new_nac.shape)
        nac=Image.fromarray(nac)
        # print(nac.size)
        shape=(shape[1],shape[0])
        nac=nac.resize(shape)
        nac=np.array(nac)
        nac=nac-np.min(nac)
        nac=nac/np.max(nac)
        self.cut_pipeline(self._hp.image_size,self._hp.figsize,nac=nac)
    def cut_tr(self,left,top):
        nac=copy.deepcopy(self.nac)
        new_nac=np.zeros((nac.shape[0]+left,nac.shape[1]+top))
        new_nac[left:,top:]=nac
        # self.nac=nac
        # if cut:
        new_nac=new_nac-np.min(new_nac)
        new_nac=new_nac/np.max(new_nac)
        self.cut_pipeline(self._hp.image_size,self._hp.figsize,nac=new_nac)
        # return nac
    def cut_size(self,shape,nac=None):
        if nac is None:
            nac=self.nac
        nac=copy.deepcopy(nac)
        nac=Image.fromarray(nac)
        nac=nac.resize(shape)
        nac=np.array(nac)
        nac=nac-np.min(nac)
        nac=nac/np.max(nac)
        self.cut_pipeline(self._hp.image_size,self._hp.figsize,nac=nac)
        # return nac
    def get_og_size(self,nac,left=0,top=0):
        if nac.shape[0]!=self.nac.shape[0]+left or nac.shape[1]!=self.nac.shape[1]+top:
            nac=Image.fromarray(nac)
            nac=nac.resize((self.nac.shape[1]+left,self.nac.shape[0]+top))
            nac=np.array(nac)
            print()
            print(np.unique(nac))
            nac=np.round(nac)
            print(np.unique(nac))
        nac=nac[-self.nac.shape[0]:,-self.nac.shape[1]:]
        return nac
    def compare(self,im1,im2):
        return np.where(im1==im2,im1,0)
    def unify_segmentation(self,segs,centers=None,figsize=None,shape=None,save=True,verbose=True):
        if centers is None:
            centers=self.centers
        if figsize is None:
            figsize=self._hp.figsize
        if shape is None:
            shape=self.nac.shape
        shape_pred=np.array(segs).shape
        segs=np.reshape(np.array(segs),(shape_pred[0]*shape_pred[1],shape_pred[2],shape_pred[3]))
        shape_cent=np.array(centers).shape
        centers=np.reshape(np.array(centers),(shape_cent[0]*shape_cent[1],shape_cent[2]))
        unify=np.zeros((shape[0],shape[1]))
        if verbose:
            bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.RESET)
            t=tqdm(range(len(centers)),unit='image',dynamic_ncols=True,bar_format=bar_format,desc='Unifying')
        for xy,seg in zip(centers,segs):
            seg=np.array(seg)
            x=xy[0]
            y=xy[1]
            sx=max(x-figsize//2,0)
            sy=max(y-figsize//2,0)
            fx=min(x+figsize//2,shape[0])
            fy=min(y+figsize//2,shape[1])
            mancax=figsize-(fx-sx)
            mancay=figsize-(fy-sy)
            ssx=0 if mancax==0 or x<shape[0]//2 else mancax
            sfx=figsize if mancax==0 or x>shape[0]//2 else figsize-mancax
            ssy=0 if mancay==0 or y<shape[1]//2 else mancay
            sfy=figsize if mancay==0 or y>shape[1]//2 else figsize-mancay
            # print(unify[sx:fx,sy:fy].shape,seg[ssx:sfx:,ssy:sfy].shape)
            # print(sx,sy,fx,fy)
            # print(ssx,ssy,sfx,sfy)
            unify[sx:fx,sy:fy]=seg[ssx:sfx,ssy:sfy]
            # unify=unify[-self.nac.shape[0]:,-self.nac.shape[1]:]
            if verbose:
                t.update(1)
        if save:
            self.save(unify)
        return unify
    def save(self,unify,processed=False):
        # unify=unify[-self.nac.shape[0]:,-self.nac.shape[1]:]
        if unify.shape[0]!=self.nac.shape[0] or unify.shape[1]!=self.nac.shape[1]:
            unify=Image.fromarray(unify)
            unify=unify.resize((self.nac.shape[1],self.nac.shape[0]))
            unify=np.array(unify)
        unify=np.where(self.nac==0,2.5,unify)
        # unify=np.where(np.isnan(unify),0.0,unify)
        # unify=np.where(self.nac_og<-self.eps,self.nac_og,unify)
        # unify=np.where(self.nac_og>self.eps,self.nac_og,unify)
        path=os.path.join(os.path.split(self.output_images)[0],'segmented_'+os.path.split(self.nac_path)[-1])
        if processed:
            path=path.replace('.tif','_processed.tif')
        # unify=self.clean(unify)
        plt.imsave(path,unify,cmap='gray',format='tiff')
        return unify
    def downsampling(self):
        nac=cv2.pyrDown(self.nac)
        self.cut_pipeline(self._hp.image_size,self._hp.figsize,nac)
        return nac.shape
    def unify_pipeline(self,segs,centers=None,shape=None,save=True):
        if centers is None:
            centers=self.centers
        segmentations=[]
        # centri=[]
        i=0
        # print(len(segs))
        while i*self._hp.npos<len(segs):
            im_segmentations=segs[i*self._hp.npos:i*self._hp.npos+self._hp.npos]
            # if i==0:
            #     im_centers=centers[i*self._hp.npos:i*self._hp.npos+self._hp.npos]
            # print(im_centers)
            i+=1    
            seg=self.unify_segmentation(im_segmentations,self.small_centers,shape=(self._hp.image_size,self._hp.image_size),save=False,verbose=False)
            segmentations.append(seg)
            # centri.append(im_centers[0])
        # print(len(segmentations),len(centers))
        unify=self.unify_segmentation(segmentations,centers,figsize=self._hp.image_size,shape=shape,save=save)
        return unify

        # im_centers=

