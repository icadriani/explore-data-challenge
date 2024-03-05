import torch
import torch.nn as nn
import numpy as np
import copy

class LossWrapper(nn.Module):
    def __init__(self,loss,n_classes=None,weights=None,requires_grad=True,device=None):
        super().__init__()
        self.loss=loss
        self.n_classes=n_classes
        if n_classes==1 or (n_classes is None and weights is None):
            weights=1
            self.n_classes=1
        elif weights is None:
            weights=[1]*n_classes
            self.n_classes=n_classes
        elif n_classes is None:
            self.n_classes=min(1,len(weights))
        # elif isinstance(weights,list):
        #     weights=np.array(weights)
        # if type(weights) in [int,float,list,np.ndarray]:
        #     weights=nn.ModuleList()
        if type(weights) in [int,float]:
            self.weights=torch.tensor(weights,dtype=float)
            if requires_grad:
                self.weights=nn.Parameter(self.weights)
            if device is not None:
                self.weights=self.weights.to(device)
        else:
            self.weights=[torch.tensor(w,dtype=float) for w in weights]
            if requires_grad:
                self.weights=[nn.Parameter(w) for w in self.weights]
            if device is not None:
                self.weights=[w.to(device) for w in self.weights]
            self.weights=nn.ParameterList(self.weights)
        self.__private_weights=[1]*len(weights)
        self.__private_index=-1
    def activate(self):
        # return
        if self.__private_index>=0 and self.__private_index<len(self.weights):
            # print(type(self.weights[self.__private_index]))
            self.weights[self.__private_index]=self.weights[self.__private_index]*10
            # self.__private_weights=copy.deepcopy(self.__private_weights)
            # self.__private_weights[self.__private_index]=10
            self.__private_index-=1
        # else:
        #     self.__private_weights=[1/sum(self.weights)]*len(self.weights)
    def forward(self,logits,target,first_epoch=False):
        logits=logits.flatten(0,-2)
        target=target.flatten(0)
        return 10*self.loss(logits,target)
        # print(self.weights)
        logits=logits.squeeze()
        if self.n_classes==1:
            return self.weights*self.loss(logits,target)
        # loss=[[w,i] for i,w in enumerate(self.weights)]
        # print(loss,len(self.weights))
        # print(loss,enumerate(self.weights))
        # print(logits.shape,target.shape)
        # self.weights=self.weights.clamp(0.1)
        # target=target.argmax(-1)
        itarget=[target==i for i in range(len(self.weights))]
        # print('--------------------------------------------------------------------')
        # print([int(len(target[itarget[i]])>0) for i,w in enumerate(self.weights)])
        # print([int(len(logits[itarget[i]])>0) for i,w in enumerate(self.weights)])
        # print([max(w,1)for i,w in enumerate(self.weights)])
        # print([self.loss(logits[itarget[i]],target[itarget[i]]) for i,w in enumerate(self.weights)])
        # loss=[max(w*(1-int(first_epoch)),1)*self.loss(logits[itarget[i]],target[itarget[i]]) for i,w in enumerate(self.weights) if len(target[itarget[i]])>0 and len(logits[itarget[i]])>0]
        # sw=sum(self.weights)
        # print([w for w in self.weights])
        loss=[[logits[t].flatten(0,-2),target[t].flatten(0)] for t in itarget]
        # print([[y.shape for y in x] for x in loss])
        loss=[max(1,w)*self.loss(loss[i][0],loss[i][1]) for i,w in enumerate(self.weights) if len(target[itarget[i]])>0 and len(logits[itarget[i]])>0]
        # print(loss)
        # loss=[w*l for w,l in zip(self.__private_weights,loss)]
        loss=[x if x>0 else abs(x)*10000 for x in loss]
        loss=sum(loss)
        # loss=self.loss(logits,target)
        # print(loss)
        # print()
        return loss