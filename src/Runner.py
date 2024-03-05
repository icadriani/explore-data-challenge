import torch
from Evals import Evals
import os
import torch.nn as nn
from tqdm import tqdm
from colorama import Fore
from statistics import mean
import numpy as np
from torch.optim import Adam
from sklearn.metrics import accuracy_score
from LossWrapper import LossWrapper

class Runner():
    def __init__(self,model,hp,model_name):
        self.model=model
        self.hp=hp
        self.model_name=model_name
        self.loss_lab=torch.nn.CrossEntropyLoss()
        self.loss_lab=LossWrapper(self.loss_lab,hp.n_classes,hp.classes_weights,requires_grad=True,device=hp.device)
        self.loss_optim=Adam(self.loss_lab.parameters(),lr=3e-4)
        self.optimizer=Adam(model.parameters(),lr=hp.lr)
        self.model.to(hp.device)
        self.evals=Evals(hp.eval_path)
        self.model_path=os.path.join(self.hp.model_path,model_name+'.h5')
        # self.sigmoid=nn.Sigmoid()
        self.eps=1e-8
        self.gamma_plus=0
        self.gamma_minus=2
        # self.alpha=torch.tensor(8,requires_grad=True,dtype=float).to(hp.device)
        # self.beta=torch.tensor(4,requires_grad=True,dtype=float).to(hp.device)
        # self.gamma=torch.tensor(4,requires_grad=True,dtype=float).to(hp.device)
        self.phi=0.97
    def loss(self,logits,target,alpha,beta,gamma):
        loss=self.model.alpha*self.loss_lab(logits[target==1],target[target==1])+self.model.beta*self.loss_lab(logits[target==2],target[target==2])+self.model.gamma*self.loss_lab(logits[target==0],target[target==0])
        # loss=self.loss_lab(logits,target,np.array([alpha,beta,gamma]))
        return loss
    def get_weights(self,predictions,target):
        # acc0=accuracy_score(target[target==0],predictions[target==0])
        acc1=accuracy_score(target[target==1],predictions[target==1])#,average='macro',zero_division=0)
        acc2=accuracy_score(target[target==2],predictions[target==2])#,average='macro',zero_division=0)
        alpha=self.alpha if acc1<self.phi else 1
        beta=self.beta if acc2<self.phi else 1
        # gamma=self.gamma if acc0<self.phi else 1
        gamma=1
        return alpha,beta,gamma
    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
    def load_model(self,model_path):
        self.model.load_state_dict(torch.load(model_path,map_location=self.hp.device))
    def loss_seg(self,pred,target):
        # pred=self.sigmoid(logits)
        pos=pred
        neg=1-pred
        pos=torch.clamp(pos,self.eps,1-self.eps)
        neg=torch.clamp(neg,self.eps,1-self.eps)
        y_pos=target
        y_neg=1-target
        loss_pos=y_pos*(neg**self.gamma_plus)*torch.log(pos)
        loss_neg=y_neg*(pos**self.gamma_minus)*torch.log(neg)
        loss=loss_pos+loss_neg
        # loss=loss/target.shape[-1]
        loss=torch.mean(loss)
        # print(loss)
        return -loss
    def train(self,dataset):
        losses=[]
        val_losses=[]
        metrics={'accuracy':[],'precision':[],'recall':[],'f1':[]}
        val_metrics={'accuracy':[],'precision':[],'recall':[],'f1':[]}
        alpha=1
        beta=1
        gamma=1
        for epoch in range(self.hp.epochs):
            self.model.train()
            dataset.train()
            dataset.shuffle()
            eloss=[]
            emetrics={k:[] for k in metrics}
            bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Fore.RESET)
            # print(alpha,beta,gamma)
            # if epoch%10==0:
            #     self.loss_lab.activate()
            #     self.loss_optim=Adam(self.loss_lab.parameters(),lr=3e-4)
            with tqdm(total=dataset.sizes['train'],unit='batch',dynamic_ncols=True,bar_format=bar_format) as t:
                t.set_description('Epoch '+str(epoch+1)+'/'+str(self.hp.epochs))
                for i in range(dataset.sizes['train']):
                    batch=dataset.get_batch(i)
                    inbatch=batch['input']
                    # target_seg=batch['target_segmentations']
                    target_lab=batch['target_labels']
                    # print(target_lab)
                    # target_lab=target_lab.argmax(-1)#,keepdims=True)
                    unpadded_len=batch['unpadded_len']
                    positions=batch['positions']
                    self.optimizer.zero_grad()
                    self.loss_optim.zero_grad()
                    logits_class=self.model(inbatch,positions)
                    # logits_seg=logits_seg[:unpadded_len]
                    logits_class=logits_class[:unpadded_len]
                    # target_seg=target_seg[:unpadded_len]
                    # print(target_lab)
                    target_lab=target_lab[:unpadded_len]
                    # loss_lab=self.loss_seg(logits_class,target_lab)
                    # print(logits_class.shape,target_lab.shape)
                    # shape=target_seg.shape
                    # target_lab_=target_seg.unsqueeze(-1)
                    # print(shape,target_lab_.shape,target_lab.shape,target_lab_.shape[-1],target_lab.shape[-1])
                    # target_lab_=target_lab_.repeat(1,1,1,target_lab.shape[-1])
                    # target_lab_=target_lab_.flatten(1)
                    # print(target_lab)
                    # target_lab=target_lab.flatten()
                    # target_seg=target_seg.flatten(1)
                    # logits_seg=logits_seg.flatten(1)
                    # logits_class_shape=logits_class.shape
                    # logits_class=logits_class.reshape((self.hp.batch_size*self.hp.figsize**2,self.hp.n_classes))
                    # print(logits_class.shape)
                    # print(target_lab)
                    # print(target_lab)
                    # print(logits_class.shape,target_lab.shape)
                    loss=self.loss_lab(logits_class,target_lab)#,alpha,beta,gamma)
                    # loss=alpha*self.loss_lab(logits_class[target_lab==1],target_lab[target_lab==1])+beta*self.loss_lab(logits_class[target_lab==2],target_lab[target_lab==2])+gamma*self.loss_lab(logits_class[target_lab==0],target_lab[target_lab==0])
                    # loss=alpha*self.loss_seg(logits_class.flatten(1)[target_lab_>0],target_lab.flatten(1)[target_lab_>0])+beta*self.loss_seg(logits_class.flatten(1)[target_lab_==0],target_lab.flatten(1)[target_lab_==0])
                    # loss_seg=beta*self.loss_seg(logits_seg[target_seg==0],target_seg[target_seg==0])+alpha*self.loss_seg(logits_seg[target_seg>0],target_seg[target_seg>0])
                    # loss+=loss_seg
                    # logits_class=logits_class.reshape(logits_class_shape)
                    # target_lab=target_lab.reshape(logits_class_shape[:-1])
                    # target_seg=target_seg.reshape(shape)
                    # logits_seg=logits_seg.reshape(shape)
                    # loss_seg=self.loss_seg(logits_seg,target_seg)
                    # loss=loss_lab+loss_seg
                    # loss=self.loss_seg(logits_class.flatten(1),target_lab.flatten(1))
                    loss.backward()
                    self.optimizer.step()
                    self.loss_optim.step()
                    curr_loss=loss.item()
                    predictions=self.model.predict_lab(logits_class)
                    # predictions_seg=self.model.predict_seg(logits_seg)
                    # print(np.min(predictions_lab),np.max(predictions_lab))
                    # predictions_lab=np.argmax(predictions_lab,-1)
                    # predictions_lab=np.expand_dims(predictions_lab,-1)
                    # predictions_lab=np.repeat(predictions_lab,predictions_seg.shape[-1],-1)
                    # predictions=predictions_seg*predictions_lab
                    # predictions*=predictions_seg
                    target=target_lab.cpu().detach().numpy()
                    # target=np.argmax(target,axis=-1)
                    # target_seg=target_seg.cpu().detach().numpy()
                    # print(np.min(target_lab),np.max(target_lab))
                    # print(np.min(predictions_lab),np.max(predictions_lab))
                    # print('----------------------------------')
                    # target=np.argmax(target,-1)
                    # target_lab=np.expand_dims(target_lab,-1)
                    # target_lab=np.repeat(target_lab,target_seg.shape[-1],-1)
                    # target=target_seg*target_lab
                    # print(target.tolist())
                    target=target.ravel()
                    predictions=predictions.ravel()
                    cmetrics=self.evals.metrics(target,predictions)
                    # alpha,beta,gamma=self.get_weights(predictions,target)
                    # if alpha==1:
                    #     # eps=0.3
                    #     # if acc1<eps or acc2<eps:
                    #     if acc0>eps:
                    #         alpha=self.alpha
                    #         # beta=1
                    # if beta==1:
                    target0=target==0
                    target1=target==1
                    target2=target==2
                    acc0=accuracy_score(target[target0],predictions[target0]) if len(target[target0])>0 and len(predictions[target0])>0 else 0
                    acc1=accuracy_score(target[target1],predictions[target1]) if len(target[target1])>0 and len(predictions[target1])>0 else 0
                    acc2=accuracy_score(target[target2],predictions[target2]) if len(target[target2])>0 and len(predictions[target2])>0 else 0
                    #     if acc1>eps and acc2>eps:
                    #         # alpha=1
                    #         beta=self.beta
                    # alpha=self.alpha if acc1<eps or acc2<eps else 1
                    # beta=self.beta if alpha==1 else 1
                    for k in emetrics:
                        emetrics[k].append(cmetrics[k])
                    eloss.append(curr_loss)
                    t.postfix='loss: '+str(round(curr_loss,4))+', acc: '+str(round(100*cmetrics['accuracy'],2))+', f1: '+str(round(100*cmetrics['f1'],2))+', 0: '+str(round(100*acc0,2))+' '+str(len([x for x in target0 if x]))+', 1: '+str(round(100*acc1,2))+' '+str(len([x for x in target1 if x]))+', 2: '+str(round(100*acc2,2))+' '+str(len([x for x in target2 if x]))+', alpha: '+str(self.loss_lab.weights[0].item())+', beta: '+str(self.loss_lab.weights[1].item())+', gamma: '+str(self.loss_lab.weights[2].item())
                    t.update(1)
                eloss=mean(eloss)
                losses.append(eloss)
                emetrics={k:mean(v) for k,v in emetrics.items()}
                for k in metrics:
                    metrics[k].append(round(100*emetrics[k],2))
                t.postfix='loss: '+str(round(eloss,4))+', acc: '+str(round(100*emetrics['accuracy'],2))+', f1: '+str(round(100*emetrics['f1'],2))
                t.bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.RESET)
            val_loss,val_metric=self.test(dataset,'valid',alpha,beta,gamma)
            val_losses.append(val_loss)
            for k in val_metrics:
                val_metrics[k].append(val_metric[k])
            self.save_model()
            print()
        self.save_model()
        self.evals.plot('Loss',losses,val_losses)
        self.evals.save_metric('loss',losses,val_losses)
        for k in metrics:
            self.evals.plot(k[0].upper()+k[1:],metrics[k],val_metrics[k])        
            self.evals.save_metric(k,metrics[k],val_metrics[k])
    def test(self,test_set,dataset_type='test',alpha=1,beta=1,gamma=1):
        self.model.eval()
        test_set.eval(dataset_type)
        bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Fore.RESET)
        with tqdm(total=test_set.sizes[dataset_type],unit='batch',dynamic_ncols=True,bar_format=bar_format) as t:
            if dataset_type=='test':
                t.set_description('Test')
            else:
                t.set_description('Validation')
            eloss=[]
            emetrics={'accuracy':[],'precision':[],'recall':[],'f1':[]}
            targets=[]
            predictions=[]
            with torch.no_grad():
                for i in range(test_set.sizes[dataset_type]):
                    batch=test_set.get_batch(i,dataset_type)
                    inbatch=batch['input']
                    target_seg=batch['target_segmentations']
                    target_lab=batch['target_labels']
                    # target_lab=target_lab.argmax(-1)#,keepdims=True)
                    unpadded_len=batch['unpadded_len']
                    positions=batch['positions']
                    logits_class=self.model(inbatch,positions)
                    # logits_seg=logits_seg[:unpadded_len]
                    logits_class=logits_class[:unpadded_len]
                    # target_seg=target_seg[:unpadded_len]
                    target_lab=target_lab[:unpadded_len]

                    # target_lab_=target_lab_.repeat(1,1,1,target_lab.shape[-1])
                    # target_lab_=target_lab_.flatten(1)
                    # target_lab=target_lab.flatten()
                    # # target_seg=target_seg.flatten(1)
                    # # logits_seg=logits_seg.flatten(1)
                    # logits_class_shape=logits_class.shape
                    # logits_class=logits_class.reshape((self.hp.batch_size*self.hp.figsize**2,self.hp.n_classes))
                    # print(logits_class.shape,target_lab.shape)
                    loss=self.loss_lab(logits_class,target_lab)#,alpha,beta,gamma)
                    # loss=alpha*self.loss_seg(logits_class.flatten(1)[target_lab_>0],target_lab.flatten(1)[target_lab_>0])+beta*self.loss_seg(logits_class.flatten(1)[target_lab_==0],target_lab.flatten(1)[target_lab_==0])
                    # loss_seg=beta*self.loss_seg(logits_seg[target_seg==0],target_seg[target_seg==0])+alpha*self.loss_seg(logits_seg[target_seg>0],target_seg[target_seg>0])
                    # loss+=loss_seg
                    # logits_class=logits_class.reshape(logits_class_shape)
                    # target_lab=target_lab.reshape(logits_class_shape[:-1])
                    # target_seg=target_seg.reshape(shape)

                    # shape=target_seg.shape
                    # target_lab_=target_seg.unsqueeze(-1)
                    # print(shape,target_lab_.shape,target_lab.shape,target_lab_.shape[-1],target_lab.shape[-1])
                    # target_lab_=target_lab_.repeat(1,1,1,target_lab.shape[-1])
                    # target_lab_=target_lab_.flatten(1)
                    # target_seg=target_seg.flatten(1)
                    # logits_seg=logits_seg.flatten(1)
                    # loss=alpha*self.loss_seg(logits_class.flatten(1)[target_lab_>0],target_lab.flatten(1)[target_lab_>0])+beta*self.loss_seg(logits_class.flatten(1)[target_lab_==0],target_lab.flatten(1)[target_lab_==0])
                    # loss_seg=beta*self.loss_seg(logits_seg[target_seg==0],target_seg[target_seg==0])+alpha*self.loss_seg(logits_seg[target_seg>0],target_seg[target_seg>0])
                    # loss+=loss_seg
                    # target_seg=target_seg.reshape(shape)
                    # logits_seg=logits_seg.reshape(shape)
                    # loss_seg=self.loss_seg(logits_seg[target_seg==0],target_seg[target_seg==0])+self.alpha*self.loss_seg(logits_seg[target_seg>0],target_seg[target_seg>0])
                    # loss_seg=self.loss_seg(logits_seg,target_seg)
                    # loss=loss_lab+loss_seg
                    curr_loss=loss.item()
                    prediction=self.model.predict_lab(logits_class)
                    # predictions_seg=self.model.predict_seg(logits_seg)
                    # predictions_lab=np.argmax(predictions_lab,-1)
                    # predictions_lab=np.expand_dims(predictions_lab,-1)
                    # predictions_lab=np.repeat(predictions_lab,predictions_seg.shape[-1],-1)
                    # prediction=predictions_seg*predictions_lab
                    # prediction*=predictions_seg
                    # if 1 in prediction:
                    #     print('hiiiii')
                    target=target_lab.cpu().detach().numpy()
                    # target=np.argmax(target,axis=-1)
                    # target_seg=target_seg.cpu().detach().numpy()
                    # target=np.argmax(target,-1)
                    # target_lab=np.expand_dims(target_lab,-1)
                    # target_lab=np.repeat(target_lab,target_seg.shape[-1],-1)
                    # target=target_seg*target_lab
                    target=target.ravel()
                    prediction=prediction.ravel()
                    metrics=self.evals.metrics(target,prediction)
                    # acc0=accuracy_score(target[target==0],prediction[target==0])
                    # acc1=accuracy_score(target[target==1],prediction[target==1])#,average='macro',zero_division=0)
                    # acc2=accuracy_score(target[target==2],prediction[target==2])#,average='macro',zero_division=0)
                    # alpha=self.alpha if acc1<self.phi else 1
                    # beta=self.beta if acc2<self.phi else 1
                    # gamma=self.gamma if acc0<self.phi else 1
                    # alpha,beta,gamma=self.get_weights(prediction,target)

                    # if metrics['f1']<0.7:
                    #     e=list(range(self.hp.figsize**2))
                    #     print(prediction[e])
                    #     print(target[e])
                    #     print(np.min(prediction[e]),np.max(prediction[e]))
                    #     print(np.min(target[e]),np.max(target[e]))
                    #     print(np.min(predictions_seg[0]),np.max(predictions_seg[0]))
                    #     print(np.min(target_seg[0]),np.max(target_seg[0]))
                    #     print(np.min(predictions_lab[0]),np.max(predictions_lab[0]))
                    #     print(np.min(target_lab[0]),np.max(target_lab[0]))
                    if dataset_type=='test':
                        targets+=target.tolist()
                        predictions+=prediction.tolist()
                    eloss.append(curr_loss)
                    for k in metrics:
                        emetrics[k].append(metrics[k])
                    t.postfix=' loss: '+str(round(curr_loss,4))+', acc: '+str(round(100*metrics['accuracy'],2))+', f1: '+str(round(100*metrics['f1'],2))
                    t.update(1)
                eloss=mean(eloss)
                emetrics={k:round(100*mean(v),2) for k,v in emetrics.items()}
                t.postfix=' loss: '+str(round(eloss,4))+', acc: '+str(emetrics['accuracy'])+', f1: '+str(emetrics['f1'])
                t.bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.RESET)
                if dataset_type=='test':
                    targets=test_set.decode_labels(targets)
                    predictions=test_set.decode_labels(predictions)
                    self.evals.heatmap(self.model_name,targets,predictions)
                    self.evals.heatmap(self.model_name,targets,predictions,norm='true')
                return eloss,emetrics
    def test_nac(self,data):
        # predictions={'images':[],'centers':[],'segmentations':[],'obstacle types':[],'indeces':[],'filenames':[],'segmented_filenames':[]}
        predictions=[]
        with torch.no_grad():
            bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.RESET)
            with tqdm(range(data.size),unit='batch',dynamic_ncols=True,bar_format=bar_format,desc='Predicting nac subimages') as t:
                for i in t:
                    batch=data.get_batch(i)
                    # pos=torch.LongTensor(np.array(list(range(self.hp.batch_size)))).to(self.hp.device)
                    logits_class=self.model(batch['images'],batch['positions'])
                    # prediction=self.model.predict_seg(logits_seg)
                    prediction=self.model.predict_lab(logits_class)
                    # predictions_lab=np.argmax(predictions_lab,-1)
                    # predictions_lab=np.expand_dims(predictions_lab,-1)
                    # predictions_lab=np.repeat(predictions_lab,prediction.shape[-1],-1)
                    # prediction=prediction*predictions_lab
                    # prediction=np.reshape(prediction,(self.hp.batch_size,self.hp.figsize,self.hp.figsize))
                    prediction=prediction.tolist()
                    predictions+=prediction
        return predictions




