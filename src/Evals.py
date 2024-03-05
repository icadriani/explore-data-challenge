import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

# import matplotlib.pyplot as plt
# import numpy as np

# new cmap code from stackoverflow 
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

arr = np.linspace(0, 50, 100).reshape((10, 10))
# fig, ax = plt.subplots(ncols=2)

cmap = plt.get_cmap('hot')
new_cmap = truncate_colormap(cmap, 0.25, 0.8)
# ax[0].imshow(arr, interpolation='nearest', cmap=cmap)
# ax[1].imshow(arr, interpolation='nearest', cmap=new_cmap)
# plt.show()

class Evals():
    """ A class to plot result of the training and testing processes. This class is the same one from homework 1 with some tweaks. """
    def __init__(self,images_path):
        """  
            Initialization of the class parameters and if the given folder doesn't exits, it will create it.

            Inputs:
                images_path:str, path to the folder where to save the plots and heatmaps.
            Outputs:
                None
        """
        self.images_path=images_path
        self.linewidth=1.5
    def save_metric(self,metric,train,val):
        n_epochs=len(train)
        epochs=list(range(1,n_epochs+1))
        s='Epochs,Train,Validation\n'
        for i in range(n_epochs):
            s+=str(epochs[i])+','+str(train[i])+','+str(val[i])+'\n'
        with open(os.path.join(self.images_path,metric)+'.csv','w+') as f:
            f.write(s)
    def read_metric(self,metric):
        with open(os.path.join(self.images_path,metric)+'.csv','r') as f:
            values=f.read().split('\n')
        train=[]
        val=[]
        for e in values[1:]:
            e=e.split(',')
            train.append(e[1].strip())
            val.append(e[2].strip())
        return train,val
    def plot(self,metric,train,val=None):
        """  
            Plots the losses of each epoch.

            Inputs:
                model_name:str, whether the results regard predicate idetification, predicate diambiguation,
                                argument identification or argument disambiguation.        
                losses:List[float], list of training losses
                val_losses:List[float], Optional. List of validation losses.
            Outputs:
                None
        """
        plt.close('all')
        plt.figure()
        plt.title(metric.capitalize())
        epochs=[x+1 for x in range(len(train))]
        plt.plot(epochs,train,linewidth=self.linewidth,color='blue',label='Train')
        if val is not None:
            plt.plot(epochs,val,linewidth=self.linewidth,color='red',label='Validation')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.images_path,metric)+'.png')
        plt.close()
    def metrics(self,y_true,y_pred):
        """  
            Computes the precision,recall and f1-score percentages. 

            Inputs:
                y_true:List[float] or List[str], set of targets (true labels).
                y_pred:List[float] or List[str], set of predictions. The elements type must be the same of the targets.
            Outputs:
                None
        """
        accuracy=accuracy_score(y_true,y_pred)
        precision=precision_score(y_true,y_pred,average='macro',zero_division=0)
        recall=recall_score(y_true,y_pred,average='macro',zero_division=0)
        f1=f1_score(y_true,y_pred,average='macro',zero_division=0)
        return {'accuracy':accuracy,'precision':precision,'recall':recall,'f1':f1}
    def heatmap(self,model_name,y_true,y_pred,name='Heatmap',norm=None):
        """  
            Computes the confusion matrix and visualizes it as a heatmap.
            
            Inputs:
                model_name:str, whether the results regard predicate idetification, predicate diambiguation,
                                argument identification or argument disambiguation.   
                name:str, title to give to the heatmap. Heatmaps of different datasets may have different names.
                y_true:List[str], list of the targets. Note: the tick labels will be computed from these.
                y_pred: List[str], list of predictions.
                norm:str, confusion matrix normalization type. "true" to normalize along the true labels. Note that if this is not None the
                        heatmap will show the percentages by multiplying the normalize value by 100.
            Outputs:
                None
        """
        cm=confusion_matrix(y_true,y_pred,normalize=norm)
        if norm is not None:
            cm*=100
        fig, ax = plt.subplots()
        # cmap=plt.cm.hot
        im = ax.imshow(cm,cmap=new_cmap)
        cbar = ax.figure.colorbar(im, ax=ax,cmap=new_cmap,fraction=0.0475,pad=0.005)
        cbar.ax.set_ylabel('', rotation=-90, va="bottom")
        labels=np.unique(y_true)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        rotation=45 if len(labels)>2 else 0
        ax.set_xticklabels(labels)#,rotation=rotation)
        ax.set_yticklabels(labels)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                text=ax.text(j, i, round(cm[i, j],2), ha="center", va="center", color="black")
        ax.set_title(name)
        plt.tight_layout()
        normalized='nomalized' if norm is not None else 'cases'
        plt.savefig(os.path.join(self.images_path,model_name+'_'+normalized+'_heatmap.png'))

