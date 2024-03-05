import os
import torch
from math import ceil

class HParams():
    def __init__(self,src_path):

        # filesystem
        self.src_path=src_path
        self.model_path=src_path.replace('src','model')
        os.makedirs(self.model_path,exist_ok=True)
        self.eval_path=src_path.replace('src','results')
        os.makedirs(self.eval_path,exist_ok=True)
        self.data_path=src_path.replace('src','data')

        self.epochs=50
        self.lr=3e-5
        # self.lr=1e-3
        self.classes_weights=[1,8,3]

        self.n_augs=1
        self.figsize=50
        self.image_size=50
        self.npos=(ceil(self.image_size/self.figsize))**2
        self.batch_size=8

        self.surface_samples=10000
        # self.surface_samples=0

        self.ground_color=0.5

        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.device='cpu'

        self.ignore_index=0

        self.vit=False

        # model params

        # convs

        self.conv1_nodes=32
        self.conv1_kernel=5
        self.conv1_stride=2

        self.conv2_nodes=64
        self.conv2_kernel=3
        self.conv2_stride=2

        self.conv3_nodes=128
        self.conv3_kernel=3
        self.conv3_stride=1

        self.conv4_nodes=128
        self.conv4_kernel=3
        self.conv4_stride=1

        self.conv5_nodes=256
        self.conv5_kernel=3
        self.conv5_stride=2

        self.conv6_nodes=1024
        self.conv6_kernel=3
        self.conv6_stride=2

        # pools

        self.pool1_kernel=3
        self.pool1_stride=1

        self.pool2_kernel=3
        self.pool2_stride=2

        self.pool3_kernel=3
        self.pool3_stride=2

        self.pool4_kernel=3
        self.pool4_stride=2

        self.pool5_kernel=3
        self.pool5_stride=2

        self.pool6_kernel=3
        self.pool6_stride=2

        # upsample

        self.upsample1_scale_factor=4
        self.upsample2_scale_factor=4
        self.upsample3_scale_factor=2

        # ViT

        self.embed_ims_size=128
        self.num_heads=4
        self.n_tranfs=1
        self.dropout=0.5

        self.lstm_hidden=self.embed_ims_size//2
        self.lstm_nlayers=1
        self.lstm_bidirectional=True

    def set_n_classes(self,classes):
        self.n_classes=classes
    def set_ground_color(self,color):
        self.ground_color=color