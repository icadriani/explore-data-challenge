import torch
import torch.nn as nn
# from torchvision.ops import MLP

class Model(nn.Module):
    def __init__(self,hp):
        super(Model,self).__init__()
        self.hp=hp
        #activation
        self.relu=nn.ReLU()
        self.softmax=nn.Softmax(-1)
        self.sigmoid=nn.Sigmoid()
        #dropout
        self.dropout=nn.Dropout(hp.dropout)

        # self.alpha=nn.Parameter(torch.tensor(1,requires_grad=True,dtype=float))
        # self.beta=nn.Parameter(torch.tensor(1,requires_grad=True,dtype=float))
        # self.gamma=nn.Parameter(torch.tensor(1,requires_grad=True,dtype=float))

        if hp.vit:
            self.embed_ims=nn.Embedding(256,hp.embed_ims_size)
            # self.embed_pos=nn.Embedding(hp.figsize**2+1,hp.embed_ims_size)
            self.embed_pos=nn.Embedding(hp.npos+1,hp.embed_ims_size)
            self.norm=nn.LayerNorm([hp.figsize**2,hp.embed_ims_size])
            self.attn=nn.MultiheadAttention(hp.embed_ims_size,hp.num_heads,0.5,batch_first=True)
            # self.norm2=nn.LayerNorm([hp.figsize**2,hp.embed_ims_size])
            self.lin=nn.Linear(hp.embed_ims_size,hp.embed_ims_size)
            self.classifier=nn.Linear(hp.embed_ims_size*hp.figsize**2,hp.n_classes*hp.figsize**2)
            self.lstm=nn.LSTM(hp.embed_ims_size,hp.lstm_hidden,num_layers=hp.lstm_nlayers,batch_first=True,dropout=hp.dropout,bidirectional=hp.lstm_bidirectional)
        else:
            # convs
            self.conv1=nn.Conv2d(hp.npos,hp.conv1_nodes,hp.conv1_kernel,stride=hp.conv1_stride)
            self.conv2=nn.Conv2d(hp.conv1_nodes,hp.conv2_nodes,hp.conv2_kernel,stride=hp.conv2_stride)
            self.conv4=nn.Conv2d(hp.conv2_nodes,hp.conv4_nodes,hp.conv4_kernel,stride=hp.conv4_stride)
            self.conv5=nn.Conv2d(hp.conv4_nodes,hp.conv5_nodes,hp.conv5_kernel,stride=hp.conv5_stride)

            conv_kers=[hp.conv1_kernel,hp.conv2_kernel,hp.conv4_kernel,hp.conv5_kernel]
            conv_strides=[hp.conv1_stride,hp.conv2_stride,hp.conv4_stride,hp.conv5_stride]
            pool_kers=[hp.pool1_kernel,hp.pool2_kernel,hp.pool4_kernel,hp.pool5_kernel]
            pool_strides=[hp.pool1_stride,hp.pool2_stride,hp.pool4_stride,hp.pool5_stride]
            scale_factors=[1,1,hp.upsample1_scale_factor,hp.upsample2_scale_factor]
            convh=hp.figsize
            for i in range(len(conv_kers)):
                convh*=scale_factors[i]
                convh=(convh-conv_kers[i]+conv_strides[i])//(conv_strides[i])
                convh=(convh-pool_kers[i]+pool_strides[i])//(pool_strides[i])
            #pools
            self.pool1=nn.MaxPool2d(hp.pool1_kernel,hp.pool1_stride)
            self.pool2=nn.MaxPool2d(hp.pool2_kernel,hp.pool2_stride)
            self.pool4=nn.MaxPool2d(hp.pool4_kernel,hp.pool4_stride)
            self.pool5=nn.MaxPool2d(hp.pool5_kernel,hp.pool5_stride)
            #upsample
            self.upsample1=nn.Upsample(scale_factor=hp.upsample1_scale_factor)
            self.upsample2=nn.Upsample(scale_factor=hp.upsample2_scale_factor)
            self.upsample3=nn.Upsample(scale_factor=hp.upsample3_scale_factor)
            #dense
            # self.segmentation=nn.Linear(hp.conv5_nodes*convh**2,hp.figsize**2)
            self.classifier=nn.Linear(hp.conv5_nodes*convh**2,hp.n_classes*(hp.figsize**2))
            # self.classifier=nn.Linear(hp.figsize**2,hp.n_classes*hp.figsize**2)
        # self.pos=torch.arange(hp.figsize**2)
        # self.pos=self.pos.unsqueeze(0)
        # self.pos=self.pos.repeat(hp.batch_size,1)

    def transformer_encoder(self,data,pos):
        # print(x.min(),x.max())
        # print(data.shape,pos.shape)
        # data=torch.flatten(data,1)
        data=data.squeeze()
        x=data*255
        x=x.round().int()
        # x=torch.flatten(x,1)
        pos=self.embed_pos(pos)
        # pos=pos.unsqueeze(1)
        # print(x.min(),x.max())
        x=self.embed_ims(x)
        # print(pos.shape,x.shape)
        # print(data.shape,x.shape,pos.shape)
        data=data.unsqueeze(-1)
        data=data.repeat(1,1,1,1,x.shape[-1])
        # print(data.shape,x.shape,pos.shape)
        x=data+x+pos
        x=x.reshape((x.shape[0]*x.shape[1],x.shape[2]*x.shape[3],x.shape[4]))
        n=self.norm(x)
        for _ in range(self.hp.n_tranfs):
            # print(x.shape)
            x=x+self.attn(n,n,n)[0]
            n=self.norm(x)
            # print(n.shape)
            # n=n.unsqueeze(-1)
            # print(n.shape)
            x=x+self.lin(n)
            # x=x+self.norm(x)
            # print(n.shape)
            # x=x+n
            x=self.dropout(x)
            n=self.norm(x)
        return x

    def encoder(self,x):
        x=self.conv1(x)
        x=self.pool1(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.pool2(x)
        x=self.relu(x)
        return x
    def decoder(self,x):
        x=self.upsample1(x)
        x=self.conv4(x)
        x=self.pool4(x)
        x=self.relu(x)
        x=self.upsample2(x)
        x=self.conv5(x)
        x=self.pool5(x)
        x=self.relu(x)
        return x
    def forward(self,x,pos):
        # e=self.embed_ims((x*255).round().int())
        # e=e.reshape((e.shape[0],e.shape[4],e.shape[2],e.shape[3]))
        # x=x.repeat(1,e.shape[1],1,1)
        # x=x+e
        if self.hp.vit:
            x=self.transformer_encoder(x,pos)
            x=self.lstm(x)[0]
            x=x+self.norm(x)
            # x=self.decoder(x)
            x=x.reshape((self.hp.batch_size*self.hp.npos,self.hp.embed_ims_size*self.hp.figsize*self.hp.figsize))
        else:
            # x=x.squeeze()
            x=self.encoder(x)
            x=self.decoder(x)
            x=x.flatten(1)
        x=self.dropout(x)
        # x=torch.flatten(x,1)
        label=self.classifier(x)
        # print(label.shape)
        # seg=self.segmentation(x)
        # seg=seg.reshape((self.hp.batch_size,self.hp.figsize,self.hp.figsize))
        label=label.reshape((self.hp.batch_size,self.hp.npos,self.hp.figsize,self.hp.figsize,self.hp.n_classes))
        # print(label.shape)
        # for b in range(self.hp.batch_size):
        #     for r in range(self.hp.figsize):
        #         for c in range(self.hp.figsize):
        #             print(label[b][r][c])
        # seg=self.sigmoid(seg)
        # label=self.softmax(label)
        return label
    def predict_seg(self,x,lab=False):
        x=self.sigmoid(x)
        x=torch.where(x>=0.5,1,0)
        # zeros=torch.lt(x,0.5)
        # ones=torch.ge(x,0.5)
        # x=x.masked_fill_(ones,1)
        # x=x.masked_fill_(zeros,0)
        if lab:
            x=x+1
        x=x.cpu().detach().numpy()
        return x
    def predict_lab(self,x):
        x=self.softmax(x)
        x=torch.argmax(x,-1)#,keepdim=True)
        x=x.cpu().detach().numpy()
        # x=(x>0.5).astype(int)
        # x=x+1
        # print(x)
        return x