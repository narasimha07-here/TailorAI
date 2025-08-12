import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn as nn

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)       
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class SequenceModel(nn.Module):
    def __init__(self, in_features, dropout):
        super(SequenceModel, self).__init__()
        

        self.firstseq=nn.Sequential(nn.Conv2d(in_features*2,32,kernel_size=(3,3),padding=(1,1),stride=(2,2),bias=False),
                                    nn.ReLU(),
                                   nn.Dropout2d(dropout),
                                    GeM(), #pooling
                                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(64),
                                    nn.Dropout2d(dropout),
                                    nn.Conv2d(64,128,kernel_size=3, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),)
        
        self.firstfinal=nn.Sequential(nn.Conv2d(128,64,kernel_size=(3,3),padding=(1,1),stride=(2,2),bias=False))
                                      
        self.secondseq=nn.Sequential(nn.Conv2d(64,64,kernel_size=(3,3),padding=(1,1),stride=(2,2),bias=False),
                                     nn.ReLU(),
                                     GeM(), #pooling
                                     nn.Conv2d(64,128,kernel_size=(3,3),padding=(1,1),stride=(1,1),bias=False),
                                     nn.MaxPool2d(kernel_size=(1,1),stride=(1,1)),
                                     nn.Conv2d(128,256,kernel_size=(3,3),padding=(1,1),stride=(1,1),bias=True),
                                     nn.Conv2d(256,512,kernel_size=(2,2),padding=(1,1),stride=(2,2),bias=False),
                                     nn.MaxPool2d(kernel_size=(1,1),stride=(1,1)),
                                     nn.Flatten(),
                                     nn.Dropout(dropout),
                                     nn.ReLU(),
                                     nn.BatchNorm1d(512),
                                     nn.Dropout(dropout))
        
        self.final=nn.Linear(512,14)

    def forward(self,x1,x2):

        x=torch.cat([x1,x2],dim=1)
        
        x_first=self.firstseq(x)
        x_firstfinal=self.firstfinal(x_first)
        x_second=self.secondseq(x_firstfinal)
        x_final=self.final(x_second)

        return x_final
