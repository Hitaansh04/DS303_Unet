import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self,in_channels :int ,out_channels :int):
        super(DoubleConv,self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels,out_channels,3,1,1,bias = False),
                                  nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True),
                                  nn.Conv2d(out_channels,out_channels,3,1,1,bias = False),
                                  nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True),
                                  )
    def forward(self,x):
        return self.conv(x)
        

class Unet(nn.Module):
    def __init__(self,in_channels = 3 ,out_channels = 1,features = [64,128,256,512],):
        super(Unet,self).__init__()
        self.up = nn.ModuleList()
        self.down = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        

        # Down Part

        for feature in features:
            self.down.append(DoubleConv(in_channels,feature))
            in_channels = feature


        # Up Part
        for feature in reversed(features):
            self.up.append(nn.ConvTranspose2d(feature*2,feature,kernel_size=2,stride=2,
                           )
            )
            self.up.append(DoubleConv(feature*2,feature))
        
            

        # Bottom Part
        self.bottom = DoubleConv(features[-1],features[-1]*2)
        self.last_conv = nn.Conv2d(features[0],out_channels,kernel_size=1)

    def forward(self,x):
        concats = []
        for downs in self.down:
            x = downs(x)
            concats.append(x)
            x = self.pool(x)

        x = self.bottom(x)
        concats = concats[::-1]
            # Up and double conv , hence 2 steps in a single iteration
        for index in range(0,len(self.up),2):
            x = self.up[index](x)
            concat = concats[index//2]
            if x.shape != concat.shape:
                concat = torchvision.transforms.functional.center_crop(concat, [x.shape[2], x.shape[3]])
            concatenated = torch.cat((concat,x),dim=1)
            x = self.up[index+1](concatenated)

        return self.last_conv(x)

def test():
    x = torch.randn((3,1,160,160))
    model = Unet(in_channels=1,out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)

if __name__ == "__main__":
         test()
