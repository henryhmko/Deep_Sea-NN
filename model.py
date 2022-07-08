import torch
import torch.nn as nn

def conv_block(in_channels, features, dropout = False):
    layers = []
    
    layers.append(nn.Conv2d(in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False))
    
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Conv2d(in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False))

    if dropout:
        layers.append(nn.Dropout2d(0.3))  #Droput included when dropout=True
    
    layers.append(nn.ReLU(inplace=True))
    
    return nn.Sequential(*layers)



class Unet(nn.Module):
    
    def __init__(self, in_channels=3, out_channels=3, init_features=32):
        super(Unet, self).__init__()
        
        features = init_features
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder1 = conv_block(in_channels, features) #3 => 32
        #maxpool
        self.encoder2 = conv_block(features, features * 2) #32 => 64
        #maxpool
        self.encoder3 = conv_block(features * 2, features * 4, dropout=True) #64 => 128
        #maxpool
        
        self.bottleneck = conv_block(features * 4, features * 8, dropout=True) # 128 => 256
        
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2) # 256 => 128
        #concat skip layer
        self.decoder3 = conv_block(features * 8, features * 4)
        
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2) #128 => 64
        #concat skip connection
        self.decoder2 = conv_block(features * 4, features * 2)
        
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2) #64 => 32
        #concat skip connection
        self.decoder1 = conv_block(features * 2, features)
        
        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)
        
        
    def forward(self, x):
        
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        
        bottleneck = self.bottleneck(self.pool(enc3))
        
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return self.conv(dec1)
