import torch
import torch.nn as nn
from utils import comm

def build_encoder(args):
    if args.model == 'HSI_CNNs':
        return HSI_CNNs()
    else:
        raise NotImplementedError

class HSI_CNNs(nn.Module):
    def __init__(self, num_classes=15):
        super(HSI_CNNs, self).__init__()
        # block[Conv, BN, (Pool), ReLU]
        self.patch_shape = (7, 7, 144)
        self.encoder_layer1 = nn.Sequential(
            nn.Conv2d(144, 16, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(16, momentum=0.9),
            nn.ReLU(inplace=True),
        )
        self.encoder_layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, stride=1, padding='same'),
            nn.BatchNorm2d(32, momentum=0.9),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.encoder_layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.ReLU(inplace=True),
        )
        self.encoder_layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding='same'),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),# same padding
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.encoder_layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=1, padding='same'),
            nn.Flatten(),
            # nn.Linear(256, 256),
            # nn.ReLU(),
            # nn.Linear(256, num_classes)
        )
        self.encoder = nn.ModuleList([
            self.encoder_layer1, 
            self.encoder_layer2, 
            self.encoder_layer3, 
            self.encoder_layer4,
            self.encoder_layer5
            ])
        # self.encoder = nn.Sequential(
        #     self.encoder_layer1,
        #     self.encoder_layer2,
        #     self.encoder_layer3,
        #     self.encoder_layer4,
        #     self.encoder_layer5,
        # )
        
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        # l1 = self.encoder_layer1(x)
        # l2 = self.encoder_layer2(l1)
        # l3 = self.encoder_layer3(l2)
        # l4 = self.encoder_layer4(l3)
        # l5 = self.encoder_layer5(l4)
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
        # x = self.encoder(x)
        return x

class HSI_FC(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(144, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, 128)
        self.fc5 = nn.Linear(128, 15)
        self.bn1 = nn.BatchNorm1d(16,momentum = 0.9)
        self.bn2 = nn.BatchNorm1d(32,momentum = 0.9)
        self.bn3 = nn.BatchNorm1d(64,momentum = 0.9)
        self.bn4 = nn.BatchNorm1d(128,momentum = 0.9)
        self.act = nn.ReLU(inplace = True)
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias,0)
            # elif isinstance(m,nn.BatchNorm1d):
            #     nn.init.constant_(m.weight,1)
            #     nn.init.constant_(m.bias,0)
            
    def forward(self, x):
        x1 = self.act(self.bn1(self.fc1(x)))
        x2 = self.act(self.bn2(self.fc2(x1)))
        x3 = self.act(self.bn3(self.fc3(x2)))
        x4 = self.act(self.bn4(self.fc4(x3)))
        x5 = self.act(self.fc5(x4))
        # l2_loss = torch.linalg.norm(self.fc1.weight)**2/2 + torch.linalg.norm(self.fc2.weight)**2/2 \
        #        + torch.linalg.norm(self.fc3.weight)**2/2 + torch.linalg.norm(self.fc4.weight)**2/2 + torch.linalg.norm(self.fc5.weight)**2/2
        return x5
    

# class LiDAR_CNNs(nn.Module):
#     def __init__(self, num_classes=15):
#         super(LiDAR_CNNs, self).__init__()
#         self.encoder_layer1 = nn.Sequential(
#             nn.Conv2d(21, 16, kernel_size=3, stride=1, padding='same'),
#             nn.BatchNorm2d(16),
#             nn.ReLU(inplace=True),
#         )
#         self.encoder_layer2 = nn.Sequential(
#             nn.Conv2d(16, 32, kernel_size=1, stride=1, padding='same'),
#             nn.BatchNorm2d(32),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.ReLU(inplace=True),
#         )
#         self.encoder_layer3 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same'),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#         )
#         self.encoder_layer4 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=1, stride=1, padding='same'),
#             nn.BatchNorm2d(128),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.ReLU(inplace=True),
#             nn.AvgPool2d(kernel_size=2, stride=2),
#         )
#         self.encoder_layer5 = nn.Sequential(
#             nn.Conv2d(128, num_classes, kernel_size=3, stride=1, padding='same'),
#             nn.Flatten(),
#         )
#         self.encoder = nn.ModuleList([
#             self.encoder_layer1, 
#             self.encoder_layer2, 
#             self.encoder_layer3, 
#             self.encoder_layer4,
#             self.encoder_layer5
#             ])
                
#     def forward(self, x):
#         for encoder_layer in self.encoder:
#             x = encoder_layer(x)
            
#         return x
