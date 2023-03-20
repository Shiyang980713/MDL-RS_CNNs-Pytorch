import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModal(nn.Module):
    def __init__(self, encoder, fusion, decoder, num_classes=15):
        super(MultiModal, self).__init__()
        self.encoder = encoder
        self.fusion = fusion
        self.decoder = decoder
        
    def _init_weights(self):
        # initialize bias as 0, 
        # initialize weights as normal distribution
        for param in self.parameters():
            if len(param.shape) == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.normal_(param, 0, 0.01)

    def forward(self, x):
        """x is a dict, for multi modal data, x has multiple keys,
        for example, x = {'hsi_data': hsi_data, 'lidar_data': lidar_data}
        
        """
        assert len(x.keys()) > 0, "No data in x"
        if len(x.keys()) == 1:
            ...
        elif len(x.keys()) > 1:
            ...
        feature = self.encoder(x)
        fusioned = self.fusion(feature)
        out = self.decoder(fusioned)
        
        return x
