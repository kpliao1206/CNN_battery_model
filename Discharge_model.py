import torch
import torch.nn as nn
from itertools import repeat


def conv_block(in_ch, out_ch, kernel_size, padding, activation=True):
    if activation:
        return nn.Sequential(nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
                             nn.BatchNorm1d(num_features=out_ch),
                             nn.Mish()) # Mish activation function
    else:
        return nn.Sequential(nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),)


class SpatialDropout(nn.Module):
    """
    spatial dropout是針對channel位置做dropout
    ex.若對(batch, timesteps, embedding)的輸入沿着axis=1執行
    可對embedding的數個channel整體dropout
    沿着axis=2則是對某些timestep整體dropout
    """
    def __init__(self, drop=0.2):
        super(SpatialDropout, self).__init__()
        self.drop = drop # 需要drop的inputs比例
        
    def forward(self, inputs, noise_shape=None):
        """
        @param: inputs, tensor
        @param: noise_shape, tuple, 應和inputs的shape一致, 其中值為1的即沿著drop的axis
        """
        outputs = inputs.clone()
        # 設定 noise_shape
        if noise_shape is None:
            noise_shape = (inputs.shape[0], *repeat(1, inputs.dim()-2), inputs.shape[-1])   # 默認對中间所有的shape 進行 dropout
                                            # repeat: 產生 inputs 維度-2 個 '1', * 會將其解包
        self.noise_shape = noise_shape
        # 只在 training 時 dropout
        if not self.training or self.drop == 0: 
            return inputs
            # self.training 繼承自 nn.Module, 
            # Boolean represents whether this module is in training or evaluation mode.
        else:
            noises = self._make_noises(inputs)
            if self.drop == 1:
                noises.fill_(0.0)
            else:
                noises.bernoulli_(1 - self.drop).div_(1 - self.drop)
                # Fills each location of noises with an independent sample from Bernoulli(p), 值為1或0
            noises = noises.expand_as(inputs)    
            outputs.mul_(noises)
            return outputs
            
    def _make_noises(self, inputs):
        return inputs.new().resize_(self.noise_shape) # new(): 建立一個有相同 type, device 的空tensor, resize後元素全為0


class DimReduction_1(nn.Module):
    def __init__(self, in_ch=4, out_ch=1):
        super(DimReduction_1, self).__init__()
        self.batchnorm = nn.BatchNorm1d(num_features=4)
        self.conv1_1 = conv_block(in_ch, 32, kernel_size=5, padding=2) # kernal (5x4x32), 500->500
        self.conv1_2 = conv_block(32, 32, kernel_size=5, padding=2) # (5x32x32) 500->500
        self.conv1_3 = conv_block(32, 32, kernel_size=5, padding=2) # (5x32x32) 500->500
        self.conv1_4 = conv_block(32, 32, kernel_size=5, padding=2) # (5x32x32) 500->500
        self.maxpool1_1 = nn.MaxPool1d(2, 2) # (kernal_size=2, stride=2) 500->250
        self.maxpool1_2 = nn.MaxPool1d(2, 2) # 500->250
        self.maxpool1_3 = nn.MaxPool1d(2, 2) # 250->125
        self.spatial_drop1 = SpatialDropout(drop=0.2)
        self.conv2_1 = conv_block(64, 32, kernel_size=11, padding=5) # 125->125
        self.conv2_2 = conv_block(32, 32, kernel_size=11, padding=5) # 125->125
        self.conv2_3 = conv_block(32, 64, kernel_size=7, padding=3) # 125->125
        self.conv2_4 = conv_block(64, 64, kernel_size=7, padding=3) # 125->125
        self.maxpool2_1 = nn.MaxPool1d(2, 2) # 125->62
        self.maxpool2_2 = nn.MaxPool1d(2, 2) # 125->62
        self.avgpool1 = nn.AvgPool1d(2, 2) # 62->31
        self.conv3_1 = conv_block(96, 256, kernel_size=7, padding=3) # 31->31
        self.conv3_2 = conv_block(256, 256, kernel_size=7, padding=3) # 31->31
        self.conv3_3 = conv_block(256, 256, kernel_size=5, padding=2) # 31->31
        self.conv3_4 = conv_block(256, 256, kernel_size=5, padding=2) # 31->31
        self.spatial_drop2 = SpatialDropout(drop=0.2)
        self.glbavgpool = nn.AvgPool1d(31, 1) # 31->1
        self.glbmaxpool = nn.MaxPool1d(31, 1) # 31->1
        self.linear = nn.Linear(512, out_ch)

        self.initialize_weights()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        input dims: (channel_size=4, data_size=500)
        output dims: (channel_size=1, )
        """
        input = self.batchnorm(x)

        output1 = self.conv1_1(input)
        output1 = self.conv1_2(output1)
        output2 = self.conv1_3(output1)
        output2 = self.conv1_4(output2)
        output1 = self.maxpool1_1(output1)
        output2 = self.maxpool1_2(output2)
        output3 = self.maxpool1_3(torch.cat((output1, output2), dim=1))
        output3 = self.spatial_drop1(output3)

        output3 = self.conv2_1(output3)
        output3 = self.conv2_2(output3)
        output4 = self.conv2_3(output3)
        output4 = self.conv2_4(output4)
        output3 = self.maxpool2_1(output3)
        output4 = self.maxpool2_2(output4)
        output5 = self.avgpool1(torch.cat((output3, output4), dim=1))

        output5 = self.conv3_1(output5)
        output5 = self.conv3_2(output5)
        output6 = self.conv3_3(output5)
        output6 = self.conv3_4(output6)
        output7 = self.spatial_drop2(torch.cat((output5, output6), dim=1))
        output8 = self.glbavgpool(output7)
        output9 = self.glbmaxpool(output7)
        output10 = torch.add(output8, output9)
        output = self.linear(torch.squeeze(output10))

        return output


class DimReduction_2(nn.Module):
    def __init__(self, in_ch=4, out_ch=1):
        super(DimReduction_2, self).__init__()
        self.batchnorm = nn.BatchNorm1d(num_features=4)
        self.conv1_1 = conv_block(in_ch, 128, kernel_size=11, padding=5) # 500->500
        self.conv2_1 = conv_block(128, 128, kernel_size=11, padding=5) # 500->500
        self.conv1_3 = conv_block(128, 128, kernel_size=5, padding=2) # 500->500
        self.conv1_4 = conv_block(128, 128, kernel_size=5, padding=2) # 500->500
        self.avgpool1_1 = nn.AvgPool1d(2, 2) # 500->250
        self.avgpool1_2 = nn.AvgPool1d(2, 2) # 500->250
        self.spatial_drop1 = SpatialDropout(drop=0.2)
        self.conv2_1 = conv_block(256, 128, kernel_size=3, padding=1) # 250->250
        self.conv2_2 = conv_block(128, 128, kernel_size=3, padding=1) # 250->250
        self.conv2_3 = conv_block(128, 128, kernel_size=5, padding=2) # 250->250
        self.conv2_4 = conv_block(128, 128, kernel_size=5, padding=2) # 250->250
        self.maxpool2_1 = nn.MaxPool1d(2, 2) # 250->125
        self.maxpool2_2 = nn.MaxPool1d(2, 2) # 250->125
        self.avgpool2_1 = nn.AvgPool1d(2, 2) # 125->62
        self.glbavgpool = nn.AvgPool1d(62, 1) # 62->1
        self.glbmaxpool = nn.MaxPool1d(62, 1) # 62->1
        self.linear = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, out_ch))

        self.initialize_weights()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input = self.batchnorm(x)

        output1 = self.conv1_1(input)
        output1 = self.conv1_2(output1)
        output2 = self.conv1_3(output1)
        output2 = self.conv1_4(output2)
        output1 = self.avgpool1_1(output1)
        output2 = self.avgpool1_2(output2)
        output3 = self.spatial_drop1(torch.cat((output1, output2), dim=1))

        output3 = self.conv2_1(output3)
        output3 = self.conv2_2(output3)
        output4 = self.conv2_3(output3)
        output4 = self.conv2_4(output4)
        output5 = self.avgpool2_1(torch.cat((output3, output4), dim=1))
        output5 = torch.add(self.glbavgpool(output5), self.glbmaxpool(output5))
        output = self.linear(torch.squeeze(output5))

        return output

        



