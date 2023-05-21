import torch
import torch.nn as nn
from itertools import repeat
from torchsummary import summary


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
    def __init__(self, in_ch=4, out_ch=1, drop=0.2):
        super(DimReduction_1, self).__init__()
        self.batchnorm = nn.BatchNorm1d(num_features=4)
        self.conv1_1 = conv_block(in_ch, 128, kernel_size=13, padding=6) 
        self.conv1_2 = conv_block(128, 128, kernel_size=13, padding=6) 
        self.conv1_3 = conv_block(128, 384, kernel_size=5, padding=2) 
        self.conv1_4 = conv_block(384, 384, kernel_size=5, padding=2) 
        self.maxpool1_1 = nn.MaxPool1d(2, 2)
        self.maxpool1_2 = nn.MaxPool1d(2, 2) 
        self.maxpool1_3 = nn.MaxPool1d(2, 2) 
        self.spatial_drop1 = SpatialDropout(drop=drop)
        self.conv2_1 = conv_block(512, 64, kernel_size=17, padding=8) 
        self.conv2_2 = conv_block(64, 64, kernel_size=17, padding=8) 
        self.conv2_3 = conv_block(64, 192, kernel_size=3, padding=1) 
        self.conv2_4 = conv_block(192, 192, kernel_size=3, padding=1) 
        self.maxpool2_1 = nn.MaxPool1d(2, 2) 
        self.maxpool2_2 = nn.MaxPool1d(2, 2) 
        self.avgpool1 = nn.AvgPool1d(2, 2) 
        self.conv3_1 = conv_block(256, 32, kernel_size=13, padding=6) 
        self.conv3_2 = conv_block(32, 32, kernel_size=13, padding=6) 
        self.conv3_3 = conv_block(32, 32, kernel_size=3, padding=1) 
        self.conv3_4 = conv_block(32, 32, kernel_size=3, padding=1) 
        self.spatial_drop2 = SpatialDropout(drop=drop)
        self.glbavgpool = nn.AdaptiveAvgPool1d(1) 
        self.glbmaxpool = nn.AdaptiveMaxPool1d(1) 
        self.linear = nn.Linear(64, out_ch)

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
        output1 = self.conv1_1(x)
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
    def __init__(self, in_ch=4, out_ch=1, drop=0.2):
        super(DimReduction_2, self).__init__()
        self.batchnorm = nn.BatchNorm1d(num_features=4)
        self.conv1_1 = conv_block(in_ch, 128, kernel_size=13, padding=6) 
        self.conv1_2 = conv_block(128, 128, kernel_size=13, padding=6) 
        self.conv1_3 = conv_block(128, 256, kernel_size=5, padding=2) 
        self.conv1_4 = conv_block(256, 256, kernel_size=5, padding=2) 
        self.maxpool1_1 = nn.MaxPool1d(2, 2)
        self.maxpool1_2 = nn.MaxPool1d(2, 2) 
        self.maxpool1_3 = nn.MaxPool1d(2, 2) 
        self.spatial_drop1 = SpatialDropout(drop=drop)
        self.conv2_1 = conv_block(384, 32, kernel_size=17, padding=8) 
        self.conv2_2 = conv_block(32, 32, kernel_size=17, padding=8) 
        self.conv2_3 = conv_block(32, 64, kernel_size=3, padding=1) 
        self.conv2_4 = conv_block(64, 64, kernel_size=3, padding=1) 
        self.maxpool2_1 = nn.MaxPool1d(2, 2) 
        self.maxpool2_2 = nn.MaxPool1d(2, 2) 
        self.avgpool1 = nn.AvgPool1d(2, 2) 
        self.conv3_1 = conv_block(96, 32, kernel_size=3, padding=1) 
        self.conv3_2 = conv_block(32, 32, kernel_size=3, padding=1) 
        self.conv3_3 = conv_block(32, 96, kernel_size=9, padding=4) 
        self.conv3_4 = conv_block(96, 96, kernel_size=9, padding=4) 
        self.spatial_drop2 = SpatialDropout(drop=drop)
        self.glbavgpool = nn.AdaptiveAvgPool1d(1) 
        self.glbmaxpool = nn.AdaptiveMaxPool1d(1) 
        self.linear = nn.Linear(128, out_ch)

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
        output1 = self.conv1_1(x)
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


class DimReduction_3(nn.Module):
    def __init__(self, in_ch=4, out_ch=1, drop=0.2):
        super(DimReduction_3, self).__init__()
        self.batchnorm = nn.BatchNorm1d(num_features=4)
        self.conv1_1 = conv_block(in_ch, 128, kernel_size=17, padding=8) 
        self.conv1_2 = conv_block(128, 128, kernel_size=17, padding=8) 
        self.conv1_3 = conv_block(128, 384, kernel_size=9, padding=4) 
        self.conv1_4 = conv_block(384, 384, kernel_size=9, padding=4) 
        self.maxpool1_1 = nn.MaxPool1d(2, 2)
        self.maxpool1_2 = nn.MaxPool1d(2, 2) 
        self.maxpool1_3 = nn.MaxPool1d(2, 2) 
        self.spatial_drop1 = SpatialDropout(drop=drop)
        self.conv2_1 = conv_block(512, 64, kernel_size=17, padding=8) 
        self.conv2_2 = conv_block(64, 64, kernel_size=17, padding=8) 
        self.conv2_3 = conv_block(64, 192, kernel_size=3, padding=1) 
        self.conv2_4 = conv_block(192, 192, kernel_size=3, padding=1) 
        self.maxpool2_1 = nn.MaxPool1d(2, 2) 
        self.maxpool2_2 = nn.MaxPool1d(2, 2) 
        self.avgpool1 = nn.AvgPool1d(2, 2) 
        self.conv3_1 = conv_block(256, 64, kernel_size=13, padding=6) 
        self.conv3_2 = conv_block(64, 64, kernel_size=13, padding=6) 
        self.conv3_3 = conv_block(64, 64, kernel_size=3, padding=1) 
        self.conv3_4 = conv_block(64, 64, kernel_size=3, padding=1) 
        self.spatial_drop2 = SpatialDropout(drop=drop)
        self.glbavgpool = nn.AdaptiveAvgPool1d(1) 
        self.glbmaxpool = nn.AdaptiveMaxPool1d(1) 
        self.linear = nn.Linear(128, out_ch)

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
        output1 = self.conv1_1(x)
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


class DimReduction_4(nn.Module):
    def __init__(self, in_ch=4, out_ch=1, drop=0.2):
        super(DimReduction_4, self).__init__()
        self.batchnorm = nn.BatchNorm1d(num_features=4)
        self.conv1_1 = conv_block(in_ch, 256, kernel_size=15, padding=7) 
        self.conv1_2 = conv_block(256, 256, kernel_size=15, padding=7) 
        self.conv1_3 = conv_block(256, 512, kernel_size=3, padding=1) 
        self.conv1_4 = conv_block(512, 512, kernel_size=3, padding=1) 
        self.maxpool1_1 = nn.MaxPool1d(2, 2)
        self.maxpool1_2 = nn.MaxPool1d(2, 2) 
        self.maxpool1_3 = nn.MaxPool1d(2, 2) 
        self.spatial_drop1 = SpatialDropout(drop=drop)
        self.conv2_1 = conv_block(768, 32, kernel_size=3, padding=1) 
        self.conv2_2 = conv_block(32, 32, kernel_size=3, padding=1) 
        self.conv2_3 = conv_block(32, 64, kernel_size=3, padding=1) 
        self.conv2_4 = conv_block(64, 64, kernel_size=3, padding=1) 
        self.maxpool2_1 = nn.MaxPool1d(2, 2) 
        self.maxpool2_2 = nn.MaxPool1d(2, 2) 
        self.avgpool1 = nn.AvgPool1d(2, 2) 
        self.conv3_1 = conv_block(96, 128, kernel_size=17, padding=8) 
        self.conv3_2 = conv_block(128, 128, kernel_size=17, padding=8) 
        self.conv3_3 = conv_block(128, 256, kernel_size=7, padding=3) 
        self.conv3_4 = conv_block(256, 256, kernel_size=7, padding=3) 
        self.spatial_drop2 = SpatialDropout(drop=drop)
        self.glbavgpool = nn.AdaptiveAvgPool1d(1) 
        self.glbmaxpool = nn.AdaptiveMaxPool1d(1) 
        self.linear = nn.Linear(384, out_ch)

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
        output1 = self.conv1_1(x)
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


class Predictor_3(nn.Module):
    def __init__(self, in_ch=10, out_ch=1, drop=0.2):
        super(Predictor_3, self).__init__()
        self.conv1_1 = conv_block(in_ch, 256, kernel_size=5, padding=2)
        self.conv1_2 = conv_block(256, 128, kernel_size=9, padding=4)
        self.conv1_3 = conv_block(128, 512, kernel_size=3, padding=1)
        self.spatial_drop1_1 = SpatialDropout(drop)
        self.maxpool1_1 = nn.MaxPool1d(2, 2)

        self.conv2_1 = conv_block(512, 32, kernel_size=5, padding=2)
        self.conv2_2 = conv_block(512, 32, kernel_size=9, padding=4)
        self.glbavgpool2_1 = nn.AdaptiveAvgPool1d(1)
        self.glbavgpool2_2 = nn.AdaptiveAvgPool1d(1)

        self.conv3 = nn.Sequential(
            conv_block(1, 128, kernel_size=9, padding=0),
            conv_block(128, 256, kernel_size=15, padding=0),
            conv_block(256, 64, kernel_size=5, padding=0)
        )
        self.glbavgpool3_1 = nn.AdaptiveAvgPool1d(1)
        self.glbmaxpool3_1 = nn.AdaptiveMaxPool1d(1)
        self.linear3_1 = nn.Linear(64, out_ch)

        self.conv4 = nn.Sequential(
            conv_block(1, 32, kernel_size=7, padding=0),
            conv_block(32, 128, kernel_size=5, padding=0),
            conv_block(128, 128, kernel_size=9, padding=0)
        )
        self.glbavgpool4_1 = nn.AdaptiveAvgPool1d(1)
        self.glbmaxpool4_1 = nn.AdaptiveMaxPool1d(1)
        self.linear4_1 = nn.Linear(128, out_ch)

        self.conv5 = nn.Sequential(
            conv_block(1, 256, kernel_size=13, padding=0),
            conv_block(256, 256, kernel_size=7, padding=0),
            conv_block(256, 64, kernel_size=7, padding=0)
        )
        self.glbavgpool5_1 = nn.AdaptiveAvgPool1d(1)
        self.glbmaxpool5_1 = nn.AdaptiveMaxPool1d(1)
        self.linear5_1 = nn.Linear(64, out_ch)

        self.activation = nn.Mish()
    
    def forward(self, x):
        output1 = self.conv1_1(x)
        output1 = self.conv1_2(output1)
        output1 = self.conv1_3(output1)
        output1 = self.spatial_drop1_1(output1)
        output1 = self.maxpool1_1(output1)

        output2_1 = self.conv2_1(output1)
        output2_2 = self.conv2_2(output1)
        output2_2 = torch.matmul(torch.transpose(output2_1, 1, 2), output2_2)
        output2_2 = self.activation(output2_2)
        output2_2 = self.glbavgpool2_2(output2_2)
        output2_1 = self.glbavgpool2_1(output2_1)
        output2 = torch.cat((output2_1, output2_2), dim=1).squeeze().unsqueeze(1)

        output3 = self.conv3(output2)
        output3_1 = self.glbavgpool3_1(output3)
        output3_2 = self.glbmaxpool3_1(output3)
        output3 = torch.add(output3_1, output3_2).squeeze()
        output3 = self.linear3_1(output3)

        output4 = self.conv4(output2)
        output4_1 = self.glbavgpool4_1(output4)
        output4_2 = self.glbmaxpool4_1(output4)
        output4 = torch.add(output4_1, output4_2).squeeze()
        output4 = self.linear4_1(output4)

        output5 = self.conv5(output2)
        output5_1 = self.glbavgpool5_1(output5)
        output5_2 = self.glbmaxpool5_1(output5)
        output5 = torch.add(output5_1, output5_2).squeeze()
        output5 = self.linear5_1(output5)

        output_eol = output3 + output4 + output5
        return output_eol



if __name__ == '__main__':
    # model1 = DimReduction_1().cuda()
    # summary(model1, (4, 500))

    # model2 = DimReduction_2().cuda()
    # summary(model2, (4, 500))

    # model3 = DimReduction_3().cuda()
    # summary(model3, (4, 500))

    # model4 = DimReduction_4().cuda()
    # summary(model4, (4, 500))

    model5 = Predictor_3().cuda()
    summary(model5, (10, 100))