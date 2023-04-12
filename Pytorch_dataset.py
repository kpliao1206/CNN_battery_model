import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
# import scipy.stats as stats
# from torch.nn.functional import normalize

class DimReduction_dataset(Dataset):
    def __init__(self, train=True, pred_target='EOL', norm=True, set_code=''):
        """
        pred_target可更改為 EOL 或 chargetime
        """
        self.train = train
        self.pred_target = pred_target
        self.input, self.target = load_dataset(train=train, norm=norm, pred_target=pred_target, set_code=set_code)
        # (95*100, 4, 500), (95*100, 1) 
    
    def __getitem__(self, index):
        return self.input[index].type(torch.FloatTensor), self.target[index].type(torch.FloatTensor)

    def __len__(self):
        return self.input.shape[0]
    
    def plot(self, index=0, feature_id=0):
        feature_lst = ['Voltage', 'Discharge capacity', 'Current', 'Temperature']
        curve = self.input[index, feature_id, :]
        curve = np.squeeze(curve) # 消除多餘維度
        plt.plot(np.arange(curve.shape[0]), curve)
        plt.ylabel(feature_lst[feature_id])
        plt.xlabel('time')
        plt.show()
    
    def visualize(self, index):
        feature_lst =  ['Voltage', 'Discharge capacity', 'Current', 'Temperature']
        curve = self.input.numpy()[index]
        plt.figure(figsize=(12, 4))
        for i in range(4):
            plt.subplot(1, 4, i+1)
            plt.plot(np.arange(500)+1, curve[i, :], 'b', markersize=2.5)
            plt.ylabel(feature_lst[i])
            plt.xlabel('time series')
            plt.tight_layout()
        plt.show()
        plt.close()
    
    def check_shape(self):
        print(self.input.shape, self.target.shape)


class Predictor1_dataset(Dataset):
    def __init__(self, train=True, norm=True, padding=True, set_code=''):
        self.train = train
        self.norm = norm
        if train:
            folder = 'Severson_Dataset/DischargeDNN_TrnSet/'
            self.input, self.target = np.load(folder+'Feature_trn_'+set_code+'.npy'), np.load(folder+'Target_trn_'+set_code+'.npy')
        else:
            folder = 'Severson_Dataset/DischargeDNN_ValSet/'
            self.input, self.target = np.load(folder+'Feature_val_'+set_code+'.npy'), np.load(folder+'Target_val_'+set_code+'.npy')

        # normalize input data
        if norm:
            self.input = normalize_feature(self.input, 8).numpy()

        # last padding (augment data)
        if padding:
            aug_input = []
            cycle_length = self.input.shape[-1]
            for i in range(self.input.shape[0]):
                for cycle in range(cycle_length):
                    after_padding = self.input[i].copy()
                    after_padding[:, cycle+1:] = np.repeat(after_padding[:, cycle].reshape(-1, 1), cycle_length-cycle-1, axis=1)
                    aug_input.append(after_padding)
            self.input = np.stack(aug_input, axis=0) # (95*cycle_length, 8, 100)
            self.target = np.repeat(self.target, cycle_length, axis=0)

        self.input = torch.from_numpy(self.input)
        self.target = torch.from_numpy(self.target)
    
    def __getitem__(self, index):
        return self.input[index].type(torch.FloatTensor), self.target[index].type(torch.FloatTensor)
    
    def __len__(self):
        return self.input.shape[0]
    
    def visualize(self, index):
        feature_lst =  ['charge capacity', 'discharge capacity', 'chargetime', 'TAvg', 'TMin', 'TMax', 'EOL feature', 'chargetime feature']
        curve = self.input.numpy()[index]
        plt.figure(figsize=(14, 7))
        for i in range(8):
            # mean, std = np.mean(curve[i, :]), np.std(curve[i, :])
            # print(f'{feature_lst[i]}: mean={mean:.3f}, std={std:.3f}')
            # min, max = np.min(curve[i, :]), np.max(curve[i, :])
            # print(f'{feature_lst[i]}: min={min:.3f}, max={max:.3f}')
            plt.subplot(2, 4, i+1)
            plt.plot(np.arange(100)+1, curve[i, :], 'b', markersize=2.5)
            plt.ylabel(feature_lst[i])
            plt.xlabel('cycle')
            plt.tight_layout()
        # if self.train:
        #     folder = 'Feature_visualize/Train/'
        # else:
        #     folder = 'Feature_visualize/Valid/'
        # if self.norm:
        #     plt.savefig(folder+'predictor1_norm_'+str(index)+'.png')
        # else:
        #     plt.savefig(folder+'predictor1_'+str(index)+'.png')
        plt.show()
        plt.close()
    
    def check_shape(self):
        print(self.input.shape, self.target.shape)



def load_dataset(train=True, norm=True, pred_target='EOL', set_code=''):
    if train:
        feature, target = np.load('Severson_Dataset/feature_selector_discharge/trn_features.npy'), np.load('Severson_Dataset/feature_selector_discharge/trn_targets.npy')
    else:
        feature, target = np.load('Severson_Dataset/feature_selector_discharge/val_features.npy'), np.load('Severson_Dataset/feature_selector_discharge/val_targets.npy')
    print(feature.shape)
    feature = feature.reshape(-1, 4, 500) # (95*100, 4, 500)
    target = target.reshape(-1, 2)

    # seperate two target
    if pred_target == 'EOL':
        target = target[..., 0] # (95*100, 1) 
    elif pred_target == 'chargetime':
        target = target[..., 1] # (95*100, 1) 

    # normalization
    if norm:
        return normalize_feature(feature, 4), torch.from_numpy(target)
    else:
        return torch.from_numpy(feature), torch.from_numpy(target)


def normalize(feature, target):
    x_orgn_shape = feature.shape
    y_orgn_shape = target.shape

    scaler_x, scaler_y = StandardScaler(), StandardScaler()

    # 使用 fit 方法 shape 需為 (n_samples, n_features) 
    scaler_x.fit(feature.reshape(-1, 4)) # (95*100*500, 4)
    scaler_y.fit(target.reshape(-1, 1)) # (95*100, 1) 

    # transform
    data_x = scaler_x.transform(feature.reshape(-1, 4)).reshape(x_orgn_shape)
    data_y = scaler_y.transform(target.reshape(-1, 1)).reshape(y_orgn_shape)
    
    return torch.from_numpy(data_x), torch.from_numpy(data_y)


def normalize_feature(feature, num_feature):
    x_orgn_shape = feature.shape
    for i in range(num_feature):
        scaler_x = RobustScaler()
        # 使用 fit 方法 shape 需為 (n_samples, n_features) 
        x = feature[:, i, :].copy()
        x_orgn_shape = x.shape
        scaler_x.fit(x.reshape(-1, 1))
        for j in range(x.shape[0]):
            norm_curve = scaler_x.transform(x[j, :].reshape(-1, 1))
            feature[j, i, :] = norm_curve.reshape(-1)
        # feature_i = scaler_x.transform(x.reshape(-1, 1))
        # feature[:, i, :] = feature_i.reshape(x_orgn_shape[0], x_orgn_shape[-1])
    # for i in range(num_feature):
    #     print(np.mean(feature[:, i, :].reshape(-1)), np.std(feature[:, i, :].reshape(-1)))

    return torch.from_numpy(feature)


if __name__ == "__main__":
    # dimreduct = DimReduction_dataset(train=True, pred_target='EOL', norm=True)
    # dimreduct_nonorm = DimReduction_dataset(train=True, pred_target='EOL', norm=False)
    # dimreduct.visualize(3)
    # dimreduct_nonorm.visualize(3)
    # dimreduct.check_shape()
    # dimreduct_nonorm.check_shape()

    predictor1 = Predictor1_dataset(train=True, norm=True, padding=True)
    predictor1_nonorm = Predictor1_dataset(train=True, norm=False, padding=False)
    predictor1.visualize(100)
    predictor1.visualize(149)
    predictor1.visualize(199)

