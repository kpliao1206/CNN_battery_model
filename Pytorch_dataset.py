import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
# import scipy.stats as stats
# from torch.nn.functional import normalize

class DimReduction_dataset(Dataset):
    def __init__(self, train=True, pred_target='EOL', norm=True):
        """
        pred_target可更改為 EOL 或 chargetime
        """
        self.train = train
        self.pred_target = pred_target
        self.input, self.target = load_dataset(train=train, norm=norm, pred_target=pred_target)
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


def load_dataset(train=True, norm=True, pred_target='EOL'):
    if train:
        feature, target = np.load('Dataset\DimRdn_TrnSet\Trn_input.npy'), np.load('Dataset\DimRdn_TrnSet\Trn_target.npy')
    else:
        feature, target = np.load('Dataset\DimRdn_ValSet\Val_input.npy'), np.load('Dataset\DimRdn_ValSet\Val_target.npy')
    
    feature = feature.reshape(-1, 4, 500) # (95*100, 4, 500)
    target = target.reshape(-1, 2)

    # seperate two target
    if pred_target == 'EOL':
        target = target[..., 0] # (95*100, 1) 
    elif pred_target == 'chargetime':
        target = target[..., 1] # (95*100, 1) 

    # normalization
    if norm:
        return normalize(feature, target)
    else:
        return torch.from_numpy(feature), torch.from_numpy(target)


# def normalize(feature, target):
#     """
#     using z-score normalization
#     """
#     # np.seterr(invalid='ignore') # ignore error
#     # norm_x = (feature - np.mean(feature, axis=3, keepdims=True)) / np.std(feature, axis=3, keepdims=True)
#     # norm_y = (target - np.mean(target, axis=1, keepdims=True)) / np.std(target, axis=1, keepdims=True)
#     norm_x = stats.zscore(feature, axis=2, ddof=1e-6)
#     norm_y = stats.zscore(target, axis=0, ddof=1e-6)
#     return torch.from_numpy(norm_x), torch.from_numpy(norm_y)


# def get_scaler(feature, target):
#     scaler_x, scaler_y = StandardScaler(), StandardScaler()

#     # 使用 fit 方法 shape 需為 (n_samples, n_features) 
#     scaler_x.fit(feature.reshape(-1, 4)) # (95*100*500, 4)
#     scaler_y.fit(target) # (95*100, 1) 
    
#     return scaler_x, scaler_y

def normalize(feature, target):
    x_orgn_shape = feature.shape
    y_orgn_shape = target.shape

    scaler_x, scaler_y = StandardScaler(), StandardScaler()

    # 使用 fit 方法 shape 需為 (n_samples, n_features) 
    scaler_x.fit(feature.reshape(-1, 4)) # (95*100*500, 4)
    scaler_y.fit(target.reshape(-1, 1)) # (95*100, 1) 

    # trandform
    data_x = scaler_x.transform(feature.reshape(-1, 4)).reshape(x_orgn_shape)
    data_y = scaler_y.transform(target.reshape(-1, 1)).reshape(y_orgn_shape)
    
    return torch.from_numpy(data_x), torch.from_numpy(data_y)


if __name__ == "__main__":
    dataset = DimReduction_dataset(norm=True)
    dataset.plot(feature_id=1)
    dataloader = DataLoader(dataset=dataset, shuffle=False)
    example = iter(dataloader)
    feature, target = next(example)
    print(feature.shape, target.shape)

    



