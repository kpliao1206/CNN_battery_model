import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler


def shape_after_filter(IS, FS, padding, stride):
    OS = (IS - FS + 2*padding) / stride + 1
    return OS

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss()
        eps = 1e-6
        loss = torch.sqrt(criterion(x, y) + eps)
        return loss

def divide_no_nan(a, b):
    """
    Auxiliary funtion to handle divide by 0
    """
    div = a / b
    div[div != div] = 0.0
    div[div == float('inf')] = 0.0
    return div

def MAPELoss(y, y_hat, mask=None):
    """MAPE Loss
    Calculates Mean Absolute Percentage Error between
    y and y_hat. MAPE measures the relative prediction
    accuracy of a forecasting method by calculating the
    percentual deviation of the prediction and the true
    value at a given time and averages these devations
    over the length of the series.
    Parameters
    ----------
    y: tensor (batch_size, output_size)
        actual values in torch tensor.
    y_hat: tensor (batch_size, output_size)
        predicted values in torch tensor.
    mask: tensor (batch_size, output_size)
        specifies date stamps per serie
        to consider in loss
    Returns
    -------
    mape:
    Mean absolute percentage error.
    """
    mask = divide_no_nan(mask, torch.abs(y))
    mape = torch.abs(y - y_hat) * mask
    mape = torch.mean(mape)
    return mape


def pred_inverse_transform(model_outputs, train=True, pred_target='EOL'):
    orgn_shape = model_outputs.shape
    model_outputs = model_outputs.numpy()
    
    if train:
        target = np.load('Dataset\DimRdn_TrnSet\Trn_target.npy')
    else:
        target = np.load('Dataset\DimRdn_ValSet\Val_target.npy')
    
    target = target.reshape(-1, 2)

    # seperate two target
    if pred_target == 'EOL':
        target = target[..., 0] # (95*100, 1) 
    elif pred_target == 'chargetime':
        target = target[..., 1] # (95*100, 1)

    scaler_y = StandardScaler()
    scaler_y.fit(target.reshape(-1, 1)) # (95*100, 1) 
    real_pred_value = scaler_y.inverse_transform(model_outputs.reshape(-1, 1))

    return torch.from_numpy(real_pred_value).view(orgn_shape)