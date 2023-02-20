import os
import random
# import h5py
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib import cm
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import torch

def get_all_batterys():
    # load dataset from pkl files
    pklFile_1 = open('../batch1.pkl', 'rb')
    batch1 = pickle.load(pklFile_1)
    pklFile_2 = open('../batch2.pkl', 'rb')
    batch2 = pickle.load(pklFile_2)
    pklFile_3 = open('../batch3.pkl', 'rb')
    batch3 = pickle.load(pklFile_3)

    # clean error baterys
    b1_err = ['b1c0', 'b1c1', 'b1c2', 'b1c3', 'b1c4', 'b1c8', 'b1c10', 'b1c12', 'b1c13', 'b1c18', 'b1c22']
    b2_err = ['b2c1', 'b2c6', 'b2c9', 'b2c10', 'b2c21', 'b2c25']
    b3_err = ['b3c23', 'b3c32', 'b3c37']

    for bat in b1_err:
        if bat in batch1:
            batch1.pop(bat)

    for bat in b2_err:
        if bat in batch2:
            batch2.pop(bat)

    for bat in b3_err:
        if bat in batch3:
            batch3.pop(bat)
    
    # 將所有電池資料合併到一個 dict
    all_bat_dict = {}
    for batch in [batch1, batch2, batch3]:
        for bat, info in batch.items():
            all_bat_dict[bat] = info
    
    # save as pkl
    with open('Dataset/all_batterys.pkl', 'wb') as fp:
        pickle.dump(all_bat_dict, fp)


def preprocessing_for_DimReduction(filename='Dataset/all_batterys.pkl', cycle_idx=[0, 100]):
    """
    此資料前處理作為兩個特徵篩選器使用, 分別預測
    1. RUL in discharge half-cycle m
    2. Total charge time in discharge half-cycle m

    - 把每顆電池 1 到 100 循環內資料 V, Qd, I, T 線性插值成 500 個資料點後
      分別存成 (100, 4, 500) 的 npy file
    - 把每顆電池 1 到 100 循環 的 RUL 分別存成 (100, 1) 的 npy file
    - 把所有電池的 EOL 存成一個 (n_batterys, ) 的 npy file 
    - 把所有電池的 charge time 存成一個 (n_batterys, ) 的 npy file
    """
    path = 'Dataset/DimReduction_Each_battery/'
    dataset = pickle.load(open(filename, 'rb'))
    chargetime = []
    eol = []
    for bat in tqdm(dataset.keys()):
        curve, rul = [], [] # 充放電曲線, 電池剩餘壽命
        cl = dataset[bat]['cycle_life']
        chargetime.append(dataset[bat]['summary']['chargetime'][-1]) # 取最後一個循環的 chargetime
        eol.append(cl[0][0])
        n_cycles = len(dataset[bat]['cycles'].keys())
        for i in range(n_cycles):
            cycle = i+1
            if cl-cycle >= 1 and cycle >= 1: # 循環資料沒有紀錄到 cl
                # 取得循環內資料
                V = dataset[bat]['cycles'][str(i)]['V']
                Qd = dataset[bat]['cycles'][str(i)]['Qd']
                I = dataset[bat]['cycles'][str(i)]['I']
                T = dataset[bat]['cycles'][str(i)]['T']
                if len(V) < 1: # wrong data points number
                    print(bat)
                    continue # 遇到有錯誤的電池強制跳出本迴圈

                # 把資料線性插值到 500 個 data points
                interp_pt = np.linspace(0, len(V)-1, 500) # 設定要插值的點 (四種測量原資料長度相同)

                V_interp = np.interp(interp_pt, np.arange(len(V)), V).reshape(1, -1) # (1, 500)
                Qd_interp = np.interp(interp_pt, np.arange(len(Qd)), Qd).reshape(1, -1)
                I_interp = np.interp(interp_pt, np.arange(len(I)), I).reshape(1, -1)
                T_interp = np.interp(interp_pt, np.arange(len(T)), T).reshape(1, -1)
                
                # 去掉沒把charge資料清乾淨的discharge curve
                # if np.max(I_interp) > 0.5:
                #     print(bat)
                #     continue
                
                # 合併四種測量
                VQIT = np.concatenate([V_interp, Qd_interp, I_interp, T_interp], axis=0) # (4, 500)
                curve.append(VQIT) # (4, 500)
                rul.append(cl-cycle) # (1, 1)
        
        # save data as npy file
        curve = np.array(curve) # (n_cycles, 4, 500)
        rul = np.array(rul).reshape(-1, 1) # (n_cycles, 1)
        np.save(path+'Curve/'+bat+'_Curve.npy', curve[cycle_idx[0]:cycle_idx[1]]) # 只取前 100 循環的資料
        np.save(path+'RUL/'+bat+'_RUL.npy', rul[cycle_idx[0]:cycle_idx[1]])
    np.save(path+'Target/'+"EOL.npy", np.array(eol))
    np.save(path+'Target/'+"chargetime.npy", np.array(chargetime))


def dataset_preprocessing_DimReduction(train_val_split=0.8, seed=0, folder='CNN/Dataset/'):
    """
    分成三組: EoL<600, 600<=EoL<=1200, EoL>1200
    training, validation 和 testing 要平均分配樣本
    high:mid:low = 46:67:8
    """
    path = 'Dataset/DimReduction_Each_battery/'
    filename_lst = sorted(os.listdir(path+'Curve')) # 列出資料夾內所有檔案
    chargetime_arr = np.load(path+'Target/chargetime.npy')
    eol_arr = np.load(path+'Target/EOL.npy')

    low_x, mid_x, high_x = [], [], []
    low_t, mid_t, high_t = [], [], []
    for i in range(len(filename_lst)):
        curve = np.load(path+'Curve/'+filename_lst[i])
        eol = np.full((len(curve), 1), eol_arr[i]) # 擴展為 (n_cycles, 1)
        chargetime = np.full((len(curve), 1), chargetime_arr[i]) # 擴展為 (n_cycles, 1)
        target = np.concatenate((eol, chargetime), axis=1)

        # 依照壽命長短將電池分成三類
        if eol_arr[i] < 600:
            low_x.append(curve)
            low_t.append(target)
        elif eol_arr[i] > 1200:
            high_x.append(curve)
            high_t.append(target)
        else:
            mid_x.append(curve)
            mid_t.append(target)
    
    # 將三個分類的電池分別打亂
    low_split, mid_split, high_split = int(len(low_x)*train_val_split),\
                                       int(len(mid_x)*train_val_split),\
                                       int(len(high_x)*train_val_split) # 計算 training set split index
    np.random.seed(seed)
    low, mid, high = list(zip(low_x, low_t)), list(zip(mid_x, mid_t)), list(zip(high_x, high_t))
    np.random.shuffle(low), np.random.shuffle(mid), np.random.shuffle(high)
    low_x, low_t = zip(*low) # 還原 zip 的資料, 加 * 可以把列表內的資料拆開, 傳入函式中
    mid_x, mid_t = zip(*mid)
    high_x, high_t = zip(*high)
    low_x, low_t = np.array(low_x), np.array(low_t) # (num_battery, num_cycle, num_feature, num_point),
                                                    # (num_battery, num_cycle, num_target)
    mid_x, mid_t = np.array(mid_x), np.array(mid_t)
    high_x, high_t = np.array(high_x), np.array(high_t)

    trn_input = np.concatenate(((low_x[:low_split]), mid_x[:mid_split], high_x[:high_split]), axis=0)
    val_input = np.concatenate(((low_x[low_split:]), mid_x[mid_split:], high_x[high_split:]), axis=0)
    trn_target = np.concatenate(((low_t[:low_split]), mid_t[:mid_split], high_t[:high_split]), axis=0)
    val_target = np.concatenate(((low_t[low_split:]), mid_t[mid_split:], high_t[high_split:]), axis=0)
    
    # save as npy
    np.save(folder+'DimRdn_TrnSet/Trn_input.npy', trn_input)
    np.save(folder+'DimRdn_TrnSet/Trn_target.npy', trn_target)
    np.save(folder+'DimRdn_ValSet/Val_input.npy', val_input)
    np.save(folder+'DimRdn_ValSet/Val_target.npy', val_target)


def dataset_preprocessing_Predictor1(length=100):
    """
    取得每顆電池的循環間資料 (summary) 並存成
    (8, 100) 的特徵: QC, QD, TAVG, TMIN, TMAX, chargetime, eol_feature, chargetime_feature
    (2,) 的預測目標: eol, eol_chargetime
    其中 eol_feature, chargetime_feature 是分別來自 Dimension reduction 1, Dimension reduction 2 的預測結果
    """
    pklfile = 'Dataset/all_batterys.pkl'
    savepath = 'Dataset/DischargeDNN_each_battery/'
    dataset = pickle.load(open(pklfile, 'rb'))
    feature_selector = ['Model/eol_model_new_2.pth', 'Model/ct_model_new_3.pth']
    scaler_X, _ = get_scaler()
    for bat in dataset.keys():
        # 循環間特徵 (取前100循環)
        qc = dataset[bat]['summary']['QC'][1:length+1].reshape(1, -1)
        qd = dataset[bat]['summary']['QD'][1:length+1].reshape(1, -1)
        tavg = dataset[bat]['summary']['Tavg'][1:length+1].reshape(1, -1)
        tmin = dataset[bat]['summary']['Tmin'][1:length+1].reshape(1, -1)
        tmax = dataset[bat]['summary']['Tmax'][1:length+1].reshape(1, -1)
        chargetime = dataset[bat]['summary']['chargetime'][1:length+1].reshape(1, -1) # 該循環充電時間
        feature = np.concatenate([qc, qd, tavg, tmin, tmax, chargetime], axis=0) # (6, 100)

        # 取得 targets (EOL, EOL_chargetime)
        eol = dataset[bat]['cycle_life'][0, 0]
        eol_chargetime = dataset[bat]['summary']['chargetime'][-1]
        target = np.array([eol, eol_chargetime])

        # 取得 feature selector 特徵
        n_cycles = len(dataset[bat]['cycles'].keys())
        curve = []
        for i in range(n_cycles):
            cycle = i+1
            if eol-cycle >= 1 and cycle > 1:
                # 取得循環內資料
                V = dataset[bat]['cycles'][str(i)]['V']
                Qd = dataset[bat]['cycles'][str(i)]['Qd']
                I = dataset[bat]['cycles'][str(i)]['I']
                T = dataset[bat]['cycles'][str(i)]['T']

                # 把資料線性插值到 500 個 data points
                interp_pt = np.linspace(0, len(V)-1, 500) # 設定要插值的點 (四種測量原資料長度相同)
                V_interp = np.interp(interp_pt, np.arange(len(V)), V).reshape(1, -1) # (1, 500)
                Qd_interp = np.interp(interp_pt, np.arange(len(Qd)), Qd).reshape(1, -1)
                I_interp = np.interp(interp_pt, np.arange(len(I)), I).reshape(1, -1)
                T_interp = np.interp(interp_pt, np.arange(len(T)), T).reshape(1, -1)
                
                fs_data = np.concatenate([V_interp, Qd_interp, I_interp, T_interp], axis=0)
                curve.append(fs_data) # (4, 500)
                
                if len(curve) == length:
                    break
        
        curve = np.stack(curve, axis=0) # (100, 4, 500)
        
        # load feature selector
        if torch.cuda.is_available():
            device = torch.device('cuda')
        dimreduct1 = torch.load(feature_selector[0]).to(device)
        dimreduct2 = torch.load(feature_selector[1]).to(device)
        dimreduct1.eval()
        dimreduct2.eval()

        # normalize curve
        x_orgn_shape = curve.shape
        curve = scaler_X.transform(curve.reshape(-1, 4)).reshape(x_orgn_shape)

        # predict
        curve = torch.from_numpy(curve).float().to(device)
        with torch.no_grad():
            pred_EOL = dimreduct1(curve).detach().cpu().numpy()
            pred_EOL_chargetime = dimreduct2(curve).detach().cpu().numpy()
        feature = np.concatenate([feature, pred_EOL.reshape(1, -1), pred_EOL_chargetime.reshape(1, -1)], axis=0) # (8, 100)
        
        # save
        np.save(savepath+bat+'_predictor1_feature', feature)
        np.save(savepath+bat+'_predictor1_target', target)


def get_scaler(train=True, pred_target='EOL'):
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

    scaler_x, scaler_y = StandardScaler(), StandardScaler()

    # 使用 fit 方法 shape 需為 (n_samples, n_features) 
    scaler_x.fit(feature.reshape(-1, 4)) # (95*100*500, 4)
    scaler_y.fit(target.reshape(-1, 1)) # (95*100, 1)

    return scaler_x, scaler_y


if __name__ == '__main__':
    # get_all_batterys()
    # f = open('Dataset/all_batterys.pkl', 'rb')
    # data = pickle.load(f)
    # print(data.keys()) # 120

    # preprocessing_for_DimReduction(filename='Dataset/all_batterys.pkl', cycle_idx=[0, 100])
    # dataset_preprocessing_DimReduction(train_val_split=0.8, seed=0, folder='Dataset/')
    
    # dataset_preprocessing_Predictor1()