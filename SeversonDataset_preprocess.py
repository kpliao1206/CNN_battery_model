import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from Pytorch_dataset import DimReduction_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
# from data_preprocessing import Feature_Selector_Dataset
# import discharge_model
# import full_model


"""
於'Data-driven prediction of battery cycle life before capacity degradation'中使用的dataset
由124顆商用LFP電池(APR18650M1A)組成 以快充及4C放電循環至EoL
其額定電容量為1.11Ah 額定電壓為3.3V
資料被分為三個bath
"""
def mat_to_npy(save_path='Severson_Dataset/npdata_each_cell/', cycle_length=100):
    """
    將mat檔中需要的資料截取至npy檔
    包含各電池summary(所有cycle的Qc, Qd, Tmin, Tmax, Tavg, Chargetime)
    以及partial資訊(到cycle_length為止的各cycle Q, V, I ,T)
    """
    filename = ['../2017-05-12_batchdata_updated_struct_errorcorrect.mat',
                '../2017-06-30_batchdata_updated_struct_errorcorrect.mat',
                '../2018-04-12_batchdata_updated_struct_errorcorrect.mat']

    # 各batch中discharge部分有問題的電池 要加以清理
    b1_err = [0, 1, 2, 3, 4, 5, 8, 10, 12, 13, 18, 14, 15]
    b2_err = [1, 6, 9, 10, 21, 25, 12, 15, 44]
    b3_err = [23, 32, 37]
    err_list = [b1_err, b2_err, b3_err]
    batch_name = ['b1c', 'b2c', 'b3c']
    for b in range(len(filename)): # batch數
        f = h5py.File(filename[b], 'r')
        batch = f['batch']
        num_cells = batch['summary'].shape[0]
        for i in range(num_cells): # 該batch下的電池cell數量
            if i in err_list[b]:
                print('skip err cell: batch %d, cell_id %d'%(b+1, i))
                continue
            Cycle_life = f[batch['cycle_life'][i, 0]][()]
            Qc_summary = np.hstack(f[batch['summary'][i, 0]]['QCharge'][0, :].tolist())
            Qd_summary = np.hstack(f[batch['summary'][i, 0]]['QDischarge'][0, :].tolist())
            Chargetime = np.hstack(f[batch['summary'][i, 0]]['chargetime'][0, :].tolist())
            Tavg = np.hstack(f[batch['summary'][i, 0]]['Tavg'][0, :].tolist())
            Tmin = np.hstack(f[batch['summary'][i, 0]]['Tmin'][0, :].tolist())
            Tmax = np.hstack(f[batch['summary'][i, 0]]['Tmax'][0, :].tolist())
            key = batch_name[b] + str(i).zfill(2)
            # 儲存循環間資訊
            summary = np.vstack([Qc_summary, Qd_summary, Tmin, Tmax, Tavg, Chargetime]) # shape:(6, n_cycle)
            if b==0:
                np.save(save_path+key+'_summary', summary[:, 1:])
            else:
                np.save(save_path+key+'_summary', summary)

            cycles = f[batch['cycles'][i, 0]]
            cycle_info = []
            for j in range(1, cycle_length+1): # 選擇前n個cyle
                temper = np.hstack((f[cycles['T'][j, 0]]))
                current = np.hstack((f[cycles['I'][j, 0]]))
                voltage = np.hstack((f[cycles['V'][j, 0]]))
                Qc = np.hstack((f[cycles['Qc'][j, 0]]))
                Qd = np.hstack((f[cycles['Qd'][j, 0]]))
                Qdd = np.diff(np.diff(Qd)) # 放電容量二次微分
                ch_s = 0 # 充電開始
                ch_e = np.where(current==0)[0][1] # 充電結束, 電流歸零
                dis_s = np.where(np.diff(Qd)>=1e-3)[0][0] # 放電開始
                dis_e = np.where(Qdd>1e-4)[0][-1]+1 # 放電結束

                charge_info = linear_interpolation([Qc[ch_s:ch_e], voltage[ch_s:ch_e], current[ch_s:ch_e], temper[ch_s:ch_e]], points=500) 
                discharge_info = linear_interpolation([Qd[dis_s:dis_e], voltage[dis_s:dis_e], current[dis_s:dis_e], temper[dis_s:dis_e]], points=500) 
                cycle_info.append(np.expand_dims(np.hstack([charge_info, discharge_info]), axis=0)) # [0:500] charge, [500:] discharge
            np.save(save_path+key+'_cycle', np.concatenate(cycle_info, axis=0))
            print(key+' finished')


def linear_interpolation(seq, points=500):
    interp_list = []
    for s in seq:
        interp_id = np.linspace(0, len(s)-1, points)
        interp_list.append(np.interp(interp_id, np.arange(len(s)), s).reshape(1, -1))       
    return np.vstack(interp_list)


def data_visualization(f_id):
    feature_list = ['charge capacity', 'discharge capacity', 'chargetime', 'TAvg', 'TMin', 'TMax']
    path = 'Severson_Dataset/npdata_each_cell/'
    cmap = plt.get_cmap('coolwarm_r')
    filename = sorted(os.listdir(path))
    print('n cells: %d' % (len(filename)//2))
    eols = []
    for i in range((len(filename))//2):
        summary = np.load(path+filename[2*i+1])
        eol = summary.shape[1]
        eols.append(eol)
        plt.plot(np.arange(eol), summary[f_id, :], c=cmap((eol-200)/1800), alpha=0.7)
    print("min EOL: %d, max EOL: %d"%(min(eols), max(eols)))
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=200, vmax=2000),cmap='coolwarm_r')
    sm.set_array([])
    plt.colorbar(sm)
    plt.ylabel(feature_list[f_id])
    plt.xlabel('cycle')
    plt.show()
    plt.close()

    
def train_val_split(train_ratio=0.8, seed=15, cycle_used=100, save_path='Severson_Dataset/feature_selector_discharge/', set_code=''):
    load_path = 'Severson_Dataset/npdata_each_cell/'
    filename = sorted(os.listdir(load_path))
    features, targets, all_summary = [], [], []
    for i in range((len(filename))//2):
        curve = np.load(load_path+filename[2*i])
        summary = np.load(load_path+filename[2*i+1])
        eol, chargetime_end = len(summary[0]), summary[5, -1]
        features.append(curve)
        targets.append(np.array([eol, chargetime_end]))
        all_summary.append(np.expand_dims(summary[:, :100], axis=0))

    # 根據seed設定隨機調整順序
    dataset = list(zip(features, targets, all_summary))
    np.random.seed(seed)
    np.random.shuffle(dataset) 
    features = [f[0] for f in dataset]
    targets = [f[1] for f in dataset]
    all_summary = [f[2] for f in dataset]
    split_point = int(len(targets)*train_ratio)
    print(np.concatenate(features[:split_point]).shape)
    print(np.repeat(np.vstack(targets[:split_point]), 100, axis=0).shape)
    print(np.concatenate(all_summary[:split_point]).shape)
    np.save(save_path+'trn_features_'+set_code, np.concatenate(features[:split_point]))
    np.save(save_path+'val_features_'+set_code, np.concatenate(features[split_point:]))
    np.save(save_path+'trn_targets_'+set_code, np.repeat(np.vstack(targets[:split_point]), 100, axis=0))
    np.save(save_path+'val_targets_'+set_code, np.repeat(np.vstack(targets[split_point:]), 100, axis=0))
    np.save(save_path+'trn_summary_'+set_code, np.concatenate(all_summary[:split_point]))
    np.save(save_path+'val_summary_'+set_code, np.concatenate(all_summary[split_point:]))


def train_val_split_by_batch(train_ratio=0.8, seed=0, save_path_dc='Severson_Dataset/feature_selector_discharge/', save_path_c = 'Severson_Dataset/feature_selector_charge/', set_code=''):
    load_path = 'Severson_Dataset/npdata_each_cell/'
    filename = sorted(os.listdir(load_path))
    b1_f, b1_t, b1_s = [], [], []
    b2_f, b2_t, b2_s = [], [], []
    b3_f, b3_t, b3_s = [], [], []
    for file in filename:
        if ('b1c' in file):
            if ('cycle' in file):
                curve = np.load(load_path+file)
                b1_f.append(curve)
            if ('summary' in file):
                summary = np.load(load_path+file)
                eol, ct = len(summary[0]), summary[5, -1]
                b1_t.append(np.array([eol, ct]))
                b1_s.append(np.expand_dims(summary[:, :100], axis=0))
        
        elif ('b2c' in file):
            if ('cycle' in file):
                curve = np.load(load_path+file)
                b2_f.append(curve)
            if ('summary' in file):
                summary = np.load(load_path+file)
                eol, ct = len(summary[0]), summary[5, -1]
                b2_t.append(np.array([eol, ct]))
                b2_s.append(np.expand_dims(summary[:, :100], axis=0))

        elif ('b3c' in file):
            if ('cycle' in file):
                curve = np.load(load_path+file)
                b3_f.append(curve)
            if ('summary' in file):
                summary = np.load(load_path+file)
                eol, ct = len(summary[0]), summary[5, -1]
                b3_t.append(np.array([eol, ct]))
                b3_s.append(np.expand_dims(summary[:, :100], axis=0))
    print('Number of cells in b1: '+str(len(b1_f)))
    print('Number of cells in b2: '+str(len(b2_f)))
    print('Number of cells in b3: '+str(len(b3_f)))

    b1_split = train_test_split(b1_f, b1_t, b1_s, train_size=train_ratio, random_state=seed)
    b1_f_train, b1_f_test, b1_t_train, b1_t_test, b1_s_train, b1_s_test = b1_split

    b2_split = train_test_split(b2_f, b2_t, b2_s, train_size=train_ratio, random_state=seed)
    b2_f_train, b2_f_test, b2_t_train, b2_t_test, b2_s_train, b2_s_test = b2_split

    b3_split = train_test_split(b3_f, b3_t, b3_s, train_size=train_ratio, random_state=seed)
    b3_f_train, b3_f_test, b3_t_train, b3_t_test, b3_s_train, b3_s_test = b3_split

    feature_train, feature_test = b1_f_train + b2_f_train + b3_f_train, b1_f_test + b2_f_test + b3_f_test
    target_train, target_test = b1_t_train + b2_t_train + b3_t_train, b1_t_test + b2_t_test + b3_t_test
    summary_train, summary_test = b1_s_train + b2_s_train + b3_s_train, b1_s_test + b2_s_test + b3_s_test

    feature_train, feature_test = np.array(feature_train).reshape(-1, 4, 1000), np.array(feature_test).reshape(-1, 4, 1000)
    target_train, target_test = np.repeat(np.array(target_train), 100, axis=0), np.repeat(np.array(target_test), 100, axis=0)
    summary_train, summary_test = np.array(summary_train).reshape(-1, 6, 100), np.array(summary_test).reshape(-1, 6, 100)
    
    feature_train_dc = feature_train[..., 500:]
    feature_test_dc = feature_test[..., 500:]

    feature_train_c = feature_train[..., :500]
    feature_test_c = feature_test[..., :500]

    print(f'Discharge curve feature shape: {feature_train_dc.shape}')
    print(f'Charge curve feature shape: {feature_train_c.shape}')
    print(f'Target shape: {target_train.shape}')
    print(f'Summary shape: {summary_train.shape}')
    
    # discharge
    np.save(save_path_dc+'trn_features_'+set_code, feature_train_dc)
    np.save(save_path_dc+'val_features_'+set_code, feature_test_dc)
    np.save(save_path_dc+'trn_targets_'+set_code, target_train)
    np.save(save_path_dc+'val_targets_'+set_code, target_test)
    np.save(save_path_dc+'trn_summary_'+set_code, summary_train)
    np.save(save_path_dc+'val_summary_'+set_code, summary_test)

    # charge
    np.save(save_path_c+'trn_features_'+set_code, feature_train_c)
    np.save(save_path_c+'val_features_'+set_code, feature_test_c)
    np.save(save_path_c+'trn_targets_'+set_code, target_train)
    np.save(save_path_c+'val_targets_'+set_code, target_test)


def predictor1_preprocess(folder='Severson_Dataset/feature_selector_discharge/'):
    selectors = []
    for i in range(1, 3, 1):
        model = discharge_model.__dict__['Dim_Reduction_'+str(i)](4, 1, 0.0).cuda()
        model.load_state_dict(torch.load('models/discharge/Dim_Reduction_'+str(i)+'_seed41.pth'))
        model.eval()
        selectors.append(model)
    trn_set = Feature_Selector_Dataset(train=True, pred_target='EOL', part='discharge')
    val_set = Feature_Selector_Dataset(train=False, pred_target='EOL', part='discharge')
    trn_summary = np.load(folder+'trn_summary.npy')
    val_summary = np.load(folder+'val_summary.npy')
    trn_feature = np.zeros((len(trn_summary), 8, 100))
    val_feature = np.zeros((len(val_summary), 8, 100))
    trn_feature[:, :6, :] = trn_summary
    val_feature[:, :6, :] = val_summary
    with torch.no_grad():
        for i in range(len(trn_summary)):
            for j, slt in enumerate(selectors):
                feature = slt(torch.tensor(trn_set[(100*i):(100*(i+1))][0]).cuda().float())
                trn_feature[i, 6+j, :] = feature.detach().cpu().numpy().squeeze()
        for i in range(len(val_summary)):
            for j, slt in enumerate(selectors):
                feature = slt(torch.tensor(val_set[(100*i):(100*(i+1))][0]).cuda().float())
                val_feature[i, 6+j, :] = feature.detach().cpu().numpy().squeeze()
    np.save(folder+'predictor1_trn_feature', trn_feature)
    np.save(folder+'predictor1_val_feature', val_feature)


def predictor3_preprocess(folder='Severson_Dataset/feature_selector_discharge/'):
    selectors = []
    for i in range(1, 5, 1):
        model = full_model.__dict__['Dim_Reduction_'+str(i)](4, 1, 0.0).cuda()
        model.load_state_dict(torch.load('models/full/Dim_Reduction_'+str(i)+'_seed41.pth'))
        model.eval()
        selectors.append(model)
    dis_trn_set = Feature_Selector_Dataset(train=True, pred_target='EOL', part='discharge')
    dis_val_set = Feature_Selector_Dataset(train=False, pred_target='EOL', part='discharge')
    ch_trn_set = Feature_Selector_Dataset(train=True, pred_target='EOL', part='charge')
    ch_val_set = Feature_Selector_Dataset(train=False, pred_target='EOL', part='charge')
    trn_summary = np.load(folder+'trn_summary.npy')
    val_summary = np.load(folder+'val_summary.npy')
    trn_feature = np.zeros((len(trn_summary), 10, 100))
    val_feature = np.zeros((len(val_summary), 10, 100))
    trn_feature[:, :6, :] = trn_summary[:, :, :100]
    val_feature[:, :6, :] = val_summary[:, :, :100]
    with torch.no_grad():
        for i in range(len(trn_summary)):
            for j, slt in enumerate(selectors):
                trn_set = dis_trn_set if j<2 else ch_trn_set
                feature = slt(torch.tensor(trn_set[(100*i):(100*(i+1))][0]).cuda().float())
                trn_feature[i, 6+j] = feature.detach().cpu().numpy().squeeze()
        for i in range(len(val_summary)):
            for j, slt in enumerate(selectors):
                val_set = dis_val_set if j<2 else ch_val_set
                feature = slt(torch.tensor(val_set[(100*i):(100*(i+1))][0]).cuda().float())
                val_feature[i, 6+j] = feature.detach().cpu().numpy().squeeze()
    np.save(folder+'predictor3_trn_feature', trn_feature)
    np.save(folder+'predictor3_val_feature', val_feature)


def dataset_preprocessing_Predictor1(cycle_used=100, dimreduct1_path='Model/eol_model1.pth', dimreduct2_path='Model/ct_model1.pth', set_code='', file_tag=''):
    """
    對 curve data 做 dimension reduction 後, 與 summary data 合併成 Predictor1 的訓練資料
    (8, 100) 的特徵: QC, QD, TAVG, TMIN, TMAX, chargetime, eol_feature, chargetime_feature
    (2,) 的預測目標: eol, eol_chargetime
    其中 eol_feature, chargetime_feature 是分別來自 Dimension reduction 1, Dimension reduction 2 的預測結果
    """
    feature_selector = [torch.load(dimreduct1_path), torch.load(dimreduct2_path)]
    summary_trn, summary_val = np.load('Severson_Dataset/feature_selector_discharge/trn_summary_'+set_code+'.npy'),\
                               np.load('Severson_Dataset/feature_selector_discharge/val_summary_'+set_code+'.npy')
    if cycle_used != 100:
        summary_trn = summary_trn[:, :, :cycle_used]
        summary_val = summary_val[:, :, :cycle_used]
        print(summary_trn.shape)

    # Dataset and DataLoader
    batch_size = cycle_used
    
    # load EOL dataset & dataloader
    eol_trn_dataset = DimReduction_dataset(train=True, pred_target='EOL', norm=True, cycle_used=cycle_used, set_code=set_code)
    eol_val_dataset = DimReduction_dataset(train=False, pred_target='EOL', norm=True, cycle_used=cycle_used, set_code=set_code)
    eol_trn_loader = DataLoader(eol_trn_dataset, batch_size=batch_size, shuffle=False)
    eol_val_loader = DataLoader(eol_val_dataset, batch_size=batch_size, shuffle=False)

    # load EOL_chargetime dataset & dataloader
    ct_trn_dataset = DimReduction_dataset(train=True, pred_target='chargetime', norm=True, cycle_used=cycle_used, set_code=set_code)
    ct_val_dataset = DimReduction_dataset(train=False, pred_target='chargetime', norm=True, cycle_used=cycle_used, set_code=set_code)
    ct_trn_loader = DataLoader(ct_trn_dataset, batch_size=batch_size, shuffle=False)
    ct_val_loader = DataLoader(ct_val_dataset, batch_size=batch_size, shuffle=False)

    # load feature selector
    if torch.cuda.is_available():
        device = torch.device('cuda')
    dimreduct1 = feature_selector[0].to(device)
    dimreduct2 = feature_selector[1].to(device)
    dimreduct1.eval()
    dimreduct2.eval()

    # predict
    pred_eol_trn, pred_eol_val = [], []
    true_eol_trn, true_eol_val = [], []
    pred_ct_trn, pred_ct_val = [], []
    true_ct_trn, true_ct_val = [], []
    with torch.no_grad():
        for inputs, targets in eol_trn_loader:
            inputs = inputs.to(device)
            pred_EOL = dimreduct1(inputs).cpu().numpy()
            
            pred_eol_trn.append(pred_EOL)
            true_eol_trn.append(targets.numpy())
        
        for inputs, targets in eol_val_loader:
            inputs = inputs.to(device)
            pred_EOL = dimreduct1(inputs).cpu().numpy()
            
            pred_eol_val.append(pred_EOL)
            true_eol_val.append(targets.numpy())
            
        for inputs, targets in ct_trn_loader:
            inputs = inputs.to(device)
            pred_EOL_chargetime = dimreduct2(inputs).cpu().numpy()

            pred_ct_trn.append(pred_EOL_chargetime)
            true_ct_trn.append(targets.numpy())

        for inputs, targets in ct_val_loader:
            inputs = inputs.to(device)
            pred_EOL_chargetime = dimreduct2(inputs).cpu().numpy()

            pred_ct_val.append(pred_EOL_chargetime)
            true_ct_val.append(targets.numpy())
    
    pred_eol_trn, pred_eol_val = np.array(pred_eol_trn), np.array(pred_eol_val)
    true_eol_trn, true_eol_val = np.array(true_eol_trn), np.array(true_eol_val)
    pred_ct_trn, pred_ct_val = np.array(pred_ct_trn), np.array(pred_ct_val)
    true_ct_trn, true_ct_val = np.array(true_ct_trn), np.array(true_ct_val)

    if cycle_used == 1:
        pred_eol_trn, pred_eol_val = np.expand_dims(pred_eol_trn, 1), np.expand_dims(pred_eol_val, 1)
        true_eol_trn, true_eol_val = np.expand_dims(true_eol_trn, 1), np.expand_dims(true_eol_val, 1)
        pred_ct_trn, pred_ct_val = np.expand_dims(pred_ct_trn, 1), np.expand_dims(pred_ct_val, 1)
        true_ct_trn, true_ct_val = np.expand_dims(true_ct_trn, 1), np.expand_dims(true_ct_val, 1)

    pred_eol_all = np.concatenate((pred_eol_trn, pred_eol_val), axis=0) # (120, 100, 1)
    true_eol_all = np.concatenate((true_eol_trn, true_eol_val), axis=0) # (120, 100)
    pred_ct_all = np.concatenate((pred_ct_trn, pred_ct_val), axis=0) # (120, 100, 1)
    true_ct_all = np.concatenate((true_ct_trn, true_ct_val), axis=0) # (120, 100)

    print('EOL RMSE of all data: '+str(mean_squared_error(true_eol_all.ravel(), pred_eol_all.ravel(), squared=False)))
    print('CT RMSE of all data: '+str(mean_squared_error(true_ct_all.ravel(), pred_ct_all.ravel(), squared=False)))

    # combine summary data & pred value of dimreduct
    feature_trn = np.concatenate((summary_trn,
                                  np.moveaxis(np.log(pred_eol_trn), 2, 1),
                                  np.moveaxis(pred_ct_trn, 2, 1)), axis=1) # (96, 8, 100)
    target_trn = np.concatenate((true_eol_trn[:, 0].reshape(-1, 1),
                                 true_ct_trn[:, 0].reshape(-1, 1)), axis=1) # (96, 2)

    feature_val = np.concatenate((summary_val,
                                  np.moveaxis(np.log(pred_eol_val), 2, 1),
                                  np.moveaxis(pred_ct_val, 2, 1)), axis=1) # (24, 8, 100)
    target_val = np.concatenate((true_eol_val[:, 0].reshape(-1, 1),
                                 true_ct_val[:, 0].reshape(-1, 1)), axis=1) # (24, 2)
    
    print(feature_trn.shape, target_trn.shape)
    print(feature_val.shape, target_val.shape)

    # save
    np.save('Severson_Dataset/DischargeDNN_TrnSet/'+'Feature_trn_'+set_code+'_'+file_tag+'.npy', feature_trn)
    np.save('Severson_Dataset/DischargeDNN_TrnSet/'+'Target_trn_'+set_code+'_'+file_tag+'.npy', target_trn)
    np.save('Severson_Dataset/DischargeDNN_ValSet/'+'Feature_val_'+set_code+'_'+file_tag+'.npy', feature_val)
    np.save('Severson_Dataset/DischargeDNN_ValSet/'+'Target_val_'+set_code+'_'+file_tag+'.npy', target_val)


def dataset_preprocessing_Predictor3(cycle_used=100,
                                     dimreduct1_path='',
                                     dimreduct2_path='',
                                     dimreduct3_path='',
                                     dimreduct4_path='',
                                     set_code='',
                                     file_tag=''):
    """
    對 curve data 做 dimension reduction 後, 與 summary data 合併成 Predictor1 的訓練資料
    (8, 100) 的特徵: QC, QD, TAVG, TMIN, TMAX, chargetime, eol_feature, chargetime_feature
    (2,) 的預測目標: eol, eol_chargetime
    其中 eol_feature, chargetime_feature 是分別來自 Dimension reduction 1, Dimension reduction 2 的預測結果
    """
    feature_selector = [torch.load(dimreduct1_path), torch.load(dimreduct2_path), torch.load(dimreduct3_path), torch.load(dimreduct4_path)]
    summary_trn, summary_val = np.load('Severson_Dataset/feature_selector_discharge/trn_summary_'+set_code+'.npy'),\
                               np.load('Severson_Dataset/feature_selector_discharge/val_summary_'+set_code+'.npy')
    if cycle_used != 100:
        summary_trn = summary_trn[:, :, :cycle_used]
        summary_val = summary_val[:, :, :cycle_used]
        print(summary_trn.shape)

    # Dataset and DataLoader
    batch_size = cycle_used
    
    # load EOL dataset & dataloader
    eol_trn_dataset_dc = DimReduction_dataset(train=True, pred_target='EOL', norm=True, cycle_used=cycle_used, segment='discharge', set_code=set_code)
    eol_val_dataset_dc = DimReduction_dataset(train=False, pred_target='EOL', norm=True, cycle_used=cycle_used, segment='discharge', set_code=set_code)
    eol_trn_dataset_c = DimReduction_dataset(train=True, pred_target='EOL', norm=True, cycle_used=cycle_used, segment='charge', set_code=set_code)
    eol_val_dataset_c = DimReduction_dataset(train=False, pred_target='EOL', norm=True, cycle_used=cycle_used, segment='charge', set_code=set_code)
    eol_trn_loader_dc = DataLoader(eol_trn_dataset_dc, batch_size=batch_size, shuffle=False)
    eol_val_loader_dc = DataLoader(eol_val_dataset_dc, batch_size=batch_size, shuffle=False)
    eol_trn_loader_c = DataLoader(eol_trn_dataset_c, batch_size=batch_size, shuffle=False)
    eol_val_loader_c = DataLoader(eol_val_dataset_c, batch_size=batch_size, shuffle=False)

    # load EOL_chargetime dataset & dataloader
    ct_trn_dataset_dc = DimReduction_dataset(train=True, pred_target='chargetime', norm=True, cycle_used=cycle_used, segment='discharge', set_code=set_code)
    ct_val_dataset_dc = DimReduction_dataset(train=False, pred_target='chargetime', norm=True, cycle_used=cycle_used, segment='discharge', set_code=set_code)
    ct_trn_dataset_c = DimReduction_dataset(train=True, pred_target='chargetime', norm=True, cycle_used=cycle_used, segment='charge', set_code=set_code)
    ct_val_dataset_c = DimReduction_dataset(train=False, pred_target='chargetime', norm=True, cycle_used=cycle_used, segment='charge', set_code=set_code)
    ct_trn_loader_dc = DataLoader(ct_trn_dataset_dc, batch_size=batch_size, shuffle=False)
    ct_val_loader_dc = DataLoader(ct_val_dataset_dc, batch_size=batch_size, shuffle=False)
    ct_trn_loader_c = DataLoader(ct_trn_dataset_c, batch_size=batch_size, shuffle=False)
    ct_val_loader_c = DataLoader(ct_val_dataset_c, batch_size=batch_size, shuffle=False)

    # load feature selector
    if torch.cuda.is_available():
        device = torch.device('cuda')
    dimreduct1 = feature_selector[0].to(device).eval()
    dimreduct2 = feature_selector[1].to(device).eval()
    dimreduct3 = feature_selector[2].to(device).eval()
    dimreduct4 = feature_selector[3].to(device).eval()

    # predict
    dim1_trn, dim1_val = [], []
    true_dim1_trn, true_dim1_val = [], []
    dim2_trn, dim2_val = [], []
    # true_dim2_trn, true_dim2_val = [], []
    dim3_trn, dim3_val = [], []
    # true_dim3_trn, true_dim3_val = [], []
    dim4_trn, dim4_val = [], []
    # true_dim4_trn, true_dim4_val = [], []
    with torch.no_grad():
        # discharge
        for inputs, targets in eol_trn_loader_dc:
            inputs = inputs.to(device)
            pred_EOL = dimreduct1(inputs).cpu().numpy()
            
            dim1_trn.append(pred_EOL)
            true_dim1_trn.append(targets.numpy())
        
        for inputs, targets in eol_val_loader_dc:
            inputs = inputs.to(device)
            pred_EOL = dimreduct1(inputs).cpu().numpy()
            
            dim1_val.append(pred_EOL)
            true_dim1_val.append(targets.numpy())
            
        for inputs, targets in ct_trn_loader_dc:
            inputs = inputs.to(device)
            pred_EOL_chargetime = dimreduct2(inputs).cpu().numpy()

            dim2_trn.append(pred_EOL_chargetime)
            # true_dim2_trn.append(targets.numpy())

        for inputs, targets in ct_val_loader_dc:
            inputs = inputs.to(device)
            pred_EOL_chargetime = dimreduct2(inputs).cpu().numpy()

            dim2_val.append(pred_EOL_chargetime)
            # true_dim2_val.append(targets.numpy())
        
        # charge
        for inputs, targets in eol_trn_loader_c:
            inputs = inputs.to(device)
            pred_EOL = dimreduct3(inputs).cpu().numpy()
            
            dim3_trn.append(pred_EOL)
            # true_dim3_trn.append(targets.numpy())
        
        for inputs, targets in eol_val_loader_c:
            inputs = inputs.to(device)
            pred_EOL = dimreduct3(inputs).cpu().numpy()
            
            dim3_val.append(pred_EOL)
            # true_dim3_val.append(targets.numpy())
            
        for inputs, targets in ct_trn_loader_c:
            inputs = inputs.to(device)
            pred_EOL_chargetime = dimreduct4(inputs).cpu().numpy()

            dim4_trn.append(pred_EOL_chargetime)
            # true_dim4_trn.append(targets.numpy())

        for inputs, targets in ct_val_loader_c:
            inputs = inputs.to(device)
            pred_EOL_chargetime = dimreduct4(inputs).cpu().numpy()

            dim4_val.append(pred_EOL_chargetime)
            # true_dim4_val.append(targets.numpy())
    
    dim1_trn, dim1_val = np.array(dim1_trn), np.array(dim1_val)
    true_dim1_trn, true_dim1_val = np.array(true_dim1_trn), np.array(true_dim1_val)
    dim2_trn, dim2_val = np.array(dim2_trn), np.array(dim2_val)
    # true_dim2_trn, true_dim2_val = np.array(true_dim2_trn), np.array(true_dim2_val)
    dim3_trn, dim3_val = np.array(dim3_trn), np.array(dim3_val)
    # true_dim3_trn, true_dim3_val = np.array(true_dim3_trn), np.array(true_dim3_val)
    dim4_trn, dim4_val = np.array(dim4_trn), np.array(dim4_val)
    # true_dim4_trn, true_dim4_val = np.array(true_dim4_trn), np.array(true_dim4_val)

    if cycle_used == 1:
        dim1_trn, dim1_val = np.expand_dims(dim1_trn, 1), np.expand_dims(dim1_val, 1)
        true_dim1_trn, true_dim1_val = np.expand_dims(true_dim1_trn, 1), np.expand_dims(true_dim1_val, 1)
        dim2_trn, dim2_val = np.expand_dims(dim2_trn, 1), np.expand_dims(dim2_val, 1)
        # true_dim2_trn, true_dim2_val = np.expand_dims(true_dim2_trn, 1), np.expand_dims(true_dim2_val, 1)
        dim3_trn, dim3_val = np.expand_dims(dim3_trn, 1), np.expand_dims(dim3_val, 1)
        # true_dim3_trn, true_dim3_val = np.expand_dims(true_dim3_trn, 1), np.expand_dims(true_dim3_val, 1)
        dim4_trn, dim4_val = np.expand_dims(dim4_trn, 1), np.expand_dims(dim4_val, 1)
        # true_dim4_trn, true_dim4_val = np.expand_dims(true_dim4_trn, 1), np.expand_dims(true_dim4_val, 1)

    # combine summary data & pred value of dimreduct
    feature_trn = np.concatenate((summary_trn,
                                  np.moveaxis(np.log(dim1_trn), 2, 1),
                                  np.moveaxis(dim2_trn, 2, 1),
                                  np.moveaxis(np.log(dim3_trn), 2, 1),
                                  np.moveaxis(dim4_trn, 2, 1)), axis=1) # (91, 8, 100)
    target_trn = true_dim1_trn[:, 0].reshape(-1, 1) # (91, 1)

    feature_val = np.concatenate((summary_val,
                                  np.moveaxis(np.log(dim1_val), 2, 1),
                                  np.moveaxis(dim2_val, 2, 1),
                                  np.moveaxis(np.log(dim3_val), 2, 1),
                                  np.moveaxis(dim4_val, 2, 1)), axis=1) # (24, 8, 100)
    
    target_val = true_dim1_val[:, 0].reshape(-1, 1) # (24, 1)
    
    print(feature_trn.shape, target_trn.shape)
    print(feature_val.shape, target_val.shape)

    # save
    np.save('Severson_Dataset/FullModel_TrnSet/'+'Feature_trn_'+set_code+'_'+file_tag+'.npy', feature_trn)
    np.save('Severson_Dataset/FullModel_TrnSet/'+'Target_trn_'+set_code+'_'+file_tag+'.npy', target_trn)
    np.save('Severson_Dataset/FullModel_ValSet/'+'Feature_val_'+set_code+'_'+file_tag+'.npy', feature_val)
    np.save('Severson_Dataset/FullModel_ValSet/'+'Target_val_'+set_code+'_'+file_tag+'.npy', target_val)

if __name__ == '__main__':
    # mat_to_npy()
    # data_visualization(3)
    # data_visualization(4)
    # data_visualization(5)
    # train_val_split()
    # train_val_split(seed=41)
    # predictor1_preprocess()
    # dataset_preprocessing_Predictor1(set_code='seed15', file_tag='f100')
    # train_val_split_by_batch()
    cycle_info = np.load('Severson_Dataset/feature_selector_discharge/trn_features_state1.npy')
    print(cycle_info.shape)