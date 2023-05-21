import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from Pytorch_dataset import DimReduction_dataset
from Discharge_model import DimReduction_1, DimReduction_2
from early_stopping import EarlyStopping
from torchmetrics.functional import mean_absolute_percentage_error, mean_squared_error


def dimreduct1_train(model, eol_train_loader, eol_test_loader,
                num_epochs=500,
                lr=1e-3,
                weight_decay=1e-3,
                cosine_period=20,
                min_lr=1e-6,
                delta_huber=1,
                patience=20,
                state_dict_path=''):
                
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('cuda is available')

    optimizer = optim.AdamW(model.parameters(), lr=lr, amsgrad=True, weight_decay=weight_decay) # Adam with weight decay
    scheduler = CosineAnnealingLR(optimizer, T_max=cosine_period, eta_min=min_lr) # 依照cosine週期衰減
    criterion = nn.MSELoss() # combines advantages of both L1Loss and MSELoss

    start = time.time()

    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    n_total_steps = len(eol_train_loader)

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(num_epochs):

        ##### Training loop #####
        model.train() # prep model for training
        for i, (inputs, targets) in enumerate(eol_train_loader):
            inputs = inputs.to(device)
            targets = targets.view(-1, 1).to(device)

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            
        # update lr
        scheduler.step()

        ##### Validation loop #####
        model.eval() # prep model for evaluation
        for inputs, targets in eol_test_loader:
            inputs = inputs.to(device)
            targets = targets.view(-1, 1).to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            valid_losses.append(loss.item())
        
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        # print(f'[Epoch {epoch+1}/{num_epochs}] train_loss: {train_loss:.2f}, valid_loss: {valid_loss:.2f}')

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # save state_dict
        # 創建資料夾
        if (epoch+1)>=20 and (epoch+1)<=50:
            path = "F:/KuoPing/Severson_data/Re-preprocess/State_dict_dimreduct1/" + state_dict_path
            if not os.path.isdir(path):
                os.makedirs(path)
            torch.save(model, path+'/epoch'+str(epoch+1))

        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        # if epoch+1 > 10: # 至少訓練30個epoch才開始找early stop
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch{epoch+1}")
            break
        
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('Checkpoints/checkpoint.pt'))

    end = time.time()
    print('====================================================')
    print(f'Training is end. Total trainig time: {(end-start)/60:.1f} minutes')

    return  model, avg_train_losses, avg_valid_losses


def dimreduct2_train(model, ct_train_loader, ct_test_loader,
                num_epochs=500,
                lr=1e-3,
                weight_decay=1e-3,
                cosine_period=20,
                min_lr=1e-6,
                delta_huber=1,
                patience=20,
                state_dict_path=''):

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('cuda is available')
            

    optimizer = optim.AdamW(model.parameters(), lr=lr, amsgrad=True, weight_decay=weight_decay) # Adam with weight decay
    scheduler = CosineAnnealingLR(optimizer, T_max=cosine_period, eta_min=min_lr) # 依照cosine週期衰減
    criterion = nn.MSELoss() # combines advantages of both L1Loss and MSELoss

    start = time.time()

    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    n_total_steps = len(ct_train_loader)

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(num_epochs):

        ##### Training loop #####
        model.train() # prep model for training
        for i, (inputs, targets) in enumerate(ct_train_loader):
            inputs = inputs.to(device)
            targets = targets.view(-1, 1).to(device)

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            
        # update lr
        scheduler.step()

        ##### Validation loop #####
        model.eval() # prep model for evaluation
        for inputs, targets in ct_test_loader:
            inputs = inputs.to(device)
            targets = targets.view(-1, 1).to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            valid_losses.append(loss.item())
        
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        # print(f'[Epoch {epoch+1}/{num_epochs}] train_loss: {train_loss:.2f}, valid_loss: {valid_loss:.2f}')

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        if (epoch+1)>=20 and (epoch+1)<=50:
            path = "F:/KuoPing/Severson_data/Re-preprocess/State_dict_dimreduct2/" + state_dict_path
            if not os.path.isdir(path):
                os.makedirs(path)
            torch.save(model, path+'/epoch'+str(epoch+1))

        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        # if epoch+1 > 10: # 至少訓練10個epoch才開始找early stop
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch{epoch+1}")
            break
        
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('Checkpoints/checkpoint.pt'))

    end = time.time()
    print(f'Training is end. Total trainig time: {(end-start)/60:.1f} minutes')

    return  model, avg_train_losses, avg_valid_losses


def loss_plot(avg_train_losses, avg_test_losses):
    plt.figure()
    plt.plot(avg_train_losses, 'r-', label='train')
    plt.plot(avg_test_losses, 'b-', label='valid')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('recorded loss')
    plt.legend()
    plt.show()


def dimreduct1_eval(eol_model, batch_size=32, model_num=0, t_norm=False, set_code=''):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('cuda is available')

    # load train data
    eol_train_dataset = DimReduction_dataset(train=True, pred_target='EOL', norm=True, set_code=set_code)
    eol_train_loader = DataLoader(eol_train_dataset, batch_size=1, shuffle=True)

    # load test dataDataLoader(eol_train_dataset, batch_size=batch_size, shuffle=False)
    eol_test_dataset = DimReduction_dataset(train=False, pred_target='EOL', norm=True, set_code=set_code)
    eol_test_loader = DataLoader(eol_test_dataset, batch_size=1, shuffle=True)

    eol_model = eol_model.to(device).eval()
    train_rmse_lst = []
    test_rmse_lst = []
    train_mape_lst = []
    test_mape_lst = []
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(eol_train_loader):
            inputs = inputs.to(device)
            targets = targets
            outputs = eol_model(inputs)

            mape_train = mean_absolute_percentage_error(outputs.view(-1, 1), targets.view(-1, 1).cuda())
            rmse_train = mean_squared_error(outputs.view(-1, 1), targets.view(-1, 1).cuda(), squared=False)
            
            train_rmse_lst.append(rmse_train.item())
            train_mape_lst.append(mape_train.item())
            if i == 0:
                plt.plot(outputs.cpu(), targets, 'ro', markersize=2.5, label='train')
            else:
                plt.plot(outputs.cpu(), targets, 'ro', markersize=2.5)
        print(f'Training set|RMSE: {np.average(train_rmse_lst):.2f}, MAPE: {np.average(train_mape_lst):.2f}')
        # print(train_rmse_lst)

        for i, (inputs, targets) in enumerate(eol_test_loader):
            inputs = inputs.to(device)
            targets = targets
            outputs = eol_model(inputs)
            
            mape_test = mean_absolute_percentage_error(outputs.view(-1, 1), targets.view(-1, 1).cuda())
            rmse_test = mean_squared_error(outputs.view(-1, 1), targets.view(-1, 1).cuda(), squared=False)
            
            test_rmse_lst.append(rmse_test.item())
            test_mape_lst.append(mape_test.item())
            # plot
            if i == 0:
                plt.plot(outputs.cpu(), targets, 'bo', markersize=2.5, label='valid')
            else:
                plt.plot(outputs.cpu(), targets, 'bo', markersize=2.5)
        print(f'Testing set|RMSE: {np.average(test_rmse_lst):.2f}, MAPE: {np.average(test_mape_lst):.2f}')
        # print(test_rmse_lst)

        plt.title('EOL model_'+str(model_num))
        plt.plot([0,2000], [0,2000], 'k--', linewidth=1.0)
        plt.xlabel('predicted')
        plt.ylabel('ground truth')
        plt.legend()
        plt.savefig('Figures/eol_model'+str(model_num)+'.jpg')


def dimreduct2_eval(eol_model, batch_size=23, model_num=0, t_norm=False, set_code=''):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('cuda is available')

    # load train data
    eol_train_dataset = DimReduction_dataset(train=True, pred_target='chargetime', norm=True, set_code=set_code)
    eol_train_loader = DataLoader(eol_train_dataset, batch_size=1, shuffle=True)

    # load test dataDataLoader(eol_train_dataset, batch_size=batch_size, shuffle=False)
    eol_test_dataset = DimReduction_dataset(train=False, pred_target='chargetime', norm=True, set_code=set_code)
    eol_test_loader = DataLoader(eol_test_dataset, batch_size=1, shuffle=True)

    eol_model = eol_model.to(device).eval()
    train_rmse_lst = []
    test_rmse_lst = []
    train_mape_lst = []
    test_mape_lst = []
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(eol_train_loader):
            inputs = inputs.to(device)
            targets = targets
            outputs = eol_model(inputs)

            mape_train = mean_absolute_percentage_error(outputs.view(-1, 1), targets.view(-1, 1).cuda())
            rmse_train = mean_squared_error(outputs.view(-1, 1), targets.view(-1, 1).cuda(), squared=False)
            
            train_rmse_lst.append(rmse_train.item())
            train_mape_lst.append(mape_train.item())
            if i == 0:
                plt.plot(outputs.cpu(), targets, 'ro', markersize=2.5, label='train')
            else:
                plt.plot(outputs.cpu(), targets, 'ro', markersize=2.5)
        print(f'Training set|RMSE: {np.average(train_rmse_lst):.2f}, MAPE: {np.average(train_mape_lst):.2f}')
        # print(train_rmse_lst)

        for i, (inputs, targets) in enumerate(eol_test_loader):
            inputs = inputs.to(device)
            targets = targets
            outputs = eol_model(inputs)
            
            mape_test = mean_absolute_percentage_error(outputs.view(-1, 1), targets.view(-1, 1).cuda())
            rmse_test = mean_squared_error(outputs.view(-1, 1), targets.view(-1, 1).cuda(), squared=False)
            
            test_rmse_lst.append(rmse_test.item())
            test_mape_lst.append(mape_test.item())
            # plot
            if i == 0:
                plt.plot(outputs.cpu(), targets, 'bo', markersize=2.5, label='valid')
            else:
                plt.plot(outputs.cpu(), targets, 'bo', markersize=2.5)
        print(f'Testing set|RMSE: {np.average(test_rmse_lst):.2f}, MAPE: {np.average(test_mape_lst):.2f}')
        # print(test_rmse_lst)

        plt.title('Chargetime model_'+str(model_num))
        plt.plot([12.5, 27.5], [12.5, 27.5], 'k--', linewidth=1.0)
        plt.xlabel('predicted (mins)')
        plt.ylabel('ground truth (mins)')
        plt.legend()
        plt.savefig('Figures/ct_model'+str(model_num)+'.jpg')