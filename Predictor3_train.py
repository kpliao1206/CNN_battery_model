import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from Pytorch_dataset import Predictor3_dataset
from torchmetrics.functional import mean_absolute_percentage_error, mean_squared_error

def train_model(model,
                train_loader, test_loader,
                num_epochs=500,
                lr=1e-3,
                weight_decay=1e-3,
                cosine_period=20,
                min_lr=1e-6,
                patience=20,
                loss_func='MSE',
                state_dict_path=''):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('cuda is available')

    optimizer = optim.AdamW(model.parameters(), lr=lr, amsgrad=True, weight_decay=weight_decay) # Adam with weight decay
    scheduler = CosineAnnealingLR(optimizer, T_max=cosine_period, eta_min=min_lr) # 依照cosine週期衰減
    if loss_func == 'L1':
        criterion = nn.L1Loss()
    elif loss_func == 'MSE':
        criterion = nn.MSELoss()

    start = time.time()

    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    train_rmses = []
    valid_rmses = []
    avg_train_rmse = []
    avg_valid_rmse = []
    n_total_steps = len(train_loader)

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(num_epochs):

        ##### Training loop #####
        model.train() # prep model for training
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            rmse = mean_squared_error(outputs.view(-1, 1), targets.view(-1, 1), squared=False)

            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_rmses.append(rmse.item())
            
        # update lr
        scheduler.step()

        ##### Validation loop #####
        model.eval() # prep model for evaluation
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs[:, 0], targets[:, 0])
            rmse = mean_squared_error(outputs.view(-1, 1), targets.view(-1, 1), squared=False)
            valid_losses.append(loss.item())
            valid_rmses.append(rmse.item())
        
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        train_rmse = np.average(train_rmses)
        valid_rmse = np.average(valid_rmses)

        print(f'[Epoch {epoch+1}/{num_epochs}] train_loss: {train_loss:.3f}, valid_loss: {valid_loss:.3f}')
        print(f'....train_rmse: {train_rmse:.2f}, valid_rmse: {valid_rmse:.2f}')

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # save state_dict
        # 創建資料夾
        if (epoch+1)>=10 and (epoch+1)<=40:
            path = "F:/KuoPing/Severson_data/Re-preprocess/Full_model/Saved_predictor3/" + state_dict_path
            if not os.path.isdir(path):
                os.makedirs(path)
            torch.save(model, path+'/epoch'+str(epoch+1))

        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        # if epoch+1 > 10:
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch{epoch+1}")
            break
        
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('Checkpoints/checkpoint.pt'))

    end = time.time()
    print(f'Training is end. Total trainig time: {(end-start)/60:.1f} minutes')

    return  model, avg_train_losses, avg_valid_losses


def train_model_RLROP(model,
                      train_loader, test_loader,
                      num_epochs=500,
                      lr=1e-3,
                      weight_decay=1e-3,
                      factor=0.3,
                      min_lr=1e-6,
                      scheduler_patience=5,
                      verbose=False,
                      patience=20,
                      loss_func='MSE',
                      state_dict_path=''):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('cuda is available')

    optimizer = optim.AdamW(model.parameters(), lr=lr, amsgrad=True, weight_decay=weight_decay) # Adam with weight decay
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=scheduler_patience, verbose=verbose, min_lr=min_lr) # 依照cosine週期衰減
    if loss_func == 'L1':
        criterion = nn.L1Loss()
    elif loss_func == 'MSE':
        criterion = nn.MSELoss()

    start = time.time()

    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    train_rmses = []
    valid_rmses = []
    avg_train_rmse = []
    avg_valid_rmse = []
    n_total_steps = len(train_loader)

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(num_epochs):

        ##### Training loop #####
        model.train() # prep model for training
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            rmse = mean_squared_error(outputs.view(-1, 1), targets.view(-1, 1), squared=False)

            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_rmses.append(rmse.item())

        ##### Validation loop #####
        model.eval() # prep model for evaluation
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs[:, 0], targets[:, 0])
            rmse = mean_squared_error(outputs.view(-1, 1), targets.view(-1, 1), squared=False)
            valid_losses.append(loss.item())
            valid_rmses.append(rmse.item())
        # update lr
        scheduler.step(loss)
        
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        train_rmse = np.average(train_rmses)
        valid_rmse = np.average(valid_rmses)

        print(f'[Epoch {epoch+1}/{num_epochs}] train_loss: {train_loss:.3f}, valid_loss: {valid_loss:.3f}')
        print(f'....train_rmse: {train_rmse:.2f}, valid_rmse: {valid_rmse:.2f}')

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # save state_dict
        # 創建資料夾
        if (epoch+1)>=10 and (epoch+1)<=40:
            path = "F:/KuoPing/Severson_data/Re-preprocess/Full_model/Saved_predictor3/" + state_dict_path
            if not os.path.isdir(path):
                os.makedirs(path)
            torch.save(model, path+'/epoch'+str(epoch+1))

        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        # if epoch+1 > 10:
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



def model_evaluate(model, norm=True, train_batch_size=91, test_batch_size=24, model_num=0, cycle_used=100, set_code='', file_tag=''):
    # load train data
    train_dataset = Predictor3_dataset(train=True, norm=norm, padding=True, eval=True, cycle_used=cycle_used,
                                       set_code=set_code,
                                       file_tag=file_tag)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    # load test data
    test_dataset = Predictor3_dataset(train=False, norm=norm, padding=True, eval=True, cycle_used=cycle_used,
                                      set_code=set_code,
                                      file_tag=file_tag)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

    model = model.cuda()
    train_rmse_lst = []
    test_rmse_lst = []
    train_mape_lst = []
    test_mape_lst = []

    fig, ax = plt.subplots(figsize=(6, 6))
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.cuda()
            print(inputs.shape)
            targets = targets.view(-1, 1)
            outputs = model(inputs).view(-1, 1)

            mape_train = mean_absolute_percentage_error(outputs.view(-1, 1), targets.view(-1, 1).cuda())
            rmse_train = mean_squared_error(outputs.view(-1, 1), targets.view(-1, 1).cuda(), squared=False)
            
            train_rmse_lst.append(rmse_train.item())
            train_mape_lst.append(mape_train.item())
            if i == 0:
                ax.plot(outputs.cpu(), targets, 'ro', markersize=5, label='train')
            else:
                ax.plot(outputs.cpu(), targets, 'ro', markersize=5)
        print(f'Training set|RMSE: {np.average(train_rmse_lst):.2f}, MAPE: {np.average(train_mape_lst):.2f}')

        for i, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.cuda()
            targets = targets
            outputs = model(inputs)
            
            mape_test = mean_absolute_percentage_error(outputs.view(-1, 1), targets.view(-1, 1).cuda())
            rmse_test = mean_squared_error(outputs.view(-1, 1), targets.view(-1, 1).cuda(), squared=False)
            
            test_rmse_lst.append(rmse_test.item())
            test_mape_lst.append(mape_test.item())
            # plot
            if i == 0:
                ax.plot(outputs.cpu(), targets, 'bo', markersize=5, label='valid')
            else:
                ax.plot(outputs.cpu(), targets, 'bo', markersize=5)
        print(f'Testing set|RMSE: {np.average(test_rmse_lst):.2f}, MAPE: {np.average(test_mape_lst):.2f}')
        
        ax.set_title('Predictor3 EOL')
        # plt.plot(, [12.5, 27.5], 'k--', linewidth=1.0)
        xlim, ylim = 0, 2000
        ax.set_xlim(xlim, ylim)
        ax.set_ylim(xlim, ylim)
        ax.plot([xlim, ylim], [xlim, ylim], 'k--', linewidth=1.5)
        ax.set_xlabel('predicted')
        ax.set_ylabel('ground truth')
        ax.legend()
        plt.savefig('Figures/predictor3_eval_'+str(model_num)+'.jpg')
        plt.show()
        plt.close()
    
    return np.average(train_rmse_lst), np.average(test_rmse_lst)