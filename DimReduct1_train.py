import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from Pytorch_dataset import DimReduction_dataset
from Discharge_model import DimReduction_1
from early_stopping import EarlyStopping
from torch.utils.tensorboard import SummaryWriter

# TensorBoard
writer = SummaryWriter('runs/eol_5')

# params
num_epochs = 50
batch_size = 32
lr = 1e-2
weight_decay = 1e-3
cosine_period = 5
min_lr = 1e-5
delta_huber = 1
FILE = "Model/eol_model5.pth"

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('cuda is available')

eol_train_dataset = DimReduction_dataset(train=True, pred_target='EOL', norm=False)
eol_test_dataset = DimReduction_dataset(train=False, pred_target='EOL', norm=False)

eol_train_loader = DataLoader(eol_train_dataset, batch_size=batch_size, shuffle=True)
eol_test_loader = DataLoader(eol_test_dataset, batch_size=batch_size, shuffle=False)

model_eol = DimReduction_1(in_ch=4, out_ch=1).to(device) # weights initialize by xavier normal distribution
optimizer = optim.AdamW(model_eol.parameters(), lr=lr, amsgrad=True, weight_decay=weight_decay) # Adam with weight decay
scheduler = CosineAnnealingLR(optimizer, T_max=cosine_period, eta_min=min_lr) # 依照cosine週期衰減
criterion = nn.HuberLoss(delta=delta_huber) # combines advantages of both L1Loss and MSELoss

# draw model graph
# example = iter(eol_train_loader)
# d, t = next(example)
# print(d.shape, t.shape)
# writer.add_graph(model_eol, d.to(device))
# writer.close()
# sys.exit()

def train_model(model,
                file = "Model/eol_model0.pth",
                num_epochs=500,
                lr=1e-3,
                weight_decay=1e-3,
                cosine_period=5,
                min_lr=1e-5,
                delta_huber=1,
                patience=20):

    optimizer = optim.AdamW(model_eol.parameters(), lr=lr, amsgrad=True, weight_decay=weight_decay) # Adam with weight decay
    scheduler = CosineAnnealingLR(optimizer, T_max=cosine_period, eta_min=min_lr) # 依照cosine週期衰減
    criterion = nn.HuberLoss(delta=delta_huber) # combines advantages of both L1Loss and MSELoss

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

            # if (i+1) % 60 == 0:
            #     print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item()}')
            #     # tensorboard
            #     writer.add_scalar('training loss', record_loss / 60, epoch * n_total_steps +i) # global step
            #     record_loss = 0.0

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

        print(f'[Epoch {epoch+1}/{num_epochs}] train_loss: {train_loss:.2f}, valid_loss: {valid_loss:.2f}')

        # tensorboard
        writer.add_scalar('train loss', train_loss, epoch * n_total_steps +i) # global step
        writer.add_scalar('valid loss', valid_loss, epoch * n_total_steps +i)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))

    end = time.time()
    print(f'Training is end. Total trainig time: {(end-start)/60:.1f} minutes')

    return  model, avg_train_losses, avg_valid_losses

# save model
# torch.save(model, file)
# print('Model has been saved.')
            




