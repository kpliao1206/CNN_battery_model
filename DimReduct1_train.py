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

# training
start = time.time()
record_loss = 0.0
n_total_steps = len(eol_train_loader)
for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(eol_train_loader):
        inputs = inputs.to(device)
        targets = targets.view(-1, 1).to(device)

        # forward
        outputs = model_eol(inputs)
        loss = criterion(outputs, targets)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        record_loss += loss.item()

        if (i+1) % 60 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item()}')
            # tensorboard
            writer.add_scalar('training loss', record_loss / 60, epoch * n_total_steps +i) # global step
            record_loss = 0.0

end = time.time()
print(f'Training is end. Total trainig time: {(end-start)/60:.1f} minutes')
        
        # if i == 0:
        #     print(outputs, targets)
        # if i == 10:
        #     print(outputs, targets)
        # if i == 20:
        #     print(outputs, targets)

# save model
torch.save(model_eol, FILE)
print('Model has been saved.')
            




