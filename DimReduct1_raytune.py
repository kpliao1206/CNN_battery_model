from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from Pytorch_dataset import DimReduction_dataset
from Discharge_model import DimReduction_1
from torchmetrics.functional import mean_absolute_percentage_error, mean_squared_error
from torch.optim.lr_scheduler import CosineAnnealingLR

################### Check file path #####################
seed = 2
set_code = 'state2'
model_num = 'raytune1'
#########################################################


def load_data():
    trainset = DimReduction_dataset(train=True, pred_target='EOL', norm=True, cycle_used=100, set_code=set_code)
    testset = DimReduction_dataset(train=False, pred_target='EOL', norm=True, cycle_used=100, set_code=set_code)
    
    return trainset, testset


def train_dimreduct1(config, checkpoint_dir=r"F:\KuoPing\Severson_data\Re-preprocess\Checkpoints"):

    dimreduct1 = DimReduction_1()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            dimreduct1 = nn.DataParallel(dimreduct1)
    dimreduct1.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(dimreduct1.parameters(), lr=config['lr'], amsgrad=True, weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["T_max"], eta_min=config["eta_min"])

    loaded_checkpoint = session.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
           model_state, optimizer_state = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
 
        dimreduct1.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainset, testset = load_data()

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=50,
        shuffle=True,
        num_workers=8)
    valloader = torch.utils.data.DataLoader(
        testset,
        batch_size=50,
        shuffle=True,
        num_workers=8)

    for epoch in range(200):  # loop over the dataset multiple times
        #### Training loop ####
        dimreduct1.train()
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.view(-1, 1).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = dimreduct1(inputs).view(-1, 1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0
        
        # update lr
        scheduler.step()

        #### Validation loop ####
        dimreduct1.eval()
        val_loss = 0.0
        val_steps = 0
        total = 0
        rmse_lst = []
        mape_lst = []
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.view(-1, 1).to(device)

                outputs = dimreduct1(inputs).view(-1, 1)
                rmse = mean_squared_error(outputs.view(-1, 1), targets.view(-1, 1), squared=False)
                mape = mean_absolute_percentage_error(outputs.view(-1, 1), targets.view(-1, 1))
                rmse_lst.append(rmse.item())
                mape_lst.append(mape.item())
                total += targets.size(0)

                loss = criterion(outputs, targets)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        os.makedirs("my_model", exist_ok=True)
        torch.save(
            (dimreduct1.state_dict(), optimizer.state_dict()), "my_model/checkpoint.pt")
        checkpoint = Checkpoint.from_directory("my_model")
        session.report({"loss": (val_loss / val_steps), "RMSE": np.average(rmse_lst), "MAPE": np.average(mape_lst)}, checkpoint=checkpoint)
        # with tune.checkpoint_dir(epoch) as checkpoint_dir:
        #     path = os.path.join(checkpoint_dir, "checkpoint")
        #     torch.save((predictor1.state_dict(), optimizer.state_dict()), path)

        # tune.report(loss=(val_loss / val_steps), RMSE=np.average(rmse_lst), MAPE=np.average(mape_lst))
    print("Finished Training")


def test_accuracy(predictor1, device="cpu"):
    trainset, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=50, shuffle=False, num_workers=2)

    rmse = 0.0
    total = 0
    rmse_lst = []
    mape_lst = []
    with torch.no_grad():
        for data in testloader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.view(-1, 1).to(device)
            outputs = predictor1(inputs).view(-1, 1)
            total += outputs.size(0)
            rmse = mean_squared_error(outputs.view(-1, 1), targets.view(-1, 1), squared=False)
            mape = mean_absolute_percentage_error(outputs.view(-1, 1), targets.view(-1, 1))
            rmse_lst.append(rmse.item())
            mape_lst.append(mape.item())
    return np.average(rmse_lst), np.average(mape_lst)


if __name__ == "__main__":
    data_dir = 'data'
    num_samples = 10
    max_num_epochs = 60
    gpus_per_trial = 1

    config = {
        "lr": tune.loguniform(1e-3, 1e-1),
        "weight_decay": tune.loguniform(1e-4, 1e-1),
        # "batch_size": tune.choice([50, 100, 150, 200]),
        # "drop": tune.uniform(0.1, 0.4),
        "T_max": tune.uniform(5, 20),
        "eta_min": tune.loguniform(1e-6, 1e-4)
    }
    
    load_data()

    scheduler = ASHAScheduler(
        # metric="loss",
        # mode="min",
        max_t=max_num_epochs,
        grace_period=10,
        reduction_factor=2)
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_dimreduct1),
            resources={"cpu": 6, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
    )
    results = tuner.fit()

    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))

    best_trained_model = DimReduction_1()
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_result.checkpoint.to_directory()
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint.pt"))
    best_trained_model.load_state_dict(model_state)

    test_rmse, test_mape = test_accuracy(best_trained_model, device)
    print("Best trial test set RMSE: {}".format(test_rmse))
    print("Best trial test set MAPE: {}".format(test_mape))

    # save model
    torch.save(best_trained_model, 'Model/dimreduct1_'+model_num+'.pth')

