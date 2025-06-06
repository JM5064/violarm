import random
import os
from PIL import Image
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torchvision
from torchvision import transforms, datasets

from model import Model


def to_device(obj):
    if torch.cuda.is_available():
        obj = obj.to("cuda")

    return obj


def log_results(file, metrics):
    for metric in metrics:
        file.write(f'{metric}: {metrics[metric]}\t')

    file.write('\n')


def compute_class_weights(train_dir_path):
    counts = {}
    classes = []
    total = 0

    for pose in os.listdir(train_dir_path):
        if not os.path.isdir(f'{train_dir_path}/{pose}'):
            continue
        
        counts[pose] = len(os.listdir(f'{train_dir_path}/{pose}'))
        classes.append(pose)
        total += counts[pose]

    classes.sort()
    weights = [total / (len(classes) * counts[pose]) for pose in classes]
    
    return torch.tensor(weights, dtype=torch.float)
    

def validate(model, val_loader, loss_func):
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs = to_device(inputs)
            labels = to_device(labels)

            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            running_loss += loss.item()

            all_preds.extend(outputs.cpu())
            all_labels.extend(labels.cpu())


    average_val_loss = running_loss / len(val_loader)
    
    preds_concat = torch.cat(all_preds)
    labels_concat = torch.cat(all_labels)

    mae = torch.mean(torch.abs(preds_concat - labels_concat)).item()

    preds_kp = preds_concat.view(-1, 4, 2) # 4 keypoints
    labels_kp = labels_concat.view(-1, 4, 2)

    distances = torch.norm(preds_kp - labels_kp, dim=2)
    correct = (distances < 0.05).float() # if distance is < 0.05, count as correct
    pck = correct.mean().item()

    metrics = {
        "mae": mae,
        "pck": pck,
        "average_val_loss": average_val_loss,
    }

    return metrics


def train(model, num_epochs, train_loader, val_loader, test_loader, loss_func=nn.MSELoss(), optimizer=optim.AdamW, optimizer_params=None, runs_dir="arm_model/runs"):
    # create log file
    time = str(datetime.now())
    os.mkdir(runs_dir + "/" + time)
    logfile = open(runs_dir + "/" + time + "/metrics.txt", "a")
    best_val_loss = float('inf')

    optimizer = optimizer(filter(lambda p: p.requires_grad, model.parameters()), **optimizer_params)

    # training loop
    for i in range(num_epochs):
        print(f'Epoch {i+1}/{num_epochs}')

        if i == 12:
            print("Switching learning rate")
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-4

        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            inputs = to_device(inputs)
            labels = to_device(labels)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # print and log metrics
        average_train_loss = running_loss / len(train_loader)
        metrics = validate(model, val_loader, loss_func)
        metrics["average_train_loss"] = average_train_loss

        print(f'Epoch {i+1} Results:')
        print(f'Train Loss: {average_train_loss}\tValidation Loss: {metrics["average_val_loss"]}')
        print(f'MAE: {metrics["mae"]}\tPCK: {metrics["pck"]}')

        log_results(logfile, metrics)

        # save best model based on avg val loss
        average_val_loss = metrics["average_val_loss"]
        if average_val_loss < best_val_loss:
            torch.save(model.state_dict(), runs_dir + "/" + time + "/best.pt")
            best_val_loss = average_val_loss

        torch.save(model.state_dict(), runs_dir + "/" + time + "/last.pt")

    # test model and print/log testing metrics
    print("Testing Model")
    metrics = validate(model, test_loader, loss_func)
    print("Testing Results")
    print(f'MAE: {metrics["mae"]}\tPCK: {metrics["pck"]}')
    print(f'Test Loss: {metrics["average_val_loss"]}')

    test_logfile = open(runs_dir + "/" + time + "/test_metrics.txt", "a")
    log_results(test_logfile, metrics)

