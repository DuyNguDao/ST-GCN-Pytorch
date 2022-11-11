import os
import time
import torch
import pickle
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
from torch.utils import data
from torch.optim.adadelta import Adadelta
from models.stgcn import *
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
import logging
import yaml
from dataloader.dataset import processing_data
import datetime


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# clear memory cuda
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
# Get parameter
with open("./config.yaml", "r") as stream:
    try:
        data = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# parameter
input_dataset_train = data['dataset-path-train']
input_dataset_test = data['dataset-path-test']
epochs = data['epochs']
batch_size = data['batch-size']
input_size = data['img-size']
num_frame = data['num-frame']
path_save_model = data['project']

features, labels = [], []
# Load dataset train
with open(input_dataset_train, 'rb') as f:
    fts, lbs = pickle.load(f)
    features.append(fts)
    labels.append(lbs)
del fts, lbs

# ****************************************** NORMALIZE CLASS ****************************************************
labels = np.concatenate(labels, axis=0)
features = np.concatenate(features, axis=0)  # 30x34
# get 15 frame
features = features[:, ::2, :, :]

# add center point with yolov3
features = np.concatenate((features, np.expand_dims((features[:, :, 1, :] + features[:, :, 1, :]) / 2, axis=2)), axis=2)

features[:, :, :, :2] = processing_data(features[:, :, :, :2])
x_train = features
y_train = labels
print(" --------- Number class train---------")
for i in range(7):
    print(f"class {i}: {labels.tolist().count(i)}")

features, labels = [], []
with open(input_dataset_test, 'rb') as f:
    fts, lbs = pickle.load(f)
    features.append(fts)
    labels.append(lbs)
del fts, lbs
# ****************************************** NORMALIZE CLASS ****************************************************
labels = np.concatenate(labels, axis=0)
features = np.concatenate(features, axis=0)  # 30x34
# get 15 frame
features = features[:, ::2, :, :]
features = np.concatenate((features, np.expand_dims((features[:, :, 1, :] + features[:, :, 1, :]) / 2, axis=2)), axis=2)

features[:, :, :, :2] = processing_data(features[:, :, :, :2])
x_valid = features
y_valid = labels
print(" --------- Number class test---------")
for i in range(7):
    print(f"class {i}: {labels.tolist().count(i)}")

del features, labels

train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32).permute(0, 3, 1, 2),
                              torch.tensor(y_train, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(x_valid, dtype=torch.float32).permute(0, 3, 1, 2),
                            torch.tensor(y_valid, dtype=torch.float32))

del x_train, x_valid, y_train, y_valid

# create folder save
if not os.path.exists(path_save_model):
    os.mkdir(path_save_model)
count = 0
# check path save
while os.path.exists(path_save_model + f'/exp{count}'):
    count += 1
# create new folder save
path_save_model = path_save_model + f'/exp{count}'
os.mkdir(path_save_model)

# load data loader
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size, shuffle=True,
    num_workers=batch_size, pin_memory=True)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size, shuffle=False,
    num_workers=batch_size, pin_memory=True)

del train_dataset, val_dataset


def set_training(model, mode=True):
    for p in model.parameters():
        p.requires_grad = mode
    model.train(mode)
    return model


classes_name = ['Sit down', 'Lying Down', 'Walking', 'Stand up', 'Standing', 'Fall Down', 'Sitting']

# classes_name = ['Fall Down', 'Other action']
# classes_name = ['Siting', 'Lying Down', 'Walking or Standing', 'Fall Down']
print("Class name:", classes_name)

# MODEL.
# config 14 pose
graph_args = {'strategy': 'spatial', 'layout': 'coco_cut'}
# config 17 pose
model = TwoStreamSpatialTemporalGraph(graph_args, len(classes_name)).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer = Adadelta(model.parameters())
losser = torch.nn.BCELoss()
# losser = torch.nn.CrossEntropyLoss()


def train_model(model, losser, optimizer, num_epochs):
    # TRAINING.
    best_loss_acc = -1
    loss_list = {'train': [], 'valid': []}
    acc_list = {'train': [], 'valid': []}
    for epoch in range(num_epochs):
        # train
        losses_train = 0.0
        train_corrects = 0
        last_time = time.time()
        model = set_training(model, True)
        pbar_train = tqdm(train_loader, desc=f'Epoch {epoch}', unit='batch')
        for batch_vid, labels in pbar_train:
            mot = batch_vid[:, :2, 1:, :] - batch_vid[:, :2, :-1, :]
            mot, batch_vid, labels = mot.to(device), batch_vid.to(device), labels.to(device)
            outputs = model((batch_vid, mot))
            loss = losser(outputs, labels)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            losses_train += loss.item()
            _, preds = torch.max(outputs, 1)
            train_corrects += (preds == labels.data.argmax(1)).detach().cpu().numpy().mean()
            del batch_vid, labels
            # set memomy
            total_memory, used_memory_before, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
            pbar_train.set_postfix(OrderedDict({'Loss': loss.item(),
                                                'Memory': "%0.2f GB / %0.2f GB" % (used_memory_before / 1024,
                                                                                   total_memory / 1024)}))
        epoch_loss = losses_train / len(train_loader)
        loss_list['train'].append(epoch_loss)
        epoch_acc = train_corrects/len(train_loader)
        acc_list['train'].append(epoch_acc)
        logging.warning('Train: Accuracy: {}, Loss: {}, Time: {}'.format(epoch_acc, epoch_loss,
                                                                         str(datetime.timedelta(seconds=time.time() - last_time))))


        # validation
        last_time = time.time()
        losses_val = 0.0
        val_corrects = 0
        model = set_training(model, False)
        with torch.no_grad():
            for batch_vid, labels in val_loader:
                mot = batch_vid[:, :2, 1:, :] - batch_vid[:, :2, :-1, :]
                mot, batch_vid, labels = mot.to(device), batch_vid.to(device), labels.to(device)
                outputs = model((batch_vid, mot))
                loss = losser(outputs, labels)
                losses_val += loss.item()
                _, preds = torch.max(outputs, 1)
                val_corrects += (preds == labels.data.argmax(1)).detach().cpu().numpy().mean()
                del batch_vid, labels

            epoch_loss = losses_val / len(val_loader)
            loss_list['valid'].append(epoch_loss)
            epoch_acc = val_corrects / len(val_loader)
            acc_list['valid'].append(epoch_acc)
            logging.warning('Validation: Accuracy: {}, Loss: {}, Time: {}'.format(epoch_acc,
                                                                                          epoch_loss,
                                                                                          str(datetime.timedelta(seconds=time.time() - last_time))))
            if best_loss_acc == -1:
                best_loss_acc = epoch_acc
            if best_loss_acc <= epoch_acc:
                best_loss_acc = epoch_acc
                torch.save(model.state_dict(), path_save_model + '/best.pt')
                logging.warning('Saved best model at epoch {}'.format(epoch))

        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(acc_list['train'], label="Train Accuracy")
        plt.plot(acc_list['valid'], label="Val Accuracy")
        plt.xlabel("epoch")
        plt.title("Accuracy")

        # plt.grid()
        plt.legend(loc="best")
        plt.subplot(1, 2, 2)
        plt.plot(loss_list['train'], label="Train Loss")
        plt.plot(loss_list['valid'], label="Val Loss")
        plt.xlabel("epoch")
        plt.title("Loss")
        plt.legend(loc="best")
        # plt.grid()
        fig.savefig(path_save_model + '/result.png', dpi=500)
        plt.close(fig)
        del fig

    return model


def main():
    """
    function: training model
    :return:
    """
    model_trained = train_model(model, losser, optimizer, num_epochs=epochs)
    torch.save(model_trained.state_dict(), path_save_model + '/last.pt')
    logging.warning('Saved last model at {}'.format(path_save_model, "/last.pt"))
    print("Complete !")


if __name__ == '__main__':
    main()
