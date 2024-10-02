import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time
import argparse
import numpy as np
import cv2 as cv
import os
import shutil
import monai
from luna16 import *
from sklearn.model_selection import train_test_split


def save_checkpoint(state, is_best, path, prefix, filename='checkpoint.pt'):
    prefix_save = os.path.join(path, prefix)
    name = prefix_save + '_' + filename
    torch.save(state, name)
    if is_best:
        shutil.copyfile(name, prefix_save + '_model_best.pt')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=32)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--no-cuda', type=bool, default=False)
    parser.add_argument('--reload', type=bool, default=False)
    args = parser.parse_args()

    cuda = not args.no_cuda and torch.cuda.is_available()

    if args.reload:

        root_folder = 'f:/MedicalImaging/LUNA16'
        volume_folder = 'lungs_images'
        annotation_folder = 'lungs_segmentation_masks'

        all_luna16_files = luna16.LUNA16.find_files(
            root_folder,
            volume_folder,
            annotation_folder
        )

        train_files, valid_files = train_test_split(all_luna16_files, train_size=.80)

        train_data = luna16.LUNA16(
            root_folder,
            volume_folder,
            annotation_folder,
            train_files,
            max_size=None
        )

        valid_data = luna16.LUNA16(
            root_folder,
            volume_folder,
            annotation_folder,
            valid_files,
            max_size=None
        )

        torch.save(train_data, 'train.dataset')
        # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)

        torch.save(valid_data, 'valid.dataset')
        # valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, **kwargs)

    else:
        train_data = torch.load('train.dataset')
        valid_data = torch.load('valid.dataset')

    train_loader = DataLoader(train_data, batch_size=args.batch_size)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size)

    # find number of classes
    num_classes = max(train_data.classes().keys()) + 1

    # build the model
    model = monai.networks.nets.UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_classes,
        channels=(32, 64, 128),
        strides=(2, 2, 2)
    )

    # load checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("loading checkpoint '{}' ... ".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint)  # ['state_dict'])
            print("... loaded checkpoint")
        else:
            print("no checkpoint found at '{}'".format(args.resume))

    if cuda:
        model = model.cuda()

    os.makedirs(args.save, exist_ok=True)

    loss_function = monai.losses.DiceLoss(to_onehot_y=True, sigmoid=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_valid_loss = np.inf
    for epoch in range(args.num_epochs):  # loop over the dataset multiple times

        train_loss = 0.0
        model.train(True)
        for b, batch in enumerate(train_loader):
            data = batch[1]
            target = batch[2]
            if args.cuda:
                data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()
            output = model(data)
            target = target.unsqueeze(1)

            loss = loss_function(output, target)
            loss.backward()
            train_loss += loss.data.item()
            optimizer.step()

            print("Training, batch:", b + 1, " of ", len(train_loader), ", loss", loss.data.item(),
                  train_loss)

        cv.destroyAllWindows()
        valid_loss = 0.0
        for b, batch in enumerate(valid_loader):
            data = batch[1]
            target = batch[2]
            if args.cuda:
                data, target = data.cuda(), target.cuda()

            output = model(data)
            target = target.unsqueeze(1)
            loss = loss_function(output, target)
            valid_loss += loss.data.item()

        print("Train Epoch: ", epoch + 1, " of ", args.num_epochs, " | loss: ", train_loss,
              " valid loss:", valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'saved_model.pth')
            print("Saving best.")
        cv.destroyAllWindows()


if __name__ == '__main__':
    main()
