import os
import argparse
import copy

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import config
import model_dispatcher

def run():
    # create transformations for images
    transformation = {
        "train": 
        transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.RandomRotation(360),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.RandomAffine(270),
                            transforms.ToTensor(),
                            transforms.RandomErasing(p=0.4),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        "test": 
        transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    }

    # create train dataset
    train_dataset = torchvision.datasets.ImageFolder(
        config.TRAIN_DATA,
        transformation["train"]
    )

    # create test dataset
    test_dataset = torchvision.datasets.ImageFolder(
        config.TEST_DATA,
        transformation["test"]
    )

    # create train dataloader
    train_dataloader = DataLoader(
        dataset = train_dataset,
        batch_size = config.BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    # create test dataloader
    test_dataloader = DataLoader(
        dataset = test_dataset,
        batch_size = config.BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    # initialize model
    model = model_dispatcher.CatDogNN(config.MODEL_IN_FEATURES, config.MODEL_OUT_FEATURES)

    # load parameters of the model
    model.load_state_dict(torch.load(config.BEST_MODEL))

    # send model to cuda (if available) for faster training
    model = model.to(config.TRAIN_DEVICE)

    # specify parameters for training
    epochs = config.EPOCHS
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # features to help follow training
    train_losses = []
    test_losses = []
    best_acc = -100

    # run the train loop
    for epoch in range(epochs):
        train_loss = 0
        model.train()
        for data in train_dataloader:
            inputs = data[0]
            targets = data[1]
            
            inputs = inputs.to(config.TRAIN_DEVICE)
            targets = targets.to(config.TRAIN_DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*inputs.size(0)
        
        test_loss = 0
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_dataloader:
                inputs = data[0]
                targets = data[1]
                inputs = inputs.to(config.TRAIN_DEVICE)
                targets = targets.to(config.TRAIN_DEVICE)
                outputs = model(inputs)

                tloss = criterion(outputs, targets)
                test_loss += tloss.item()*inputs.size(0)
                outputs = torch.max(outputs, dim=1)[1]
                correct += (targets == outputs).sum()
                total += len(targets)

        train_loss = train_loss / len(train_dataloader.sampler)
        train_losses.append(train_loss)
        test_loss = test_loss / len(test_dataloader.sampler)
        test_losses.append(test_loss)
        acc = correct / float(total)

        if acc > best_acc:
            best_model = copy.deepcopy(model)
            best_acc = acc
            print("Model improved!!!")

        print(f"Epoch: {epoch}. Accuracy: {round(acc.detach().cpu().numpy().tolist(), 4)}. Best accuracy: {round(best_acc.detach().cpu().numpy().tolist(), 4)} Train Loss: {round(train_loss, 4)}. Test Loss: {round(test_loss, 4)}")

    # save the best model parameters after training
    torch.save(best_model.state_dict(), os.path.join(config.MODEL_OUTPUT, "best_model.pt"))
    print("The best model saved!!!")


if __name__ == "__main__":
    run()
