# importing the libraries
import os
import cv2
import random

import numpy
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as td
from prettytable import PrettyTable
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, \
    ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

EVALUATE_100_IMG = False  # If running on submitted data (100 sample images) then set to true

categories = ["no_mask", "ffp2_mask", "surgical_mask", "cloth_mask"]
img_size = 60
training_data = []
batch_size = 32
test_batch_size = 32
input_size = 10800
N = batch_size
D_in = input_size
H = 50
D_out = 4
num_epochs = 3
learning_rate = 0.001
training_dataset_path = "./data_p2_fixed/train/"
training_asian_bias_fix = "./data_p2_fixed/train_race_bias/"
mixed_testing_dataset_path = "./data_p2_fixed/test/mixed/"
male_data_set_path = "./data_p2_fixed/test/gender_split/male/"
female_data_set_path = "./data_p2_fixed/test/gender_split/female/"
asian_data_set_path = "./data_p2_fixed/test/race_split/asian/"
caucasian_data_set_path = "./data_p2_fixed/test/race_split/caucasian/"
other_data_set_path = "./data_p2_fixed/test/race_split/other/"
africanamerican_data_set_path = "./data_p2_fixed/test/race_split/african-american/"


def create_training_data(dataset_type="./data/", training_percentage=.8):
    for category in categories:
        path = dataset_type + category + "/"
        class_num = categories.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
            new_img = cv2.resize(img_array, (img_size, img_size))
            training_data.append([new_img, class_num])

    random.shuffle(training_data)
    for data in training_data:
        data[0] = torch.Tensor(data[0])
        data[0] = data[0].permute(2, 0, 1)
        training_data[training_data.index(data)] = tuple(data)

    total = len(training_data)
    if EVALUATE_100_IMG:
        training_percentage = .01
    train = training_data[:int(total * training_percentage)]
    train = training_data[:int(total * training_percentage)]
    test = training_data[int(total * training_percentage):]
    return [train, test]


def train_loader_run(batch_size, train, test, shuffle_test=False):
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                               shuffle=True, pin_memory=True)
    return train_loader


def test_loader_run(batch_size, train, test, shuffle_test=False):
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                              shuffle=shuffle_test, pin_memory=True)
    return test_loader


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(14400, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, D_out)
        )

    def forward(self, x):
        # conv layers
        x = self.conv_layer(x)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = self.fc_layer(x)
        return x


def train_model(model, train_loader, criterion, optimizer):
    print(f'Training data set size: {len(train_loader.dataset)}')
    model.train()
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())
            # Backprop and optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Train accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)
            if (i + 1) % 15 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100))


def train_kf_model(kf, training_data, model, criterion, optimizer):
    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(training_data)))):
        print("FOLD:", fold)
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        test_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
        train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, sampler=train_sampler)
        test_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, sampler=test_sampler)
        model.train()
        total_step = len(train_loader)
        loss_list = []
        acc_list = []
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                # Forward passdata/
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss_list.append(loss.item())
                # Backprop and optimisation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Train accuracy
                total = labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()
                acc_list.append(correct / total)
                if (i + 1) % 15 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                          .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                                  (correct / total) * 100))
        run_test(model, test_loader, len(train_loader.dataset))


def load_existing_model(model, filename='cnnsaved.pt'):
    # If there is a trained model do not train
    has_trained_model = os.path.isfile(filename)

    if has_trained_model:
        model.load_state_dict(torch.load(filename), strict=False)
    else:
        print("Failed to load existing model")


def print_pred_image(model, test_loader):
    for images, labels in test_loader:
        outputs = model(images)
        pred_values, predicted = torch.max(outputs.data, 1)
        rows = 2
        columns = 2
        fig = plt.figure(figsize=(10, 7))
        fig.add_subplot(rows, columns, 1)

        # showing image
        plt.imshow(images[0].numpy().transpose(1, 2, 0).astype(np.uint8))
        plt.axis('off')
        plt.title("Predicted:" + str(predicted[0].item()) + "   Original:" + str(labels[0].item()))

        # Adds a subplot at the 2nd position
        fig.add_subplot(rows, columns, 2)

        # showing image
        plt.imshow(images[1].numpy().transpose(1, 2, 0).astype(np.uint8))
        plt.axis('off')
        plt.title("Predicted " + str(predicted[1].item()) + "   Original:" + str(labels[1].item()))

        # Adds a subplot at the 3rd position
        fig.add_subplot(rows, columns, 3)

        # showing image
        plt.imshow(images[2].numpy().transpose(1, 2, 0).astype(np.uint8))
        plt.axis('off')
        plt.title("Predicted: " + str(predicted[2].item()) + "   Original:" + str(labels[2].item()))

        # Adds a subplot at the 4th position
        fig.add_subplot(rows, columns, 4)

        # showing image
        plt.imshow(images[3].numpy().transpose(1, 2, 0).astype(np.uint8))
        plt.axis('off')
        plt.title("Predicted: " + str(predicted[3].item()) + "   Original:" + str(labels[3].item()))
        plt.show()

        break


def run_test(model, test_loader, train_size=0):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        y_true = torch.tensor([])
        y_pred = torch.tensor([])
        for images, labels in test_loader:
            outputs = model(images)
            pred_values, predicted = torch.max(outputs.data, 1)
            y_true = torch.concat((y_true, labels), 0)
            y_pred = torch.concat((y_pred, predicted), 0)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predicted_size = predicted.size()

    # Evaluation using scikit learn metrics
    print('\n#### Dataset ####')
    print(f'Total training images: {train_size}')
    print(f'Total test images: {total}')
    class_labels = ['no-mask', 'ffp2', 'surgical', 'cloth']
    report = classification_report(y_true, y_pred, target_names=class_labels)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    print('\n#### Evaluation ####')
    print(report)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    cm_display.plot()
    plt.show()


def main():
    # [train, test] = create_training_data(training_dataset_path, 1)  # Original dataset
    # [train, test] = create_training_data(training_asian_bias_fix, 1)  # Asian bias fix training set
    [train, test] = create_training_data(mixed_testing_dataset_path, 0)  # Testing dataset
    # [train, test] = create_training_data(male_data_set_path, 0)  # Gender based split, male dataset
    # [train, test] = create_training_data(female_data_set_path, 0)  # Gender based split, female dataset
    # [train, test] = create_training_data(caucasian_data_set_path, 0)  # Race based split, caucasian dataset
    # [train, test] = create_training_data(asian_data_set_path, 0)  # Race based split, asian dataset
    # [train, test] = create_training_data(africanamerican_data_set_path, 0)  # Race based split, african-american dataset
    # [train, test] = create_training_data(other_data_set_path, 0)  # Race based split, other dataset
    # train_loader = train_loader_run(batch_size, train, test)
    test_loader = test_loader_run(test_batch_size, train, test)

    # Initialize model
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # KF
    kf = KFold(n_splits=10, random_state=None, shuffle=False)
    train_kf_model(kf, training_data, model, criterion, optimizer)

    # Train model
    load_existing_model(model, 'cnn-race-bias-saved.pt')
    # train_model(model, train_loader, criterion, optimizer)

    # Evaluation on test data
    run_test(model, test_loader)

    print_pred_image(model, test_loader)

    # Save trained model
    # torch.save(model.state_dict(), 'cnn-kfold-bias-fix.pt')


if __name__ == "__main__":
    main()
