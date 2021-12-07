# importing the libraries
import os
import cv2
import random

import numpy
import torch
import torch.nn as nn
import torch.utils.data as td
from prettytable import PrettyTable
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, \
    ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

EVALUATE_100_IMG = False  # If running on submitted data (100 sample images) then set to true

categories = ["no_mask", "ffp2_mask", "surgical_mask", "cloth_mask"]
img_size = 60
training_data = []


def create_training_data():
    for category in categories:
        path = "./data/" + category + "/"
        class_num = categories.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)

            new_img = cv2.resize(img_array, (img_size, img_size))
            training_data.append([new_img, class_num])


create_training_data()
random.shuffle(training_data)
for data in training_data:
    data[0] = torch.Tensor(data[0])
    data[0] = data[0].permute(2, 0, 1)
    training_data[training_data.index(data)] = tuple(data)

total = len(training_data)
training_percent = .8
if EVALUATE_100_IMG:
    training_percent = .01
train = training_data[:int(total * training_percent)]
train = training_data[:int(total * training_percent)]
test = training_data[int(total * training_percent):]


def cifar_loader(batch_size, shuffle_test=False):
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                               shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                              shuffle=shuffle_test, pin_memory=True)
    return train_loader, test_loader


batch_size = 32
test_batch_size = 32
input_size = 10800
N = batch_size
D_in = input_size
H = 50
D_out = 4
num_epochs = 10
learning_rate = 0.001
train_loader, _ = cifar_loader(batch_size)
_, test_loader = cifar_loader(test_batch_size)


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


# Initialize model
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# If there is a trained model do not train
has_trained_model = os.path.isfile('cnnsaved.pt')

if has_trained_model:
    model.load_state_dict(torch.load('cnnsaved.pt'), strict=False)
else:
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
print(f'Total training images: {len(train_loader.dataset)}')
print(f'Total test images: {total}')
# class_labels = [0, 1, 2, 3]
class_labels = ['no-mask', 'ffp2', 'surgical', 'cloth']
class_precisions = precision_score(y_true, y_pred, labels=[0, 1, 2, 3],
                                   average=None)
class_recalls = recall_score(y_true, y_pred, labels=[0, 1, 2, 3],
                             average=None)
class_f1s = f1_score(y_true, y_pred, labels=[0, 1, 2, 3],
                     average=None)
report = classification_report(y_true, y_pred, target_names=class_labels)
for index in range(4):
    class_precisions[index] = round(class_precisions[index], 3)
    class_recalls[index] = round(class_recalls[index], 3)
    class_f1s[index] = round(class_f1s[index], 3)
    macro_average_precision = round(precision_score(y_true, y_pred, labels=[0, 1, 2, 3], average='macro'), 3)
    macro_average_recall = round(recall_score(y_true, y_pred, labels=[0, 1, 2, 3], average='macro'), 3)
    macro_average_f1 = round(recall_score(y_true, y_pred, labels=[0, 1, 2, 3], average='macro'), 3)

# Tables
prec_table = PrettyTable()
prec_table.field_names = ['No mask', 'FFP2', 'Surgical', 'Cloth', 'Average']
prec_table.add_row(numpy.append(class_precisions, macro_average_precision))
rec_table = PrettyTable()
rec_table.field_names = ['No mask', 'FFP2', 'Surgical', 'Cloth', 'Average']
rec_table.add_row(numpy.append(class_recalls, macro_average_recall))
f1_table = PrettyTable()
f1_table.field_names = ['No mask', 'FFP2', 'Surgical', 'Cloth', 'Average']
f1_table.add_row(numpy.append(class_f1s, macro_average_f1))
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
print('\n#### Evaluation ####')
print(report)
# print(f'Precision Score: \n{prec_table}\n')
# print(f'Recall Score: \n{rec_table}\n')
# print(f'F1 Score: \n{f1_table}\n')
# print(f'Confusion matrix: \n{cm}')
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
cm_display.plot()
plt.show()

# Save trained model
torch.save(model.state_dict(), 'cnnsaved.pt')
