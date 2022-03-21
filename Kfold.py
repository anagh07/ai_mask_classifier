from sklearn.model_selection import KFold
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
kf = KFold(n_splits=10,random_state=None, shuffle=False)
model.train()

for fold, (train_idx,val_idx) in enumerate(kf.split(np.arange(len(training_data)))):
    print("FOLD:",fold)
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