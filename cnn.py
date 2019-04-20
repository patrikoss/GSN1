import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=(1,1))

        self.fc1 = nn.Linear(in_features=16 * 25 * 25, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=96)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 25 * 25)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_dataset():
    data_path = '/home/patryk/studia/gsn/fruits-360/Training'
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True
    )
    return train_loader


def train(model, device, train_loader, optimizer, epoch, log_interval=10):
    model.train()
    correct_predictions, all_predictions = 0, 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()


        # calculate accuracy
        _, predicted = torch.max(output, 1)
        correct = (predicted == target)
        correct_predictions += torch.sum(correct).item()
        all_predictions += predicted.shape[0]

        if batch_idx % log_interval == 0:
            print("Train accuracy: ", correct_predictions / all_predictions)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))



if __name__=='__main__':

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    train_loader = load_dataset()
    model = Net().to(device)
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters())
    for epoch in range(1, 100):
        train(model, device, train_loader, optimizer, epoch)