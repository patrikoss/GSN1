import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torchvision
import argparse

def batch_norm(channels, device, bf=0.9):
    bn_params = Variable(torch.ones(channels, 2), requires_grad=True).to(device)

    def run_batch_norm(x):
        """
        x is of shape B,C,H,W
        """
        epsilon = 0.000001

        B,C,H,W = x.shape
        m = B * H * W
        mean = 1/m * torch.sum(x, dim=[0,2,3], keepdim=True)

        x_ = x - mean
        variance = 1/m * torch.sum(x_ ** 2, dim=[0,2,3], keepdim=True)

        x_bn = (x - mean)/torch.sqrt(variance+epsilon)
        y_bn = bn_params[:,0].unsqueeze(0).unsqueeze(2).unsqueeze(3) * x_bn + \
               bn_params[:,1].unsqueeze(0).unsqueeze(2).unsqueeze(3)

        return y_bn

    return run_batch_norm

# function to extract and save intermediate grad
def set_grad(grads_acc):
    def hook(grad):
        grads_acc.append(grad)
    return hook

class Net(nn.Module):
    def __init__(self, device):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=(1,1))
        #self.batch_norm1 = nn.BatchNorm2d(num_features=3)
        self.batch_norm1 = batch_norm(channels=3, device=device)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=(1,1))

        self.batch_norm2 = batch_norm(channels=8, device=device)
        #self.batch_norm2 = nn.BatchNorm2d(num_features=8)

        self.fc1 = nn.Linear(in_features=16 * 25 * 25, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=96)

        self.grads = []

    def forward(self, x):
        x = self.batch_norm1(x)
        # save intermediate grad for visualization purposes
        x.register_hook(set_grad(self.grads))
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.batch_norm2(x)
        # save intermediate grad for visualization purposes
        x.register_hook(set_grad(self.grads))
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        # save intermediate grad for visualization purposes
        x.register_hook(set_grad(self.grads))

        x = x.view(-1, 16 * 25 * 25)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x


def load_dataset(image_dataset_path, batch_size=64):
    train_dataset = torchvision.datasets.ImageFolder(
        root=image_dataset_path,
        transform=torchvision.transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )
    print(train_dataset.classes)
    return train_loader, train_dataset.classes


def train(model, device, train_loader, optimizer, epoch, log_interval=100):
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

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            import ipdb; ipdb.set_trace()
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='PyTorch CNN for fruit classification')
    parser.add_argument('--no_train', action='store_true', default=False)
    parser.add_argument('--save_weights', help='path to where to save the weights', required=False)
    parser.add_argument('--load_weights', help='path from where to load the weights', required=False)
    parser.add_argument('--use_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--train_images', help='path to train images folder')
    parser.add_argument('--test_images', help='path to test images folder')
    args = parser.parse_args()

    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = Net(device).to(device)
    test_loader, classes = load_dataset(args.test_images)
    #import ipdb; ipdb.set_trace()

    if args.no_train:
        model.load_state_dict(torch.load(args.load_weights))
        model.eval()
        test(model, device, test_loader)
    else:
        train_loader, _ = load_dataset(args.train_images)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        for epoch in range(1, 11):
            train(model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader)
        if args.save_weights:
            torch.save(model.state_dict(), args.save_weights)


