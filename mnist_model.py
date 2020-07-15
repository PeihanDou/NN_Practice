import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import argparse

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,32,3,1)
        self.conv2 = nn.Conv2d(32,64,3,1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216,128,bias=True)
        self.fc2 = nn.Linear(128,10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2)
        x = self.dropout1(x)
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output

def train(args, model, trainloader, optimizer, epoch, device):
    model.train()
    for batch_id, (data, target) in enumerate(trainloader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        pred = model(data)
        loss = F.cross_entropy(pred, target)
        loss.backward()
        optimizer.step()
        if batch_id % 100 == 0:
            print("epoch", epoch, 100 * batch_id / len(trainloader),"%","loss:",loss.item())

def test(model, testloader, device):
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for data, target in testloader:
            data = data.to(device)
            target = target.to(device)
            pred = model(data)
            loss = F.cross_entropy(pred, target)
            test_loss += loss.sum().item()
            pred_label = pred.argmax(dim=1,keepdim=True)
            correct += pred_label.eq(target.view_as(pred_label)).sum().item()
    test_l = test_loss / len(testloader)
    accuracy = 100*correct / len(testloader.dataset)
    print("loss",test_l, "accuracy",accuracy)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--epoch',type=int,default=14)
    parser.add_argument('--lr',type=float,default=1)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--gamma',type=float,default=0.7)
    parser.add_argument('--save_model',action='store_true')
    args = parser.parse_args()


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if (use_cuda) else "cpu")
    print("using",device)
    print("set epoch to ",args.epoch)
    print("using learning rate", args.lr)


    #Dataloader
    mnist_data_train = datasets.MNIST('../data',train=True,download=True,transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ]))

    mnist_data_test = datasets.MNIST('../data',train=False,download=True,transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ]))


    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    trainloader = torch.utils.data.DataLoader(mnist_data_train, batch_size=args.batch_size,shuffle=True, **kwargs)
    testloader = torch.utils.data.DataLoader(mnist_data_test, batch_size=1000,shuffle=True, **kwargs)

    mnist_classifier = Net().to(device)
    optimizer = optim.Adadelta(mnist_classifier.parameters(),lr=args.lr)
    # scheduler = StepLR(optimizer, 30, args.gamma)

    for epoch in range(10):
        train(args,mnist_classifier,trainloader,optimizer,epoch,device)
        test(mnist_classifier,testloader, device)
        # scheduler.step()

    if args.save_model:
        torch.save(mnist_classifier.state_dict(), 'mnist_cnn.pt')

if __name__ == "__main__":
    main()


    
