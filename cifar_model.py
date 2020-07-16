import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from torchvision.transforms import ToTensor
import torch.optim as opt
from torch.optim.lr_scheduler import StepLR


label_dic = {0:'airplane', 1:'automobile',2:'bird',3:'cat',4:'deer',5:'dog',6:'frog',7:'horse',8:'ship', 9:'truck'}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("using gpu")
else:
    print("using cpu")

class cf_model(nn.Module):
    def __init__(self):
        super(cf_model, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2), # output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2), # output: 256 x 4 x 4

            nn.Flatten(), 
            nn.Linear(256*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10))
        
    def forward(self, xb):
        out = self.network(xb)
        return out

def img_show(dataset):
    img_pack = dataset[random.randint(0, len(dataset))]
    img = np.array(img_pack[0])
    label = img_pack[1]
    print(np.shape(img))
    plt.figure()
    plt.imshow(img)
    plt.title(label_dic[label])
    plt.show()

## import the dataset and see some images 
train_set = torchvision.datasets.CIFAR10('.',train = True, transform = ToTensor(),download=True)
# print(len(train_set))
# img_show(train_set) 
val_size = 5000
train_size = len(train_set) - val_size
traindata, valdata = torch.utils.data.random_split(train_set, [train_size, val_size])
print("train size and validation size:",len(traindata), len(valdata))

testdata = torchvision.datasets.CIFAR10('.', train = False, transform = ToTensor())
print("test size", len(testdata))

train_loader = torch.utils.data.DataLoader(traindata,64, shuffle=True, num_workers=1, pin_memory=True)
val_loader = torch.utils.data.DataLoader(valdata, 64, num_workers=1, pin_memory=True)
test_loader = torch.utils.data.DataLoader(testdata, 1000, shuffle = True, num_workers=1, pin_memory=True)


## put model into gpu and set the optimizer
cifar = cf_model().to(device)
optimizer = opt.Adam(cifar.parameters(),lr=1e-3)
scheduler = StepLR(optimizer, 30, 0.7)

## training
def train(model, trainloader, optimizer, epoch, device):
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

for epoch in range(10):
    train(cifar,train_loader,optimizer,epoch,device)
    test(cifar,test_loader, device)
    scheduler.step()

torch.save(cifar.state_dict(), 'cifar_cnn.pt')
