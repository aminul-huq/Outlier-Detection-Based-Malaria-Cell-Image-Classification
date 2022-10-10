import torch,argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader,random_split,Dataset
import torch.optim as optim
from networks import *
from training_config import *
import random
import pickle 
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Malaria DNN')

    # model hyper-parameter variables
    parser.add_argument('--epochs', default=25, metavar='epochs', type=int, help='Number of epochs')
    parser.add_argument('--net', default=1, metavar='net', type=int, help='0 WRN 1 RN50 2 RN152 others VGG16')
    parser.add_argument('--txt', default='wrn', metavar='txt', type=str, help='string name')
    parser.add_argument('--pickle_name', default='wrn', metavar='pickle_name', type=str, help='pickle results name')
    parser.add_argument('--gpu', default=0, metavar='gpu', type=int, help='0 for cpu 1 for gpu')
    parser.add_argument('--batch_size', default=32, metavar='batch_size', type=int, help='batch_size')
    
    args = parser.parse_args()

    
    NET = args.net
    FN = args.txt     
    PICKLE_NAME = args.pickle_name       
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    
    
with open('Dataset_Full.pickle', "rb") as fp:   # Unpickling
    x,y = pickle.load(fp)
print("Dataset loaded")

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(32),
    ])

class NewDataset(Dataset):
    
    def __init__(self,data,labels,transform=None):
        self.data = data
        self.label = labels
        self.transform = transform
    def __len__(self):
        return len(self.data)    
    def __getitem__(self,idx):
        image = self.data[idx]
        label = self.label[idx]
        return self.transform(image), label

new_trainset = NewDataset(x,y,transform_train)
lengths = [int(len(new_trainset)*0.8), int(len(new_trainset)*0.2)+1]
trainset,testset = random_split(new_trainset,lengths)

train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)
test_loader2 = DataLoader(testset, batch_size=1, shuffle=True)

if args.gpu == 0:
    print('cpu')
    device = torch.device('cpu')
else:
    print('gpu')
    device = torch.device('cuda')
    
if NET == 0:
    print("WRN")
    net=Wide_ResNet(28, 10, 0.3, num_classes = 2).to(device)
elif NET == 1:
    print("ResNet 50")
    net=ResNet(50, num_classes = 2).to(device)#18, 34, 50, 101, 152
elif NET == 2:
    print("ResNet 101")
    net=ResNet(101, num_classes = 2).to(device)#18, 34, 50, 101, 152
elif NET == 3:
    print("DenseNet")
    net = densenet_BC_cifar(190, 40, num_classes=2).to(device)
elif NET == 4:
    print("LeNet")
    net=LeNet(num_classes = 2).to(device)#18, 34, 50, 101, 152
elif NET == 5:
    print("MobileNet V2")
    net = MobileNetV2(num_classes=2).to(device)
else:
    print("VGG16")
    net = VGG('VGG16',num_classes=10).to(device)#  #11, 13, 16, 19


criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(net.parameters(),lr=0.01)

n_epochs = EPOCHS
train_loss,test_loss, train_acc, test_acc = [],[],[],[]



for i in range(n_epochs):
    a,b = train(net, train_loader, criterion, optim, i,device)
    c,d = test(net, test_loader, criterion, optim, FN, i,device)
    train_loss.append(a), test_loss.append(c),train_acc.append(b), test_acc.append(d)

y, y_pred = test2(net, test_loader2, criterion, optim, i, device)


with open('Results/' + PICKLE_NAME + '.pkl', 'wb') as f:
    pickle.dump([train_loss,test_loss,train_acc,test_acc,y,y_pred,i], f)
































