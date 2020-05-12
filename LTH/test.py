import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.models import AlexNet
import matplotlib.pyplot as plt
from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet
from scheduler import GradualWarmupScheduler

model = fc1.fc1()
optimizer = optim.SGD(params = model.parameters(), lr=0.05)
scheduler_steplr = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.1)  # means multistep will start with 0
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=scheduler_steplr)  # 20K=(idx)56, 35K=70 

plt.figure()
x = list(range(20))
y = []

for epoch in range(20):
    scheduler_warmup.step()
    lr = scheduler.get_lr()
    print(epoch, scheduler.get_lr()[0])
    y.append(scheduler.get_lr()[0])

plt.plot(x,y)