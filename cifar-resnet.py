import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from pytorch_utils import train, test
from loaders import cifar10_loader_resnet
from datetime import datetime as dt
from resnet import ResNet18
import pickle

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)

LR = 0.1
MOM = 0.9
DECAY = 5e-3
EPOCHS = 50
MILESTONES = [30, 40]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loader = cifar10_loader_resnet

train_loader = loader(train=True)
test_loader = loader()

model_normal = ResNet18('normal').to(device)
model_normal_tf = ResNet18('normal_tf').to(device)
model_hybrid = ResNet18('hybrid').to(device)
model_synergy = ResNet18('synergy').to(device)
model_synergy_ntf = ResNet18('synergy_ntf').to(device)
model_tr_synergy = ResNet18('tr_synergy').to(device)
model_tr_synergy_tf = ResNet18('tr_synergy_tf').to(device)

optimizer_normal = optim.SGD(filter(lambda p: p.requires_grad,
                             model_normal.parameters()), lr=LR, momentum=MOM, weight_decay=DECAY)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer_normal, milestones=MILESTONES, gamma=0.1)
start_time = time.time()

# ---- Normal net:

for epoch in range(1, EPOCHS + 1):
    train(model_normal, device, train_loader, optimizer_normal,
          epoch, loss_fn=F.cross_entropy)
    scheduler.step()

model_normal.freeze()
test(model_normal, device, test_loader, loss_fn=F.cross_entropy)

# ---- Hybrid net:

model_hybrid.set_conv_layers(model_normal.get_conv_layers())
model_hybrid.freeze()

optimizer_hybrid = optim.SGD(filter(lambda p: p.requires_grad,
                             model_hybrid.parameters()),
                             lr=LR, momentum=MOM, weight_decay=DECAY)

scheduler = optim.lr_scheduler.MultiStepLR(optimizer_hybrid, milestones=MILESTONES, gamma=0.1)

for epoch in range(1, EPOCHS + 1):
    train(model_hybrid, device, train_loader, optimizer_hybrid,
          epoch, loss_fn=F.cross_entropy)
    scheduler.step()

test(model_hybrid, device, test_loader, loss_fn=F.cross_entropy)


# ---- synergy net (not trained):

model_synergy.set_conv_layers(model_normal.get_conv_layers())
model_synergy.set_linear(model_normal.linear)
model_synergy.set_linear_negative(model_hybrid.linear)

test(model_synergy, device, test_loader, loss_fn=F.cross_entropy)

# ---- Normal TF net:

model_normal_tf.set_conv_layers(model_normal.get_conv_layers())

optimizer_normal_tf = optim.SGD(filter(lambda p: p.requires_grad,
                                model_normal_tf.parameters()),
                                lr=LR, momentum=MOM, weight_decay=DECAY)

scheduler = optim.lr_scheduler.MultiStepLR(optimizer_normal_tf, milestones=MILESTONES, gamma=0.1)

for epoch in range(1, EPOCHS + 1):
    train(model_normal_tf, device, train_loader, optimizer_normal_tf,
          epoch, loss_fn=F.cross_entropy)
    scheduler.step()

test(model_normal_tf, device, test_loader, loss_fn=F.cross_entropy)

# ---- synergy net normal tf (not trained):

model_synergy_ntf.set_conv_layers(model_normal_tf.get_conv_layers())

model_synergy_ntf.set_linear(model_normal_tf.linear)
model_synergy_ntf.set_linear_negative(model_hybrid.linear)

test(model_synergy_ntf, device, test_loader, loss_fn=F.cross_entropy)

# ---- Trained synergy TF:

model_tr_synergy_tf.set_conv_layers(model_normal.get_conv_layers())

optimizer_tr_synergy_tf = optim.SGD(filter(lambda p: p.requires_grad,
                                    model_tr_synergy_tf.parameters()),
                                    lr=LR, momentum=MOM, weight_decay=DECAY)

scheduler = optim.lr_scheduler.MultiStepLR(optimizer_tr_synergy_tf, milestones=MILESTONES, gamma=0.1)

for epoch in range(1, EPOCHS + 1):
    train(model_tr_synergy_tf, device, train_loader,
          optimizer_tr_synergy_tf, epoch, loss_fn=F.cross_entropy)
    scheduler.step()

test(model_tr_synergy_tf, device, test_loader, loss_fn=F.cross_entropy)

# ---- Trained synergy:

optimizer_tr_synergy = optim.SGD(filter(lambda p: p.requires_grad,
                                 model_tr_synergy.parameters()),
                                 lr=LR, momentum=MOM, weight_decay=DECAY)

scheduler = optim.lr_scheduler.MultiStepLR(optimizer_tr_synergy, milestones=MILESTONES, gamma=0.1)

for epoch in range(1, EPOCHS + 1):
    train(model_tr_synergy, device, train_loader, optimizer_tr_synergy,
          epoch, loss_fn=F.cross_entropy)
    scheduler.step()

test(model_tr_synergy, device, test_loader, loss_fn=F.cross_entropy)

# ---- Testing:

models = [model_normal, model_normal_tf, model_hybrid,
          model_synergy, model_synergy_ntf,
          model_tr_synergy_tf, model_tr_synergy]

dataset_names = ['Normal:']
datasets = [test_loader]

for dataset, name in zip(datasets, dataset_names):
    print('Testing -- ' + name)
    fdir = 'checkpoints'
    for m in models:
        index_store = test(m, device, dataset, loss_fn=F.cross_entropy)
        file_name = f'{fdir}/{m.net_type}-{name}-{dt.now()}.pth'
        with open(file_name, 'wb') as f:
            pickle.dump(index_store, f)

print('--- Total time: %s seconds ---' % (time.time() - start_time))
