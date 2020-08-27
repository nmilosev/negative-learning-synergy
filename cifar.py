import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from pytorch_utils import train, test
from loaders import cifar10_loader
from datetime import datetime as dt
from cifar_net import Net
import pickle

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)

LR = 0.001
MOM = 0.9

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loader = cifar10_loader

train_loader = loader(train=True)
test_loader = loader()

model_normal = Net('normal').to(device)
model_normal_tf = Net('normal_tf').to(device)
model_hybrid = Net('hybrid').to(device)
model_synergy = Net('synergy').to(device)
model_synergy_ntf = Net('synergy_ntf').to(device)
model_tr_synergy = Net('tr_synergy').to(device)
model_tr_synergy_tf = Net('tr_synergy_tf').to(device)

optimizer_normal = optim.SGD(filter(lambda p: p.requires_grad,
                             model_normal.parameters()), lr=LR, momentum=MOM)

start_time = time.time()

# ---- Normal net:

for epoch in range(1, 10 + 1):
    train(model_normal, device, train_loader, optimizer_normal,
          epoch, loss_fn=F.cross_entropy)

conv1 = model_normal.conv1
conv2 = model_normal.conv2
conv1.weight.requires_grad = False
conv2.weight.requires_grad = False

# ---- Hybrid net:

model_hybrid.conv1 = conv1
model_hybrid.conv2 = conv2

optimizer_hybrid = optim.SGD(filter(lambda p: p.requires_grad,
                             model_hybrid.parameters()), lr=LR, momentum=MOM)

for epoch in range(1, 10 + 1):
    train(model_hybrid, device, train_loader, optimizer_hybrid,
          epoch, loss_fn=F.cross_entropy)

# ---- Normal TF net:

model_normal_tf.conv1 = conv1
model_normal_tf.conv2 = conv2

optimizer_normal_tf = optim.SGD(filter(lambda p: p.requires_grad,
                                model_normal_tf.parameters()),
                                lr=LR, momentum=MOM)

for epoch in range(1, 10 + 1):
    train(model_normal_tf, device, train_loader, optimizer_normal_tf,
          epoch, loss_fn=F.cross_entropy)

# ---- synergy net (not trained):

model_synergy.conv1 = conv1
model_synergy.conv2 = conv2

model_synergy.fc1 = model_normal.fc1
model_synergy.fc2 = model_normal.fc2
model_synergy.fc3 = model_normal.fc3

model_synergy.fc1n = model_hybrid.fc1
model_synergy.fc2n = model_hybrid.fc2
model_synergy.fc3n = model_hybrid.fc3

# ---- synergy net normal tf (not trained):

model_synergy_ntf.conv1 = conv1
model_synergy_ntf.conv2 = conv2

model_synergy_ntf.fc1 = model_normal_tf.fc1
model_synergy_ntf.fc2 = model_normal_tf.fc2
model_synergy_ntf.fc3 = model_normal_tf.fc3

model_synergy_ntf.fc1n = model_hybrid.fc1
model_synergy_ntf.fc2n = model_hybrid.fc2
model_synergy_ntf.fc3n = model_hybrid.fc3

# ---- Trained synergy TF:

model_tr_synergy_tf.conv1 = conv1
model_tr_synergy_tf.conv2 = conv2

optimizer_tr_synergy_tf = optim.SGD(filter(lambda p: p.requires_grad,
                                    model_tr_synergy_tf.parameters()),
                                    lr=LR, momentum=MOM)

for epoch in range(1, 10 + 1):
    train(model_tr_synergy_tf, device, train_loader,
          optimizer_tr_synergy_tf, epoch, loss_fn=F.cross_entropy)

# ---- Trained synergy:

optimizer_tr_synergy = optim.SGD(filter(lambda p: p.requires_grad,
                                 model_tr_synergy.parameters()),
                                 lr=LR, momentum=MOM)

for epoch in range(1, 10 + 1):
    train(model_tr_synergy, device, train_loader, optimizer_tr_synergy,
          epoch, loss_fn=F.cross_entropy)

# ---- Testing:

models = [model_normal, model_hybrid,
          model_synergy,
          model_tr_synergy_tf, model_tr_synergy]

dataset_names = ['Normal', 'C1', 'C2', 'C3', 'SSK', 'SSKR', 'MSK', 'MSKR']
datasets = []
for _ in dataset_names:
    datasets.append(loader())

from cifar_filters import corner_kernel, single_square_kernel, single_square_kernel_random, multiple_square_kernel, multiple_square_kernel_random
filters_map = {'C1': corner_kernel(0.1),
               'C2': corner_kernel(0.2), 'C3': corner_kernel(0.3),
               'SSK': single_square_kernel(), 'SSKR': single_square_kernel_random(), 'MSK': multiple_square_kernel(), 'MSKR': multiple_square_kernel_random()}

for dataset, name in zip(datasets, dataset_names):
    print('Testing -- ' + name)
    fdir = 'checkpoints'
    for m in models:
        index_store = test(m, device, dataset, loss_fn=F.cross_entropy, dataset_name=name, filter_map=filters_map)
        file_name = f'{fdir}/{m.net_type}-{name}-{dt.now()}.pth'
        with open(file_name, 'wb') as f:
            pickle.dump(index_store, f)

for m in models:
    file_name = f'models/{m.net_type}-{dt.now()}.pth'
    torch.save(m.state_dict(), file_name)

print('--- Total time: %s seconds ---' % (time.time() - start_time))
