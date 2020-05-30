import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from pytorch_utils import train, test
from loaders import cifar10_loader
from cifar_net import Net
from datetime import datetime as dt
import pickle

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_loader = cifar10_loader(batch_size=1)

def count(model):
    model.eval()
    correct_idx = []
    incorrect_idx = []
    for i, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        if pred == target:
            correct_idx.append(i)
        else:
            incorrect_idx.append(i)
    return correct_idx, incorrect_idx

def show(idx):
    print('label')
    print(test_loader.dataset[idx][1])
    print('normal')
    print(list(model_normal(test_loader.dataset[idx][0].to(device).resize(1, 3, 32, 32)).cpu().detach().numpy()[0]))
    print('hybrid')
    print(list(model_hybrid(test_loader.dataset[idx][0].to(device).resize(1, 3, 32, 32)).cpu().detach().numpy()[0]))
    print('synergy')
    print(list(model_synergy(test_loader.dataset[idx][0].to(device).resize(1, 3, 32, 32)).cpu().detach().numpy()[0]))

if __name__ == '__main__':
    model_normal = Net('normal').to(device)
    model_normal.load_state_dict(torch.load('models/normal-2020-05-24 15:51:56.623195.pth'))
    normal_correct, normal_incorrect = count(model_normal)

    model_hybrid = Net('hybrid').to(device)
    model_hybrid.load_state_dict(torch.load('models/hybrid-2020-05-24 15:51:56.628130.pth'))
    hybrid_correct, hybrid_incorrect = count(model_hybrid)

    model_synergy = Net('synergy').to(device)
    model_synergy.load_state_dict(torch.load('models/synergy-2020-05-24 15:51:56.630193.pth'))
    synergy_correct, synergy_incorrect = count(model_synergy)

    print('correct counts, normal, hybrid, synergy')
    print(len(normal_correct), len(hybrid_correct), len(synergy_correct))

    print('normal correct, hybrid incorrect')
    print(len([x for x in normal_correct if x not in hybrid_correct]))

    print('hybrid correct, normal incorrect')
    print(len([x for x in hybrid_correct if x not in normal_correct]))

    print('both correct')
    print(len([x for x in hybrid_correct if x in normal_correct]))
    
    print('neither correct')
    print(len([x for x in hybrid_incorrect if x in normal_incorrect]))

    print('synergy correct, normal incorrect')
    print(len([x for x in synergy_correct if x not in normal_correct]))

    print('synergy correct, hybrid incorrect')
    print(len([x for x in synergy_correct if x not in hybrid_correct]))
    
    print('synergy correct, both incorrect')
    print(len([x for x in synergy_correct if x not in hybrid_correct and x not in normal_correct]))

    good_cases = [x for x in synergy_correct if x in hybrid_correct and x not in normal_correct]













