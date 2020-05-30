import torch
import torch.nn.functional as F


def train(model, device, train_loader, optimizer, epoch, loss_fn=F.nll_loss):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 5000 == 0:
            print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                  .format(model.net_type, epoch, batch_idx * len(data),
                          len(train_loader.dataset),
                          100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, loss_fn=F.nll_loss, dataset_name=None, filter_map=None):
    model.eval()
    test_loss = 0
    correct = 0
    index_store = []
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            if filter_map and dataset_name in filter_map:
              for batch in data:
                for channel in batch:
                  channel.mul_(filter_map[dataset_name].to(device))
            output = model(data)
            test_loss += loss_fn(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            c = pred.eq(target.view_as(pred)).sum().item()
            correct += c
            index_store.append((i, c))

    test_loss /= len(test_loader.dataset)

    print('[{}] Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'
          .format(model.net_type, test_loss, correct, len(test_loader.dataset),
                  100. * correct / len(test_loader.dataset)))

    return index_store
