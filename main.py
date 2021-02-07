import os
import random
import numpy as np
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader, RandomSampler
from matplotlib import pyplot as plt
from data import DriveDataSet
import repvgg
import argparse


def accuracy(outputs, targets):
    outputs = torch.log_softmax(outputs, dim=1)
    _, pred = torch.max(outputs.data, 1)
    acc = torch.sum(pred == targets.data)
    return acc


def init_seed():
    seed = 2021
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", "-batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--epochs", "-epochs", type=int, default=20, help="number of epochs")
    parser.add_argument("--num_classes", '-num_classes', type=int, default=15, help="classification nums")
    parser.add_argument('--weights', "-p", type=str, default=None,
                        help=' pretrained model weights pt file')
    parser.add_argument("--learning_rate", '-lr', type=float, default=0.01, help="learning_rate")
    parser.add_argument("--optimizer", '-opt', type=str, default='adam', help="optimizers")
    parser.add_argument("--logs", '-logs', type=str, default='logs/', help="logs dir")
    parser.add_argument("--n_mels", '-n_mels', type=int, default=64, help="Number of filter bank matrices")
    parser.add_argument("--is_split", '-split', type=bool, default=True, help='if True train 1s else 10s')
    args = parser.parse_args()
    return args


def train():
    init_seed()
    parsers = get_args()
    epochs = parsers.epochs
    batch_size = parsers.batch_size
    num_classes = parsers.num_classes
    weight_path = parsers.weights
    lr = parsers.learning_rate
    opt = parsers.optimizer
    logs = parsers.logs
    n_mels = parsers.n_mels

    print('start loading data')
    train_dataset = DriveDataSet(mode='train', num_classes=num_classes, n_mel=n_mels, is_split=True)
    val_dataset = DriveDataSet(mode='val', num_classes=num_classes, n_mel=n_mels, is_split=True)
    train_load = DataLoader(train_dataset, batch_size=batch_size, sampler=RandomSampler(train_dataset))
    val_load = DataLoader(val_dataset, batch_size=batch_size, sampler=RandomSampler(val_dataset))

    print('build model')
    model = repvgg.create_RepVGG(deploy=False, num_classes=num_classes)
    if weight_path:
        print('reloading model')
        model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))

    print('set optimizer and loss')
    if opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.0001, momentum=0.9)
    elif opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise NameError('optimizer choose error! you can reselect it in [sgd, adam]')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=epochs//4, gamma=0.2)
    loss_function = nn.CrossEntropyLoss()

    print('set GPU')
    if torch.cuda.is_available():
        model = model.cuda()
        loss_function = loss_function.cuda()

    print('start training')
    best_corrects = 0
    t_losses, t_accs, v_losses, v_accs = [], [], [], []
    for epoch in range(epochs):
        train_loss = []
        train_acc = []
        model.train()

        for batch, (data, label) in enumerate(train_load):
            data = data.float()
            label = label.view(-1)
            if torch.cuda.is_available():
                data, label = data.cuda(), label.cuda()

            output = model(data)
            loss = loss_function(output, label)
            acc = accuracy(output, label)
            train_loss.append(loss.item())
            train_acc.append(acc.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        lr = optimizer.param_groups[0]['lr']
        t_loss = sum(train_loss) / len(train_loss)
        t_acc = sum(train_acc) / train_dataset.__len__()
        t_losses.append(t_loss)
        t_accs.append(t_acc)
        print('\nEpoch:%d---training end-----lr:%f-----loss:%.6f-----acc:%.6f' % (epoch + 1, lr, t_loss, t_acc))

        with torch.no_grad():
            val_loss = []
            val_acc = []

            model.eval()
            for batch, (data, label) in enumerate(val_load):
                data = data.float()
                label = label.view(-1)
                if torch.cuda.is_available():
                    data, label = data.cuda(), label.cuda()

                output = model(data)
                loss = loss_function(output, label)
                acc = accuracy(output, label)
                val_loss.append(loss.item())
                val_acc.append(acc.item())

        v_loss = sum(val_loss) / len(val_loss)
        v_acc = sum(val_acc) / val_dataset.__len__()
        v_losses.append(v_loss)
        v_accs.append(v_acc)
        print('        validation end-----loss:%.6f-----acc:%.6f' % (v_loss, v_acc))

        if v_acc > best_corrects:
            print(f'##### saving new best model at epoch:%d, acc:%.6f' % (epoch + 1, v_acc))
            torch.save(model.state_dict(), f'{logs}best_model_state_dict_{epoch+1}.pth')
            best_corrects = v_acc
        scheduler.step()

    plt.figure(figsize=(14, 7))
    x = np.linspace(0, epochs-1, epochs, dtype=np.int)
    plt.subplot(1, 2, 1)
    plt.plot(x, t_losses, color='blue', label='train_loss')
    plt.plot(x, v_losses, color='red', label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xticks(x)
    plt.ylim(0, 5)
    plt.grid(alpha=0.5)
    plt.legend(loc=1)
    plt.subplot(1, 2, 2)
    plt.plot(x, t_accs, color='blue', label='train_acc')
    plt.plot(x, v_accs, color='red', label='val_acc')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.xticks(x)
    plt.grid(alpha=0.5)
    plt.legend(loc=4)
    plt.savefig(logs+'training_process.png')
    plt.show()


if __name__ == '__main__':
    train()
