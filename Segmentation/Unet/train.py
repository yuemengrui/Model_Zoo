# *_*coding:utf-8 *_*
# @Author : yuemengrui
# @Time : 2021-06-02 下午2:47
from model import UNet
from dataset import TableSegDataset
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import logging


logging.basicConfig(level=logging.INFO)
torch.autograd.set_detect_anomaly(True)


def load_model(device, checkpoint_path=None):
    model = UNet(num_classes=5)

    if checkpoint_path is not None:
        logging.info("resume checkpoint from {}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)

    model = model.to(device)
    return model


def lr_step(i):
    a = 0.99 - 0.001 * (i // 100)
    return max(1 / (i + 100) + a, 0.0001)


def eval_net(model, loader, device):
    model.eval()
    n_val = len(loader)
    loss = 0

    for img, label in loader:
        img = img.to(device=device, dtype=torch.float32)
        label = label.to(device=device)

        with torch.no_grad():
            pred = model(img)
            pred = pred.permute(0, 2, 3, 1)
            pred = pred.contiguous().view(-1, pred.size()[-1])

        loss += F.cross_entropy(pred, label.contiguous().view(-1)).item()

    return loss / n_val


def train_model(model, device, dataset_dir, batch_size, lr, epochs):
    train_dataset = TableSegDataset(data_dir=dataset_dir, mode='train')
    val_dataset = TableSegDataset(data_dir=dataset_dir, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_step)

    loss_fn = torch.nn.CrossEntropyLoss()

    logging.info('------ train start ------')
    best_acc = 0
    global_iter = 0
    for epoch in range(epochs):
        model.train()
        # epoch_loss = 0

        for (img, label) in train_loader:
            global_iter += 1
            img = img.to(device=device, dtype=torch.float32)
            label = label.to(device=device)

            pred = model(img)
            pred = pred.permute(0, 2, 3, 1)
            pred = pred.contiguous().view(-1, pred.size()[-1])

            loss = loss_fn(pred, label.contiguous().view(-1))

            # epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_iter % 10 == 0:
                logging.info("epoch:{}, global_iter: {}, train_loss:{}, lr:{}".format(epoch, global_iter, str(loss.item())[:6],
                                                                               str(scheduler.get_lr()[0])[:10]))

                scheduler.step(global_iter)
        # epoch_loss /= (iter + 1)

        # print("epoch:{}, train loss:{}, lr:{}".format(epoch, epoch_loss, lr))

        if epoch != 0 and (epoch + 1) % 10 == 0:
            logging.info('------ val start ------')
            val_loss = eval_net(model, val_loader, device)
            logging.info("val_loss: {}".format(val_loss))
            logging.info('------ val end ------')
            if 1 - val_loss > best_acc:
                best_acc = 1 - val_loss

            torch.save(model.state_dict(),
                       './checkpoints/best_acc/best_acc_{}_{}.pkl'.format(str(best_acc)[:6], str(epoch)))

        if epoch != 0 and (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(),
                       './checkpoints/{}_{}.pkl'.format(str(best_acc)[:6], str(epoch)))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', default=2, help="Please give a value for batch_size")
    parser.add_argument('--dataset_dir', default='./table_segmentation_dataset')
    parser.add_argument('--seed', default=1000, help="Please give a value for seed")
    parser.add_argument('--init_lr', default=0.01, help="Please give a value for init_lr")
    parser.add_argument('--epochs', default=500, help="Please give a value for epochs")
    parser.add_argument('--resume_model', default=None, help="Please give a value for epochs")

    args = parser.parse_args()

    args.seed = int(args.seed)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = 'cuda'
    if args.resume_model:
        model = load_model(device, args.resume_model)
    else:
        model = load_model(device)

    train_model(model, device, args.dataset_dir, args.batch_size, args.init_lr, args.epochs)


if __name__ == '__main__':
    main()
