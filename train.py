import argparse
import os

import torch
import torch.nn.functional as F
from torch.optim import SGD

from load_data import load_data
from model import AlexNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--init_weights", action="store_true", default=False)
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--root", type=str, default=None)

    return parser.parse_args()


def print_args(args):
    print("-" * 50)
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("-" * 50)


def train_model(
    model, train_loader, val_loader, optimizer, epochs=100, device="cpu", save_dir="models"
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(epochs):
        for i, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            model.train()
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)
            if i % 100 == 0:
                print(f"Epoch {epoch + 1}, Step {i + 1}, Loss: {loss.item()}")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        total, correct = 0, 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                b, ncrops, c, h, w = imgs.shape

                imgs = imgs.view(-1, c, h, w)
                logits = model(imgs)
                logits = logits.view(b, ncrops, -1).mean(1)

                preds = logits.argmax(dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        print(f"Epoch {epoch + 1}, Top-1 accuracy: {100 * correct / total:.2f}%")
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"{save_dir}/model_{epoch}.pth")


if __name__ == "__main__":
    args = parse_args()
    if args.root is None:
        raise ValueError("Root for ImageNet Dataset directory is not set")
    print_args(args)

    model = AlexNet(init_weights=args.init_weights).to(args.device)
    train_loader, val_loader = load_data(
        batch_size=args.batch_size, num_workers=args.num_workers, root=args.root
    )

    optimizer = SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        epochs=args.epochs,
        device=args.device,
        save_dir=args.save_dir,
    )
