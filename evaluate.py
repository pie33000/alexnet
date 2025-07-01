import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/alexnet.pth")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--root", type=str, default="data/test")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    return parser.parse_args()


def evaluate_model(model_path, device, root, batch_size, num_workers):
    model = AlexNet()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    test_loader = load_test_data(batch_size=batch_size, num_workers=num_workers, root=root)

    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            b, ncrops, c, h, w = imgs.shape

            imgs = imgs.view(-1, c, h, w)
            logits = model(imgs)
            logits = logits.view(b, ncrops, -1).mean(1)

            preds = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    print(f"Top-1 accuracy: {100 * correct / total:.2f}%")
    return model


if __name__ == "__main__":
    args = parse_args()
    evaluate_model(args.model_path, args.device, args.root, args.batch_size, args.num_workers)