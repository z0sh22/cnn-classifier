import argparse
from src.train import train
from src.model import ARCHITECTURES

parser = argparse.ArgumentParser(description="CNN Classifier — CIFAR-10 / MNIST")
parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "mnist"])
parser.add_argument("--arch",    default="all",     choices=["all", *ARCHITECTURES.keys()])
parser.add_argument("--epochs",  default=10,        type=int)
parser.add_argument("--batch",   default=64,        type=int)
args = parser.parse_args()

archs = list(ARCHITECTURES.keys()) if args.arch == "all" else [args.arch]

for arch in archs:
    train(dataset=args.dataset, arch=arch, epochs=args.epochs, batch_size=args.batch)
