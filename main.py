import time
import argparse

import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models.model_1 import Model1
from utils import misc, data_loader
from scripts.generate_results import get_metrics

def get_arguments():

    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')

    parser.add_argument('--train', action='store_true', help='Trains the model if provided')
    parser.add_argument('--test', action='store_true', help='Evaluates the model if provided')
    parser.add_argument('--dataset', type=str, default='s3dis', help='Experiment dataset')

    parser.add_argument('--root', type=str, default=None, help='Path to data')
    parser.add_argument('--output', type=str, default='output', help='Output folder')
    parser.add_argument('--model_path', type=str, help='Pretrained model path')

    parser.add_argument('--batch_size', type=int, default=1, help='Size of batch')
    parser.add_argument('--epochs', type=int, default=100, help='Number of episode to train')
    parser.add_argument('--use_adam', action='store_true', help='Uses Adam optimizer if provided')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Learning rate decay rate')
    parser.add_argument('--decay_step', type=float, default=20, help='Learning rate decay step')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay rate (L2 regularization)')
    parser.add_argument('--no_augmentation', action='store_true', help='Disables training augmentation if provided')

    parser.add_argument('--cuda', type=int, default=0, help='CUDA id. -1 for CPU')
    parser.add_argument('--workers', type=int, default=0, help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--print_summary', type=bool,  default=True, help='Whether to print epoch summary')

    return parser.parse_args()

def main():

    args = get_arguments()

    # Seed RNG
    misc.seed(args.seed)
    
    train_loader, valid_loader, test_loader = data_loader.get_loaders(args)
    model = Model1()

    if args.train:
        train(model, train_loader, valid_loader, args)

    if args.test:
        test(model, test_loader, args)


def run_one_epoch(model, tqdm_iterator, mode, get_locals=False, optimizer=None, loss_update_interval=1000):
    """Definition of one epoch procedure.
    """
    if mode == "train":
        assert optimizer is not None
        model.train()
    else:
        model.eval()

    summary = {"losses": [], "logits": []}
    device = next(model.parameters()).device

    for i, (X_cpu, y_cpu) in enumerate(tqdm_iterator):
        X, y = X_cpu.to(device), y_cpu.to(device)

        # Reset gradients
        if optimizer:
            optimizer.zero_grad()

        # Forward pass
        logits = model(X)
        loss = model.get_loss(logits, y)
        summary["losses"] += [loss.item()]

        # Backward pass
        if optimizer:
            loss.backward()
            optimizer.step()

            # Display
            if loss_update_interval > 0 and i%loss_update_interval == 0:
                tqdm_iterator.set_description("Loss: %.3f" % (np.mean(summary["losses"])))

        # Needed for prediction
        if get_locals:
            summary["logits"] += [logits.cpu().detach().numpy()]
            summary["labels"] += [y_cpu.numpy()]

    return summary

def test(model, test_loader, args):
    """Testing function
    """

    # Set device
    assert args.cuda < 0 or torch.cuda.is_available()
    device_tag = "cpu" if args.cuda == -1 else "cuda:%d"%args.cuda
    device = torch.device(device_tag)

    # Set model and loss
    model = model.to(device)

    # Get current state
    if args.model_path is not None:
        state = torch.load(args.model_path)
        model.load_state_dict(state["model_state_dict"])
        print("Loaded pre-trained model from %s"%args.model_path)

    # Define the test procedure
    def test_one_epoch():
        iterations = tqdm(test_loader, unit='batch', desc="Testing")
        summary = run_one_epoch(model, iterations, "test", get_locals=True, loss_update_interval=-1)

        preds = summary["logits"].argmax(axis=-2)
        summary = get_segmentation_metrics(summary["labels"], preds)
        summary["Loss/test"] = np.mean(summary["losses"])

        return summary

    summary = test_one_epoch()
    summary_string = misc.json.dumps(summary, indent=2)
    print("Testing summary:\n%s" % (summary_string))


def train(model, train_loader, valid_loader, args):
    """Trainer function
    """

    # Set device
    assert args.cuda < 0 or torch.cuda.is_available()
    device_tag = "cpu" if args.cuda == -1 else "cuda:%d"%args.cuda
    device = torch.device(device_tag)

    # Set model and label weights
    model = model.to(device)
    model.labelweights = torch.tensor(train_loader.dataset.labelweights, device=device, requires_grad=False)

    # Set optimizer (default SGD with momentum)
    if args.use_adam:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)

    # Save model and trainer files
    # Get current state
    module_file = misc.sys.modules[model.__class__.__module__].__file__
    state = misc.persistence(args, module_file=module_file, main_file=__file__)
    init_epoch = state["epoch"]

    if state["model_state_dict"]:
        print("Loading pre-trained model from %s"%args.model_path)
        model.load_state_dict(state["model_state_dict"])

    if state["optimizer_state_dict"]:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    # Set learning rate scheduler
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                  milestones=np.arange(
                                                      args.decay_step,
                                                      args.epochs,
                                                      args.decay_step).tolist(),
                                                  gamma=args.lr_decay,
                                                  last_epoch=init_epoch-1)

    # Define train and validation procedures
    def train_one_epoch():
        iterations = tqdm(train_loader, unit='batch', leave=False)
        ep_sum = run_one_epoch(model, iterations, "train", optimizer=optimizer, loss_update_interval=1)

        summary = {"Loss/train": np.mean(ep_sum["losses"])}
        return summary

    def eval_one_epoch():
        iterations = tqdm(valid_loader, unit='batch', leave=False, desc="Validation")
        ep_sum = run_one_epoch(model, iterations, "test", get_locals=True, loss_update_interval=-1)

        preds = [l[0].argmax(axis=0) for l in ep_sum["logits"]]
        labels = [l[0] for l in ep_sum["labels"]]
        summary = get_metrics(labels, preds)
        summary["Loss/validation"] = float(np.mean(ep_sum["losses"]))
        return summary

    # Train for multiple epochs
    tensorboard = SummaryWriter(log_dir=misc.join_path(args.output, "logs"))
    tqdm_epochs = tqdm(range(init_epoch, args.epochs), total=args.epochs, initial=init_epoch, unit='epoch', desc="Progress")
    for e in tqdm_epochs:
        train_summary = train_one_epoch()
        valid_summary = eval_one_epoch()
        summary = {**train_summary, **valid_summary}
        summary["LearningRate"] = lr_scheduler.get_lr()[-1]

        if args.print_summary:
            tqdm_epochs.clear()
            print("Epoch %d summary:\n%s\n" % (e+1, misc.json.dumps((summary), indent=2)))

        # Update learning rate and save checkpoint
        lr_scheduler.step()
        misc.save_checkpoint(args.output, {
            "epoch": e+1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": summary["Loss/validation"],
            "summary": summary
        })

        # Write summary
        for name, val in summary.items():
            tensorboard.add_scalar(name, val, global_step=e+1)

if __name__ == "__main__":
    main()
