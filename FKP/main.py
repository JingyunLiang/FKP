"""
Train FKP
"""

import torch
from torchvision.utils import save_image
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import math
import argparse
import pprint
import numpy as np

from dataloader import KernelFolder
from network import KernelPrior


parser = argparse.ArgumentParser()

# action
parser.add_argument('--train', action='store_true', default=False, help='Train a flow.')
parser.add_argument('--evaluate', action='store_true', default=False, help='Evaluate a flow.')
parser.add_argument('--restore_file', type=str, default='', help='Path to model to restore.')
parser.add_argument('--generate', action='store_true', default=True, help='Generate samples from a model.')
parser.add_argument('--val_save_path', default='../data/datasets/Kernel_validation_set',
                    help='Where to save validation set')
parser.add_argument('--output_dir', default='../data/log_FKP/FKP')
parser.add_argument('--results_file', default='results.txt', help='Filename where to store settings and test results.')
parser.add_argument('--no_cuda', action='store_true', help='Do not use cuda.')
# data
parser.add_argument('--sf', type=int, default=2, help='Scale factor')
parser.add_argument('--kernel_size', type=int, default=11,
                    help='Kernel size. 11, 15, 19 for x2, x3, x4; to be overwritten automatically')
parser.add_argument('--flip_var_order', action='store_true', default=False, help='')
parser.add_argument('--seed', type=int, default=0, help='Random seed to use.')
# model
parser.add_argument('--n_blocks', type=int, default=5, help='Number of blocks to stack in a model.')
parser.add_argument('--n_components', type=int, default=1,
                    help='Number of Gaussian clusters for mixture of gaussians models.')
parser.add_argument('--hidden_size', type=int, default=15,
                    help='Hidden layer size. 15, 20, 25 for x2, x3, x4; to be overwritten automatically')
parser.add_argument('--n_hidden', type=int, default=1, help='Number of hidden layers.')
parser.add_argument('--conditional', default=False, action='store_true', help='Whether to use a conditional model.')
parser.add_argument('--no_batch_norm', action='store_true')
# training params
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--start_epoch', default=0,
                    help='Starting epoch (for logging; to be overwritten when restoring file.')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
parser.add_argument('--log_interval', type=int, default=500, help='How often to show loss statistics and save samples.')


# --------------------
# Dataloaders
# --------------------

def fetch_dataloaders(val_save_path, kernel_size, scale_factor, batch_size, device):
    train_dataset = KernelFolder(val_save_path, train=True, kernel_size=kernel_size, scale_factor=scale_factor)
    test_dataset = KernelFolder(val_save_path, train=False, kernel_size=kernel_size, scale_factor=scale_factor)

    input_dims = (1, test_dataset.kernel_size, test_dataset.kernel_size)
    label_size = None
    alpha = test_dataset.alpha
    normalization = test_dataset.normalization

    # keep input dims, input size and label size
    train_dataset.input_dims = input_dims
    train_dataset.input_size = int(np.prod(input_dims))
    train_dataset.label_size = label_size
    train_dataset.alpha = alpha
    train_dataset.normalization = normalization

    test_dataset.input_dims = input_dims
    test_dataset.input_size = int(np.prod(input_dims))
    test_dataset.label_size = label_size
    test_dataset.alpha = alpha
    test_dataset.normalization = normalization

    # construct dataloaders
    kwargs = {'num_workers': 8, 'pin_memory': True} if device.type is 'cuda' else {}

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader


# --------------------
# Train and evaluate
# --------------------

def train(model, dataloader, optimizer, epoch, args):
    for i, data in enumerate(dataloader):
        model.train()

        # check if labeled dataset
        if len(data) == 1:
            x, y = data[0], None
        else:
            x, y = data
            y = y.to(args.device)
        x = x.view(x.shape[0], -1).to(args.device)

        loss = - model.log_prob(x, y if args.cond_label_size else None)[0].mean(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            print('epoch {:3d} / {}, step {:4d} / {}; lr: {:.6f}; loss {:.4f}'.format(
                epoch, args.start_epoch + args.n_epochs, i, len(dataloader), optimizer.param_groups[0]['lr'],
                loss.item()))


@torch.no_grad()
def evaluate(model, dataloader, epoch, args):
    model.eval()

    # conditional model
    if args.cond_label_size is not None:
        logprior = torch.tensor(1 / args.cond_label_size).log().to(args.device)  # discrete uniform distribution
        loglike = [[] for _ in range(args.cond_label_size)]

        # test classes one by one
        for i in range(args.cond_label_size):
            # make one-hot labels for class i
            labels = torch.zeros(args.batch_size, args.cond_label_size).to(args.device)
            labels[:, i] = 1

            for x, y in dataloader:
                x = x.view(x.shape[0], -1).to(args.device)
                loglike[i].append(model.log_prob(x, labels)[0])

            loglike[i] = torch.cat(loglike[i], dim=0)  # cat along data dim under this label
        loglike = torch.stack(loglike, dim=1)  # cat all data along label dim

        # log p(x) = log ∑_y p(x,y) = log ∑_y p(x|y)p(y)
        # assume uniform prior      = log p(y) ∑_y p(x|y) = log p(y) + log ∑_y p(x|y)
        logprobs = logprior + loglike.logsumexp(dim=1)

    # unconditional model
    else:
        logprobs = []
        for data in dataloader:
            x = data[0].view(data[0].shape[0], -1).to(args.device)
            logprobs.append(model.log_prob(x)[0])
        logprobs = torch.cat(logprobs, dim=0).to(args.device)

    print(args.output_dir)
    logprob_mean, logprob_std = logprobs.mean(0), 2 * logprobs.var(0).sqrt() / math.sqrt(len(dataloader.dataset))
    output = 'Evaluate ' + (epoch != None) * '(epoch {}): '.format(epoch) + '-logp(x) = {:.3f} +/- {:.3f}'.format(
        -logprob_mean, logprob_std)
    print(output)
    print(output, file=open(args.results_file, 'a'))
    return logprob_mean, logprob_std


@torch.no_grad()
def generate(model, args, step=None, n_row=10):
    model.eval()

    # conditional model
    if args.cond_label_size:
        samples = []
        labels = torch.eye(args.cond_label_size).to(args.device)

        for i in range(args.cond_label_size):
            # sample model base distribution and run through inverse model to sample data space
            u = model.base_dist.sample((n_row, args.n_components)).squeeze()
            labels_i = labels[i].expand(n_row, -1)
            sample, _ = model.inverse(u, labels_i)
            log_probs = model.log_prob(sample, labels_i)[0].sort(0)[1].flip(
                0)  # sort by log_prob; take argsort idxs; flip high to low
            samples.append(sample[log_probs])

        samples = torch.cat(samples, dim=0)

    # unconditional model
    else:
        # for a random Gaussian vector, its l2norm is always close to 1.
        # therefore, in optimization, we can constrain the optimization space to be on the sphere with radius of 1
        u = model.base_dist.sample((n_row ** 2, args.n_components)).squeeze()
        samples, _ = model.inverse(u)
        log_probs = model.log_prob(samples)[0].sort(0)[1].flip(
            0)  # sort by log_prob; take argsort idxs; flip high to low
        samples = samples[log_probs]

    # convert and save images
    samples = model.post_process(samples)
    filename = 'generated_samples' + (step != None) * '_epoch_{}'.format(step) + '.png'

    # rescale the maximum value to 1 for visualization, from
    samples_max, _ = samples.flatten(2).max(2, keepdim=True)
    samples = samples / samples_max.unsqueeze(3)
    save_image(samples, os.path.join(args.output_dir, filename), nrow=n_row, normalize=True)


def train_and_evaluate(model, train_loader, test_loader, optimizer, scheduler, args):
    best_eval_logprob = float('-inf')

    for i in range(args.start_epoch, args.n_epochs):
        train(model, train_loader, optimizer, i, args)
        scheduler.step()
        eval_logprob, _ = evaluate(model, test_loader, i, args)

        # save training checkpoint
        torch.save({'epoch': i,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict()},
                   os.path.join(args.output_dir, 'model_checkpoint.pt'))
        # save model only
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'model_state.pt'))

        # save best state
        if eval_logprob > best_eval_logprob:
            best_eval_logprob = eval_logprob
            torch.save({'epoch': i,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict()},
                       os.path.join(args.output_dir, 'best_model_checkpoint.pt'))

        # plot sample
        generate(model, args, step=i)


# --------------------
# Plot
# --------------------

def plot_density(dist, ax, ranges, flip_var_order=False):
    (xmin, xmax), (ymin, ymax) = ranges
    # sample uniform grid
    n = 200
    xx1 = torch.linspace(xmin, xmax, n).cuda()
    xx2 = torch.linspace(ymin, ymax, n).cuda()
    xx, yy = torch.meshgrid(xx1, xx2)
    xy = torch.stack((xx.flatten(), yy.flatten()), dim=-1).squeeze()

    if flip_var_order:
        xy = xy.flip(1)

    # run uniform grid through model and plot
    density = dist.log_prob(xy).exp()
    ax.contour(xx.cpu(), yy.cpu(), density.cpu().view(n, n).data.numpy())

    # format
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([xmin, xmax])
    ax.set_yticks([ymin, ymax])


def plot_dist_sample(data, ax, ranges):
    ax.scatter(data[:, 0].cpu().data.numpy(), data[:, 1].cpu().data.numpy(), s=10, alpha=0.4)
    # format
    (xmin, xmax), (ymin, ymax) = ranges
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([xmin, xmax])
    ax.set_yticks([ymin, ymax])


def plot_sample_and_density(model, target_dist, args, ranges_density=[[-5, 20], [-10, 10]],
                            ranges_sample=[[-4, 4], [-4, 4]], step=None):
    model.eval()
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))

    # sample target distribution and pass through model
    data = target_dist.sample((2000,))
    u, _ = model(data.cuda())

    # plot density and sample
    plot_density(model, axs[0], ranges_density, args.flip_var_order)
    plot_dist_sample(u, axs[1], ranges_sample)

    # format and save
    matplotlib.rcParams.update({'xtick.labelsize': 'xx-small', 'ytick.labelsize': 'xx-small'})
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'sample' + (step != None) * '_epoch_{}'.format(step) + '.png'))
    plt.close()


# --------------------
# Run
# --------------------

if __name__ == '__main__':

    args = parser.parse_args()

    # setup kernel size, hidden_size and output dir according to scale factor
    args.kernel_size = min(args.sf * 4 + 3, 21)
    args.hidden_size = min(args.sf * 5 + 5, 25)
    args.output_dir += '_x{}'.format(args.sf)

    # setup file ops
    if not os.path.isdir(args.output_dir): os.makedirs(args.output_dir)

    # setup device
    args.device = torch.device('cuda:0' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.type == 'cuda': torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # load data
    train_dataloader, test_dataloader = fetch_dataloaders(args.val_save_path, args.kernel_size, args.sf,
                                                          args.batch_size, args.device)
    args.input_size = train_dataloader.dataset.input_size
    args.input_dims = train_dataloader.dataset.input_dims
    args.cond_label_size = train_dataloader.dataset.label_size if args.conditional else None
    args.alpha = train_dataloader.dataset.alpha
    args.normalization = train_dataloader.dataset.normalization

    # model
    model = KernelPrior(args.n_blocks, args.input_size, args.hidden_size, args.n_hidden, args.kernel_size, args.alpha,
                        args.normalization, args.cond_label_size, batch_norm=not args.no_batch_norm, )

    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = MultiStepLR(optimizer, milestones=[40, 50, 60, 70, 80, 90], gamma=0.5)

    if args.restore_file:
        # load model and optimizer states
        state = torch.load(args.restore_file, map_location=args.device)
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optimizer_state'])
        args.start_epoch = state['epoch'] + 1
        args.output_dir = os.path.dirname(args.restore_file)
    args.results_file = os.path.join(args.output_dir, args.results_file)

    print('Loaded settings and model:')
    print(pprint.pformat(args.__dict__))
    print(model)
    print(pprint.pformat(args.__dict__), file=open(args.results_file, 'a'))
    print(model, file=open(args.results_file, 'a'))

    if args.train:
        train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, scheduler, args)

    if args.evaluate:
        evaluate(model, test_dataloader, None, args)

    if args.generate:
        generate(model, args)
