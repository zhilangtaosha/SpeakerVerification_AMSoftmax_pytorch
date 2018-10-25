#from __future__ import print_function
import argparse
import torch.optim as optim
import torchvision.transforms as transforms

import torch.backends.cudnn as cudnn
import os

from tqdm import tqdm
from model.model import *
from evaluate.eval_metrics import evaluate

from Dataset import DeepSpeakerSoftmaxDataset,Testset
# from model.model import PairwiseDistance
from Dataset import totensor
from utils import *

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Speaker Recognition')

parser.add_argument('--dataroot', type=str, default='./data_aishell/wav/train',
                    help='path to dataset')

parser.add_argument('--resume',default=None,
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=50, metavar='E',
                    help='number of epochs to train (default: 10)')
# Training options
parser.add_argument('--embedding-size', type=int, default=512, metavar='ES',
                    help='Dimensionality of the embedding')

parser.add_argument('--batch-size', type=int, default=512, metavar='BS',
                    help='input batch size for training (default: 128)')

parser.add_argument('--test-batch-size', type=int, default=64, metavar='BST',
                    help='input batch size for testing (default: 64)')

parser.add_argument('--test-input-per-file', type=int, default=1, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')

parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.125)')
parser.add_argument('--lr-decay', default=1e-4, type=float, metavar='LRD',
                    help='learning rate decay ratio (default: 1e-4')

parser.add_argument('--optimizer', default='adagrad', type=str,
                    metavar='OPT', help='The optimizer to use (default: Adagrad)')

parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')

parser.add_argument('--log-interval', type=int, default=1, metavar='LI',
                    help='how many batches to wait before logging training status')

parser.add_argument('--mfb', action='store_true', default=True,
                    help='start from MFB file')

parser.add_argument('--makemfb', action='store_true', default=True,
                    help='need to make mfb file')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
np.random.seed(args.seed)

if args.cuda:
    cudnn.benchmark = True

ckpt_dir = r'./ckpt_baseline_0.5'

s = 20.0
m = 0.0
wd = 0.0002

is_dropout=False
is_random_flip=False
is_SEnet = True
is_FN = True

name = os.path.split(ckpt_dir)[1]
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
# l2_dist = PairwiseDistance(2)

tongdun_train_dir = r'./aishell/data_aishell/wav/train'
tongdun_id_list = os.listdir(tongdun_train_dir)
transform = transforms.Compose([totensor()])

def create_optimizer(model, new_lr):
    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, dampening=0.9,
                              weight_decay=wd)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=wd)
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(),
                                  lr=new_lr,
                                  lr_decay=args.lr_decay,
                                  weight_decay=wd)
    return optimizer

def main_train():
    train_dir = DeepSpeakerSoftmaxDataset(dir_list=tongdun_id_list, root_dir=args.dataroot, transform=transform,is_random_flip=is_random_flip)
    # print the experiment configuration
    print('\nparsed options:\n{}\n'.format(vars(args)))
    print('\nNumber of Classes:\n{}\n'.format(len(train_dir.classes)))
    # instantiate model and initialize weights
    model = DeepSpeakerModel(embedding_size=args.embedding_size, num_classes=340, ratio=0.5, is_fn=is_FN, is_dropout=is_dropout, is_SEnet=is_SEnet)
    print(model)

    if args.cuda:
        model.cuda()

    optimizer = create_optimizer(model, args.lr)
    # optionally resume from a checkpoint

    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('=> no checkpoint found at {}'.format(args.resume))

    start = args.start_epoch
    end = start + args.epochs

    train_loader = torch.utils.data.DataLoader(train_dir, batch_size=args.batch_size, shuffle=False, **kwargs)

    for epoch in range(start, end):
        train_softmax(train_loader, model, optimizer, epoch)

def train_softmax(train_loader, model, optimizer, epoch):
    top1       = AverageMeter()
    top5       = AverageMeter()

    # switch to train mode
    model.train()
    criterion = AMSoftmax()

    pbar = tqdm(enumerate(train_loader))
    for batch_idx, (data,label) in pbar:
        data= data.cuda()
        data_var= Variable(data)

        label = label.cuda()
        label_var = Variable(label)

        # compute output
        out_fea, out_cls= model(data_var)

        # if epoch > args.min_softmax_epoch:
        loss = criterion(out_cls, label_var, scale = s, margin = m)

        prec1, prec5 = accuracy(out_cls.data, label, topk=(1, 5))
        top1.update(prec1[0], data.size(0))
        top5.update(prec5[0], data.size(0))

        # compute gradient and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            pbar.set_description(name+' Train Epoch: {:3d} [{:8d}/{:8d} ({:3.0f}%)]\tLoss: {:.6f}\tacc: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),loss.data[0],
                        top1.avg))

    # do checkpointing
    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()},
               '{}/checkpoint_{}.pth'.format(ckpt_dir, epoch))

def main_test(test_ckpt_dir, start, end):
    print(os.path.split(test_ckpt_dir)[1])

    pairs_path = r'test_pairs.txt'
    test_dir = Testset(pairs_path, transform=transform)

    # # instantiate model and initialize weights
    model = DeepSpeakerModel(embedding_size=args.embedding_size, num_classes=340,ratio=0.5, is_fn=True, is_SEnet=is_SEnet)
    print(model)

    test_loader = torch.utils.data.DataLoader(test_dir, batch_size=1, shuffle=False, **kwargs)

    if args.cuda:
        model.cuda()

    max_acc = 0.0
    max_epoch = 0

    for epoch in range(start, end):
        args.resume = os.path.join(test_ckpt_dir, 'checkpoint_' + str(epoch) + '.pth')
        if args.resume:
            if os.path.isfile(args.resume):
                # print('=> loading checkpoint {}'.format(args.resume))
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch']
                checkpoint = torch.load(args.resume)
                model.load_state_dict(checkpoint['state_dict'])
            else:
                print('=> no checkpoint found at {}'.format(args.resume))

        acc = val(test_loader, model, epoch)

        if acc > max_acc:
            max_acc = acc
            max_epoch = epoch
    print('max acc: ', max_acc)
    print('max epoch: ', max_epoch)
        # break

def val(test_loader, model, epoch):
    from numpy import linalg as LA
    def ConsinDistance(feaV1, feaV2):
        return np.dot(feaV1, feaV2) / (LA.norm(feaV1) * LA.norm(feaV2))
    # switch to evaluate mode
    model.eval()

    labels, distances, distances_flip = [], [], []
    fea1, fea2 = [], []
    for batch_idx, (data_a_list, data_p_list,data_a_flip_list, data_p_flip_list, label) in enumerate(test_loader):
        label = Variable(label)
        labels.append(label.data.cpu().numpy())

        def cal_test_fea(a_list, p_list):
            data_a = torch.cat(a_list, 0)
            data_p = torch.cat(p_list, 0)

            current_sample = 10
            data_a = data_a.resize_(current_sample, 1, data_a.size(2), data_a.size(3))
            data_p = data_p.resize_(current_sample, 1, data_a.size(2), data_a.size(3))

            if args.cuda:
                data_a, data_p = data_a.cuda(), data_p.cuda()

            data_a = Variable(data_a, volatile=True)
            data_p = Variable(data_p, volatile=True)

            # compute output
            out_a, out_p = model(data_a), model(data_p)
            out_a = out_a[0].data.cpu().numpy()
            out_p = out_p[0].data.cpu().numpy()

            a = np.sum(out_a, axis=0)
            p = np.sum(out_p, axis=0)
            return a, p

        a, p = cal_test_fea(data_a_list, data_p_list)
        dists = 1.0 - (ConsinDistance(a, p) + 1.0)/2.0
        dists = dists.reshape(1,1).mean(axis=1)
        distances.append(dists)

    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])

    tpr, fpr, accuracy, val,  far = evaluate(distances, labels)
    print('Test Epoch: ', epoch)
    print('\33[91mTest set: Accuracy: {:.8f}\n\33[0m'.format(np.mean(accuracy)))
    return np.mean(accuracy)



if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    main_train()
    # main_test(r'./ckpt_baseline_0.5_s20_m0.0_wd0.0_fn_input_64_SE', 20, 50)