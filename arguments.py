import importlib
easydict_spec = importlib.util.find_spec("easydict", package="easydict")

#done to use in Google Colab since argparse does not work
if easydict_spec is not None:
    print('EasyDict found. Using it!')
    import easydict
    args = easydict.EasyDict({
        'workers': 4,
        'epochs': 300,
        'start_epoch': 0,
        'train_batch': 128,
        'test_batch': 128,
        'lr': 0.0085,
        'drop': 0.5,
        'schedule': [150, 300],
        'gamma': 0.1,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'checkpoint': "checkpoint",
        'resume': '',
        'arch': 'resnet18',
        'depth': 29,
        'cardinality': 32,
        'base_width': 4,
        'widen_factor': 4,
        'evaluate': False,
        'pretrained': True,
        'gpu_id': '0',
        'manualSeed': None
    })
    print(args)
    state = args
else:
    print('argparse found. Using it!')
    import argparse
    # Parse arguments
    parser = argparse.ArgumentParser(description='Plant Image Training')

    # Datasets
    parser.add_argument('-d', '--data', default='path to dataset', type=str)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    # Optimization options
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                        help='train batchsize (default: 32)')
    parser.add_argument('--test-batch', default=128, type=int, metavar='N',
                        help='test batchsize (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.0085, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--drop', '--dropout', default=0.5, type=float,
                        metavar='Dropout', help='Dropout ratio')
    parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                            help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    # Checkpoints
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    # Architecture

    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet18)')

    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18')
    parser.add_argument('--depth', type=int, default=29, help='Model depth.')
    parser.add_argument('--cardinality', type=int, default=32, help='ResNet cardinality (group).')
    parser.add_argument('--base-width', type=int, default=4, help='ResNet base width.')
    parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
    # Miscs
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    #Device options
    parser.add_argument('--gpu-id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')

    args = parser.parse_args()
    state = {k: v for k, v in args._get_kwargs()}
