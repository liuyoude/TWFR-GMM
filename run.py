import os
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from extractor import SpecExtractor
from gmmer import GMMer

import utils


def main(args):
    # spectrogram
    # transform = utils.Wave2Spec(sr=args.sr)
    # log mel spectrogram
    transform = utils.Wave2Mel(sr=args.sr)
    extractor = SpecExtractor(args=args, dim=1, transform=transform)
    gmmer = GMMer(args=args, extractor=extractor)
    # test and eval on search config
    gmmer.test(train_dirs=sorted(args.train_dirs),
               valid_dirs=sorted(args.valid_dirs),
               save=args.save,
               use_search=True)
    gmmer.eval(train_dirs=sorted(args.add_train_dirs),
               test_dirs=sorted(args.test_dirs),
               save=args.save,
               use_search=True)
    # test and eval on settings in function param list
    # gmmer.test(train_dirs=sorted(args.train_dirs),
    #            valid_dirs=sorted(args.valid_dirs),
    #            save=args.save,
    #            use_search=False,
    #            gmm_n=2, # setting
    #            use_smote= False, # setting
    #            )
    # gmmer.eval(train_dirs=sorted(args.add_train_dirs),
    #            test_dirs=sorted(args.test_dirs),
    #            use_search=False,
    #            gmm_n=2,  # setting
    #            use_smote=False,  # setting
    #            )



def main_search(args):
    # transform = utils.Wave2Spec(sr=args.sr)
    transform = utils.Wave2Mel(sr=args.sr)
    extractor = SpecExtractor(args=args, dim=1, transform=transform)
    gmmer = GMMer(args=args, extractor=extractor)
    # search gmm_n on mean-gmm and max-gmm
    if args.version == 'mean-gmm':
        gmmer.search(train_dirs=sorted(args.train_dirs),
                     valid_dirs=sorted(args.valid_dirs),
                     start=100, end=101, step=1,
                     gmm_ns=[1, 2, 4, 8],
                     use_smote=False)
    elif args.version == 'max-gmm':
        gmmer.search(train_dirs=sorted(args.train_dirs),
                     valid_dirs=sorted(args.valid_dirs),
                     start=0, end=1, step=1,
                     gmm_ns=[1, 2, 4, 8],
                     use_smote=False)
    elif args.version == 'twfr-gmm':
        gmmer.search(train_dirs=sorted(args.train_dirs),
                     valid_dirs=sorted(args.valid_dirs),
                     start=0, end=102, step=2,
                     gmm_ns=[1, 2, 4, 8],
                     use_smote=False)
    elif args.version == 'smote-twfr-gmm':
        gmmer.search(train_dirs=sorted(args.train_dirs),
                     valid_dirs=sorted(args.valid_dirs),
                     start=0, end=102, step=2,
                     gmm_ns=[1, 2, 4, 8],
                     use_smote=True)


def run():
    # init config parameters
    params = utils.load_yaml(file_path='./config.yaml')
    parser = argparse.ArgumentParser(description=params['description'])
    for key, value in params.items():
        parser.add_argument(f'--{key}', default=value, type=type(value))
    args = parser.parse_args()
    # random seed
    if args.seed: utils.setup_seed(args.seed)
    log_dir = f'runs/{args.version}'
    writer = SummaryWriter(log_dir=log_dir)
    logger = utils.get_logger(filename=os.path.join(log_dir, 'running.log'))
    # run
    args.writer = writer
    args.logger = logger
    args.logger.info(args)
    print(args.version)
    main(args)
    # main_search(args)


if __name__ == '__main__':
    run()
