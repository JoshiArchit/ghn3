# Copyright (c) 2023. Samsung Electronics Co., Ltd. All Rights Reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Trains a Graph HyperNetwork (GHN-3) on DeepNets-1M and ImageNet. DistributedDataParallel (DDP) training is
used if `torchrun` is used as shown below.
This script assumes the ImageNet dataset is already downloaded and set up as described in scripts/imagenet_setup.sh.

Example:

    # To train GHN-3-T/m8 on ImageNet (make sure to put the DeepNets-1M dataset in $SLURM_TMPDIR or in ./data) on
    # single GPU, automatic mixed precision:
    python train_ghn_ddp.py -d imagenet -D $SLURM_TMPDIR -n -v 50 --ln \
    -e 75 --opt adamw --lr 4e-4 --wd 1e-2 -b 128 --amp -m 8 --name ghn3tm8 --hid 64 --scheduler cosine-warmup --debug 0

    # 4 GPUs (DDP), automatic mixed precision (as in the paper):
    export OMP_NUM_THREADS=8
    torchrun --standalone --nnodes=1 --nproc_per_node=4 train_ghn_ddp.py -d imagenet -D $SLURM_TMPDIR -n -v 50 --ln \
    -e 75 --opt adamw --lr 4e-4 --wd 1e-2 -b 128 --amp -m 8 --name ghn3tm8 --hid 64 --scheduler cosine-warmup --debug 0

    # Sometimes, there can be mysterious errors due to DDP (depending on the pytorch/cuda version).
    # So it can be a good idea to wrap this command in a for loop to continue training in case of failure.

    # Use eval_ghn_imagenet.py to evaluate the trained GHN-3 model on ImageNet.

    # To train GHN-3-T/m8 on CIFAR-10:
    python train_ghn_ddp.py -n -v 50 --ln -m 8 --name ghn3tm8-c10 --hid 64 --layers 3 --opt adamw --lr 4e-4 --wd 1e-2 \
     --scheduler cosine-warmup --debug 0

"""


import argparse
import torch.distributed as dist
from functools import partial
from ppuda.config import init_config
from ppuda.vision.loader import image_loader
from ghn3 import GHN3, log, Trainer, DeepNets1MDDP, setup_ddp, clean_ddp

log = partial(log, flush=True)


def main():
    parser = argparse.ArgumentParser(description='GHN-3 training')
    parser.add_argument('--layers', type=int, default=3, help='number of layers in GHN-3')
    parser.add_argument('--heads', type=int, default=8, help='number of self-attention heads in GHN-3')
    parser.add_argument('--compile', type=str, default=None, help='use pytorch2.0 compilation for potential speedup')
    parser.add_argument('--ghn2', action='store_true', help='train GHN-2, also can use'
                                                            ' https://github.com/facebookresearch/ppuda to train GHN-2')

    ddp = setup_ddp()
    args = None
    if ddp.ddp and ddp.rank == 0:
        args = init_config(mode='train_ghn', parser=parser, verbose=ddp.rank == 0)
    if ddp.ddp:
        dist.barrier()  # wait for the save folder to be created by rank 0 process
    if args is None:
        args = init_config(mode='train_ghn', parser=parser, verbose=ddp.rank == 0)

    if hasattr(args, 'multigpu') and args.multigpu:
        raise NotImplementedError(
            'the `multigpu` argument was meant to use nn.DataParallel in the GHN-2 code. '
            'nn.DataParallel is likely to be deprecated in PyTorch in favor of nn.DistributedDataParallel '
            '(https://github.com/pytorch/pytorch/issues/659360).'
            'Therefore, this repo is not supporting DataParallel anymore as it complicates some steps. '
            'nn.DistributedDataParallel is used if this script is called with torchrun (see examples on top).')
    is_imagenet = args.dataset.startswith('imagenet')

    log('loading the %s dataset...' % args.dataset.upper())
    train_queue, _, num_classes = image_loader(args.dataset,
                                               args.data_dir,
                                               im_size=224 if is_imagenet else 32,
                                               test=False,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               seed=args.seed,
                                               verbose=ddp.rank == 0)

    hid = args.hid
    config = {'max_shape': args.max_shape, 'num_classes': num_classes, 'hypernet': args.hypernet,
              'decoder': args.decoder, 'weight_norm': args.weight_norm, 've': args.virtual_edges > 1,
              'layernorm': args.ln, 'hid': hid, 'layers': args.layers, 'heads': args.heads, 'is_ghn2': args.ghn2}

    s = 16 if is_imagenet else 11
    default_max_shape = (hid * 2, hid * 2, s, s) if args.ghn2 else (hid, hid, s, s)
    if args.max_shape != default_max_shape:
        log('WARNING: max_shape {} is different from the default GHN2/GHN3 max_shape {}. '.format(
            args.max_shape, default_max_shape))

    ghn = GHN3(**config, debug_level=args.debug)
    graphs_queue, sampler = DeepNets1MDDP.loader(args.meta_batch_size // (ddp.world_size if ddp.ddp else 1),
                                                 dense=ghn.is_dense(),
                                                 wider_nets=is_imagenet,
                                                 split=args.split,
                                                 nets_dir=args.data_dir,
                                                 virtual_edges=args.virtual_edges,
                                                 num_nets=args.num_nets,
                                                 large_images=is_imagenet,
                                                 verbose=ddp.rank == 0,
                                                 debug=args.debug > 0)

    lr_scheduler = 'mstep' if args.scheduler is None else args.scheduler
    scheduler_args = {'milestones': args.lr_steps, 'gamma': args.gamma} if lr_scheduler == 'mstep' else None
    trainer = Trainer(ghn,
                      opt=args.opt,
                      opt_args={'lr': args.lr, 'weight_decay': args.wd},
                      scheduler=lr_scheduler,
                      scheduler_args=scheduler_args,
                      n_batches=len(train_queue),
                      grad_clip=args.grad_clip,
                      device=args.device,
                      log_interval=args.log_interval,
                      amp=args.amp,
                      amp_min_scale=1024,       # this helped stabilize AMP training
                      amp_growth_interval=100,  # this helped stabilize AMP training
                      predparam_wd=0 if args.ghn2 else 3e-5,
                      label_smoothing=0.1 if is_imagenet else 0.0,
                      save_dir=args.save,
                      ckpt=args.ckpt,
                      epochs=args.epochs,
                      verbose=ddp.rank == 0,
                      compile_mode=args.compile)

    log('\nStarting training GHN with {} parameters!'.format(sum([p.numel() for p in ghn.parameters()])))
    if ddp.ddp:
        log(f"shuffle DeepNets1MDDP train loader (set seed to {args.seed})", flush=True)
        # first set sample order according to the seed
        sampler.sampler.set_epoch(args.seed)
    graphs_queue = iter(graphs_queue)

    for epoch in range(trainer.start_epoch, args.epochs):

        log('\nepoch={:03d}/{:03d}, lr={:e}'.format(epoch + 1, args.epochs, trainer.get_lr()))

        if ddp.ddp and epoch > trainer.start_epoch:  # make sure sample order is different for each epoch
            log(f'shuffle DeepNets1MDDP train loader (set seed to {epoch})')
            sampler.sampler.set_epoch(epoch)

        trainer.reset_metrics(epoch)

        for step_, (images, targets) in enumerate(train_queue):

            step = step_ + (trainer.start_step if epoch == trainer.start_epoch else 0)
            if step >= len(train_queue):  # if we resume training from some step > 0, then need to break the loop
                break

            trainer.update(images, targets, graphs=next(graphs_queue))
            trainer.log(step)
            if args.save:
                trainer.save(epoch, step, {'args': args})  # save GHN checkpoint

        trainer.scheduler_step()  # lr scheduler step

    log('done!', flush=True)
    if ddp.ddp:
        clean_ddp()


if __name__ == '__main__':
    main()
