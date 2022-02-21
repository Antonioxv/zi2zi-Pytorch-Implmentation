import argparse
import math
import os
import random
import time
import torch

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from data import DatasetFromObj
from model import Zi2Zi

print(torch.__version__)
print(torch.cuda.is_available())

# Script part, type in parameters to determine the functions
parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--experiment_dir', required=True, help='experiment directory, data, samples,checkpoints,etc')
parser.add_argument('--gpu_ids', default=[], nargs='+', help="GPUs")
parser.add_argument('--image_size', type=int, default=256, help="size of your input and output image")
parser.add_argument('--L1_penalty', type=int, default=100, help='weight for L1 loss')
parser.add_argument('--Lconst_penalty', type=int, default=15, help='weight for const loss')
# parser.add_argument('--Ltv_penalty', dest='Ltv_penalty', type=float, default=0.0, help='weight for tv loss')
parser.add_argument('--Lcategory_penalty', type=float, default=1.0, help='weight for category loss')
parser.add_argument('--embedding_num', type=int, default=40, help="number for distinct embeddings")
parser.add_argument('--embedding_dim', type=int, default=128, help="dimension for embedding")
parser.add_argument('--epoch', type=int, default=100, help='number of epoch')
parser.add_argument('--batch_size', type=int, default=32, help='number of examples in batch')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--schedule', type=int, default=20, help='number of epochs to half learning rate')
parser.add_argument('--inst_norm', action='store_true', help='use conditional instance normalization in your model')
parser.add_argument('--sample_steps', type=int, default=10, help='number of batches in between two samples are drawn from validation set')
parser.add_argument('--checkpoint_steps', type=int, default=100, help='number of batches in between two checkpoints')
parser.add_argument('--random_seed', type=int, default=777, help='random seed for random and pytorch')
parser.add_argument('--resume', type=int, default=None, help='resume from previous training')
parser.add_argument('--input_nc', type=int, default=1,  # instead of 3
                    help='number of input images channels')


def main():
    args = parser.parse_args()

    ''' Fix random seed during the experiment. '''
    random.seed(args.random_seed)  # random seed is 103
    torch.manual_seed(args.random_seed)

    ''' Make experiment dirs. '''
    experiment_dir = args.experiment_dir
    os.makedirs(experiment_dir, exist_ok=True)
    data_dir = os.path.join(experiment_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    # Copy obj to data path
    # os.
    checkpoint_dir = os.path.join(experiment_dir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    sample_dir = os.path.join(experiment_dir, "sample")
    os.makedirs(sample_dir, exist_ok=True)
    # Tensorboard
    run_dir = os.path.join(experiment_dir, "runs")
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(run_dir)

    ''' Init the zi2zi model. '''
    # Device: CPU/GPU
    gpu_ids = args.gpu_ids if torch.cuda.is_available() else None
    # Zi2Zi does not inherit nn.Module, but Gen and Dis do, so it can be encapsulated as a nn.Module-like class.
    model = Zi2Zi(
        input_nc=args.input_nc,
        image_size=args.image_size,
        embedding_num=args.embedding_num,
        embedding_dim=args.embedding_dim,
        L1_penalty=args.L1_penalty,
        Lconst_penalty=args.Lconst_penalty,
        Lcategory_penalty=args.Lcategory_penalty,
        # Ltv_penalty=args.Ltv_penalty,
        lr=args.lr,
        gpu_ids=gpu_ids,
        save_dir=checkpoint_dir,
        is_training=True
    )
    print('Model initialized.')
    # Continue training from resume step.
    if args.resume:
        model.load_networks(args.resume)
    # Move parameters to gpu when gpu is available, and set nn.Module train or eval.
    model.setup()
    # Print model' nn architecture.
    model.print_networks(True)

    ''' Process datasets and data loaders. '''
    # Train dataset and dataloader.
    train_obj = os.path.join(data_dir, 'train.obj')
    # train_dataset = DatasetFromObj(train_obj, input_nc=args.input_nc, augment=True, bold=False, rotate=False, blur=True)
    # train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)  # See more in train part.
    # Validate dataset and dataloader.
    val_obj = os.path.join(data_dir, 'val.obj')
    val_dataset = DatasetFromObj(val_obj, input_nc=args.input_nc)  # No augment.
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)  # Val dataset load only once, no shuffle.

    ''' Train. '''
    global_steps = 0
    start_time = time.time()
    for epoch in range(args.epoch):
        # 1) Generate train dataset every epoch, so that different styles of saved char images can be trained.
        # No bold and no rotate, only generate blur, the first 2 is not performing well in the dataset!
        train_dataset = DatasetFromObj(train_obj, input_nc=args.input_nc, augment=True, bold=False, rotate=False, blur=True)
        # Shuffling.
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        # Cal once is enough.
        num_batch = math.ceil(len(train_dataset) / args.batch_size)  # Cal once is enough.

        # 2) Train one epoch
        for bid, batch in enumerate(train_dataloader):  # [bid, [labels, tgt_imgs, src_imgs]]
            # 2.1) Set input data and train one step.
            model.set_input(batch[0], batch[2], batch[1])  # [labels, src_imgs, tgt_imgs]
            category_loss, const_loss, l1_loss, cheat_loss = model.forward()  # nn.Module-like class

            # 2.2) Print log and save it to tensorboard.
            if bid % 10 == 0:
                writer.add_scalar('Train/g_loss', model.g_loss.item(), global_step=global_steps)
                writer.add_scalar('Train/d_loss', model.d_loss.item(), global_step=global_steps)
            if bid % 100 == 0:
                passed_time = time.time() - start_time
                log_format = "Epoch: [%3d], [%4d/%4d] time: %4.2f, g_loss: %.5f, d_loss: %.5f, " + \
                             "category_loss: %.5f, cheat_loss: %.5f, l1_loss: %.5f, const_loss: %.5f"
                print(log_format % (epoch, bid, num_batch, passed_time, model.g_loss.item(), model.d_loss.item(),
                                    category_loss, cheat_loss, l1_loss, const_loss))
            # 2.3) Save checkpoint.
            if global_steps % args.checkpoint_steps == 0:
                model.save_networks(global_steps)
                print("Checkpoint: save checkpoint step %d" % global_steps)
            # 2.4) Sample images from val dataset.
            if global_steps % args.sample_steps == 0:
                for vbid, val_batch in enumerate(val_dataloader):
                    model.sample(val_batch, os.path.join(sample_dir, str(global_steps)))
                print("Sample: sample step %d" % global_steps)

            global_steps += 1

        # 3) Update learning rate.
        if (epoch + 1) % args.schedule == 0:
            model.update_lr()

    ''' Validate. '''
    for vbid, val_batch in enumerate(val_dataloader):
        model.sample(val_batch, os.path.join(sample_dir, str(global_steps)))

    # Save final version of checkpoint.
    model.save_networks(global_steps)
    print("Checkpoint: save checkpoint step %d" % global_steps)

    writer.close()
    print('Terminated')


if __name__ == '__main__':
    main()
