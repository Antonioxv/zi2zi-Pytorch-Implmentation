import os
import time
import argparse

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, TensorDataset

from data import DatasetFromObj
from model import Zi2Zi

# Script part, type in parameters to determine the functions
parser = argparse.ArgumentParser(description='Infer')
parser.add_argument('--experiment_dir', required=True,
                    help='experiment directory, data, samples,checkpoints,etc')
parser.add_argument('--start_from', type=int, default=0)
parser.add_argument('--gpu_ids', default=[], nargs='+', help="GPUs")
parser.add_argument('--image_size', type=int, default=256,
                    help="size of your input and output image")
parser.add_argument('--L1_penalty', type=int, default=100, help='weight for L1 loss')
parser.add_argument('--Lconst_penalty', type=int, default=15, help='weight for const loss')
# parser.add_argument('--Ltv_penalty', dest='Ltv_penalty', type=float, default=0.0, help='weight for tv loss')
parser.add_argument('--Lcategory_penalty', type=float, default=1.0,
                    help='weight for category loss')
parser.add_argument('--embedding_num', type=int, default=40,
                    help="number for distinct embeddings")
parser.add_argument('--embedding_dim', type=int, default=128, help="dimension for embedding")
parser.add_argument('--batch_size', type=int, default=16, help='number of examples in batch')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--resume', type=int, default=None, help='resume from previous training')
parser.add_argument('--obj_path', type=str, default='./experiment/data/val.obj', help='the obj file you infer')
parser.add_argument('--input_nc', type=int, default=1)

parser.add_argument('--from_txt', action='store_true')
parser.add_argument('--src_txt', type=str, default='大威天龍大羅法咒世尊地藏波若諸佛')
parser.add_argument('--canvas_size', type=int, default=256)
parser.add_argument('--char_size', type=int, default=256)
parser.add_argument('--label', type=int, default=0)
parser.add_argument('--src_font', type=str, default='charset/cjk/FZXKTJW.TTF')
parser.add_argument('--dst_font', type=str, default=None)


def draw_single_char(ch, font, canvas_size):
    img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), ch, (0, 0, 0), font=font)
    img = img.convert('L')
    return img


def main():
    args = parser.parse_args()
    data_dir = os.path.join(args.experiment_dir, "data")
    checkpoint_dir = os.path.join(args.experiment_dir, "checkpoint")
    sample_dir = os.path.join(args.experiment_dir, "sample")
    infer_dir = os.path.join(args.experiment_dir, "infer")
    os.makedirs(infer_dir, exist_ok=True)

    t0 = time.time()
    ''' Init the zi2zi model. '''
    # Device: CPU/GPU
    gpu_ids = args.gpu_ids if torch.cuda.is_available() else None
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
        is_training=False
    )
    model.setup()
    model.print_networks(True)
    model.load_networks(args.resume)

    t1 = time.time()

    if args.from_txt:
        src = args.src_txt
        if args.dst_font is not None:
            font = ImageFont.truetype(args.dst_font, size=args.char_size)
        else:
            font = ImageFont.truetype(args.src_font, size=args.char_size)

        img_list = [transforms.Normalize(0.5, 0.5)(
            transforms.ToTensor()(
                draw_single_char(ch, font, args.canvas_size)
            )
        ).unsqueeze(dim=0) for ch in src]
        label_list = [args.label for _ in src]

        img_list = torch.cat(img_list, dim=0)
        label_list = torch.tensor(label_list)

        dataset = TensorDataset(label_list, img_list, img_list)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    else:
        val_dataset = DatasetFromObj(os.path.join(data_dir, 'val.obj'),
                                     input_nc=args.input_nc,
                                     start_from=args.start_from)
        dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    global_steps = 0
    for batch in dataloader:
        model.sample(batch, infer_dir)
        global_steps += 1

    t_finish = time.time()

    print('cold start time: %.2f, hot start time %.2f' % (t_finish - t0, t_finish - t1))


if __name__ == '__main__':
    with torch.no_grad():
        main()
