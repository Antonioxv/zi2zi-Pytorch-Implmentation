import argparse
import glob
import os
import pickle
import random

from tqdm import tqdm


def pickle_examples(image_paths, train_path, val_path, train_val_split):
    """
    Compile a list of examples into pickled format, so during
    the training, all io will happen in memory
    """
    with open(train_path, 'wb') as ft:
        with open(val_path, 'wb') as fv:
            for p, label in image_paths:
                label = int(label)
                with open(p, 'rb') as f:
                    # print("img %s" % p, label)
                    img_bytes = f.read()
                    r = random.random()
                    example = (label, img_bytes)
                    if r < train_val_split:  # split
                        pickle.dump(example, fv)
                    else:
                        pickle.dump(example, ft)


# Script part, type in parameters to determine the functions
parser = argparse.ArgumentParser(description='Compile list of images into a pickled object for training')
parser.add_argument('--image_dir', dest='image_dir', required=True, help='path of examples')
parser.add_argument('--save_dir', dest='save_dir', required=True, help='path to save pickled files')
parser.add_argument('--split_ratio', type=float, default=0.05, dest='split_ratio',
                    help='split ratio between train and val, belongs to [0, 1]')


if __name__ == "__main__":
    args = parser.parse_args()

    # Make dirs
    bin_save_dir = args.save_dir
    os.makedirs(args.save_dir, exist_ok=True)

    # Paths
    train_path = os.path.join(bin_save_dir, "train.obj")
    val_path = os.path.join(bin_save_dir, "val.obj")

    # Take all formats of image into consideration
    total_file_list = sorted(
        glob.glob(os.path.join(args.image_dir, "*.jpg")) +
        glob.glob(os.path.join(args.image_dir, "*.png")) +
        glob.glob(os.path.join(args.image_dir, "*.tif"))
    )

    # Like '%d_%05d.png'
    cur_file_list = []
    for file_name in tqdm(total_file_list):
        label = os.path.basename(file_name).split('_')[0]  # style label
        label = int(label)  # to int
        cur_file_list.append((file_name, label))

    pickle_examples(image_paths=cur_file_list, train_path=train_path, val_path=val_path,
                    train_val_split=args.split_ratio)
