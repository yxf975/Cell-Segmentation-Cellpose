import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from glob import glob


def rle_decode(mask_rle, shape, color=1):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.float32)
    for lo, hi in zip(starts, ends):
        img[lo: hi] = color
    return img.reshape(shape)


if __name__ == "__main__":
    train_src_folder = "./train2/"
    val_src_folder = "./val2/"
    mask_fold = "./train2/"
    im_paths = glob(train_src_folder + "*.png")

    df = pd.read_csv("/content/drive/MyDrive/cellSegmentation/train.csv")
    df = df.groupby('id')['annotation'].agg(lambda x: list(x)).reset_index()

    for id, annotation in df.values:
        all_masks = np.zeros((520, 704), dtype=np.float32)
        for i, mask in enumerate(annotation):
            decoded_mask = rle_decode(mask_rle=mask, shape=(520, 704), color=i)
            no_overlap_mask = np.multiply(all_masks, np.logical_not(np.logical_and(all_masks, decoded_mask)))
            all_masks = no_overlap_mask + decoded_mask
        all_masks = all_masks.astype(np.uint16)
        im = Image.fromarray(all_masks)
        im.save(mask_fold + id + "_masks.tif")
