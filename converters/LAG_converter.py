import numpy as np
import os
from converters.utils_custom import  *
import shutil
from tqdm import tqdm
import argparse
import glob

"""
I assume that the train, test and val set should be balanced, i.e. possess the same class ratio.
The script expects a directory structure like:
eyescan-dataloader/
    |
    +-----data/
    |       |
    |       +------raw/
    |               |
    |               +------LAG/
    |
    +-----LAG_converter.py
"""


if __name__ == "__main__":
    # set the seed for reproducability
    set_random_seed()

    db = "LAG"
    os.chdir("..")
    # first step: define the paths
    parser = argparse.ArgumentParser(description=f"Convert {db} dataset to a consistent structure.")
    parser.add_argument('--in_path', default=os.path.join(os.getcwd(), 'dataset', 'raw', db), type=str, help='Path to the raw dataset')
    parser.add_argument('--out_path', default=os.path.join(os.getcwd(), 'dataset', 'preprocessed', db), type=str)
    parser.add_argument('--img_lib', default='pil', type=str, help='Image library to use for loading and resizing the images. Implemented: cv2, pil')
    parser.add_argument('--num_workers', default=-1, type=int, help='Number of parallel workers. Default -1 -> number '
                                                                    'of cpu cores')
    parser.add_argument('--preprocessing', default='', type=str, help='Kind of preprocessing that should be applied on the images. Currently available: kdrd else there will be no preprocessing')
    args = parser.parse_args()

    in_path = args.in_path
    out_path = args.out_path
    out_paths = {
        "train" : [os.path.join(out_path, "train", "HE"),
                   os.path.join(out_path, "train", "DR"),
                   os.path.join(out_path, "train", "GL")],
        "val" :   [os.path.join(out_path, "val", "HE"),
                   os.path.join(out_path, "val", "DR"),
                   os.path.join(out_path, "val", "GL")],
        "test" :  [os.path.join(out_path, "test", "HE"),
                   os.path.join(out_path, "test", "DR"),
                   os.path.join(out_path, "test", "GL")]
    }

    # check if the outpath exists
    check_and_create_out_dirs(out_paths)

    # path to the images
    img_path_he = os.path.join(in_path, "non_glaucoma", "image")
    img_path_gl = os.path.join(in_path, "suspicious_glaucoma", "image")


    # create a temp path to resize the images
    # we later use this directory as a working directory
    tmp_path = os.path.join("tmp")

    # start by resizing the images:
    resize_images(img_path_he, tmp_path, mode=args.img_lib, num_workers=args.num_workers, preprocessing=args.preprocessing)
    resize_images(img_path_gl, tmp_path, mode=args.img_lib, num_workers=args.num_workers, preprocessing=args.preprocessing)

    # this time the labels are encoded in the folder name
    healthy = [os.path.split(f)[-1] for f in sorted(glob.glob(os.path.join(img_path_he, "*.jpg")))]
    diabetic = []
    glaucoma = [os.path.split(f)[-1] for f in sorted(glob.glob(os.path.join(img_path_gl, "*.jpg")))]

    # split the class lists into train, val, and test
    he_train, he_val, he_test = train_test_val_split(healthy, train=0.8, val=0.1, test=0.1)
    dr_train, dr_val, dr_test = train_test_val_split(diabetic, train=0.8, val=0.1, test=0.1)
    gl_train, gl_val, gl_test = train_test_val_split(glaucoma, train=0.8, val=0.1, test=0.1)

    # generate the new names
    he_train_out = generate_out_names(len(he_train), f"{db}_train_he_")
    he_val_out = generate_out_names(len(he_val), f"{db}_val_he_")
    he_test_out = generate_out_names(len(he_test), f"{db}_test_he_")

    dr_train_out = generate_out_names(len(dr_train), f"{db}_train_dr_")
    dr_val_out = generate_out_names(len(dr_val), f"{db}_val_dr_")
    dr_test_out = generate_out_names(len(dr_test), f"{db}_test_dr_")

    gl_train_out = generate_out_names(len(gl_train), f"{db}_train_gl_")
    gl_val_out = generate_out_names(len(gl_val), f"{db}_val_gl_")
    gl_test_out = generate_out_names(len(gl_test), f"{db}_test_gl_")

    # create lists for moving the files
    in_files = [he_train, he_val, he_test, dr_train, dr_val, dr_test, gl_train, gl_val, gl_test]
    out_files = [he_train_out, he_val_out, he_test_out, dr_train_out, dr_val_out, dr_test_out, gl_train_out, gl_val_out, gl_test_out]
    out_paths_list = [out_paths["train"][0], out_paths["val"][0], out_paths["test"][0], out_paths["train"][1], out_paths["val"][1], out_paths["test"][1], out_paths["train"][2], out_paths["val"][2], out_paths["test"][2]]

    # iterate over the lists and move the files
    for out_path, in_file, out_file in tqdm(zip(out_paths_list, in_files, out_files)):
        for in_f, out_f in zip(in_file, out_file):
            if not in_f.startswith('.'):
                shutil.move(os.path.join(tmp_path, in_f), os.path.join(out_path, out_f))

    # remove tmp directory
    shutil.rmtree(tmp_path)

    # create the overview csv
    generate_overview_csv(args.out_path)
