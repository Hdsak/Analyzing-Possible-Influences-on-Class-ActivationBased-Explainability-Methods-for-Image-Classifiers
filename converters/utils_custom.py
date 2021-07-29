import sys
import os
import argparse
import random
import csv
import cv2
from tqdm import tqdm
import re
import numpy as np
import torch
from PIL import Image
from multiprocessing import Pool
import multiprocessing as mp

import shutil

IMG_EXTENSIONS = ["jpg", "png", "jpeg", "tiff", "bmp", "gif", "ppm", "pgm", "tif", "svg"] # list of image file extension to check for when loading data



class Resizer(object):
    """
    a function object to use in resize_images.
    We use the __call__ to perform multiprocessing
    """
    def __init__(self, mode, in_path, out_path, max_size, crop_borders, preproc, frame):
        self.mode = mode
        self.in_path = in_path
        self.out_path = out_path
        self.max_size = max_size
        self.crop_borders = crop_borders
        self.preprocessing = preproc
        self.frame = frame

        self.prep_fail_files = []
        self.mask_fail_files = []

    def __call__(self, file):
        # read the image
        if self.mode == "pil":
            img_ = Image.open(os.path.join(self.in_path, file))
            img = np.array(img_)
        elif self.mode == "cv2":
            img = cv2.imread(os.path.join(self.in_path, file))
        if img is None:
            print(f"loading fail: {file}")

        # 1. cropping the borders if necessary
        if self.crop_borders and not "ACRIMA" in self.in_path and not "LAG" in self.in_path:
            img = np.array(img).astype(np.uint8)
            #print(img, img.shape, img.dtype)
            img = crop_img_borders(img)

        # if the mode is PIL we need to update the image
        if self.mode == "pil":
            img_ = Image.fromarray(img)


        # 2. resizing the image
        # find larger side
        idx = 0 if img.shape[0] > img.shape[1] else 1

        # if the longer side is larger than max_size
        if img.shape[idx] > self.max_size:
            # get width and height
            width = img.shape[1]
            height = img.shape[0]

            # calculate the new height and width
            if idx == 1:
                new_height = int(self.max_size*height/width)
                new_width = self.max_size
            elif idx == 0:
                new_height = self.max_size
                new_width = int(self.max_size*width/height)

            if self.mode == "cv2":
                # use cubic interpolation to resize the image
                new_img = cv2.resize(img, dsize=(new_width, new_height), interpolation=cv2.INTER_CUBIC)
            elif self.mode == "pil":
                new_img = img_.resize((new_width, new_height))

            img = np.array(new_img).astype(np.uint8)

        # if the mode is PIL we need to update the image
        if self.mode == "pil":
            img_ = Image.fromarray(img)

        # 3. preprocessing
        try:
            img, return_code = preprocess_img(img, self.preprocessing, frame=self.frame)
            if not return_code:
                self.mask_fail_files.append(os.path.join(self.in_path, file))
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(exc_tb.tb_lineno, e)
            self.prep_fail_files.append(os.path.join(self.in_path, file))

        img = img.astype(np.uint8)
        # if the mode is PIL we need to update the image
        if self.mode == "pil":
            img_ = Image.fromarray(img)

        # 4. save images under the new path
        handle = file.split(".")[0]
        # then save the image under out_path
        if self.mode == "cv2":
            cv2.imwrite(os.path.join(self.out_path, f"{handle}.jpg"), img)
        elif self.mode == "pil":
            img_.save(os.path.join(self.out_path, f"{handle}.jpg"))



def resize_images(in_path, out_path, max_size=512, mode="cv2", num_workers=-1, crop_borders=True, preprocessing='kdrd'):
    """
    Resizes images in a directory so that the longer side is <= max_size.
    It saves the images under out_path keeping the same name.
    num worker defines the number of processes (-1 -> use the number of cpu cores)
    """
    frame = not('ACRIMA' in in_path or 'RIM' in in_path or 'LAG' in in_path)

    # check if out_path is valid
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    # create a list of files in in_path
    img_exts = set(["jpg", "png", "jpeg", "tiff", "bmp", "gif", "ppm", "pgm", "tif", "svg", "JPG"])
    files = [f for f in sorted_nicely(os.listdir(in_path)) if os.path.isfile(os.path.join(in_path, f)) and f.split(".")[1] in img_exts]

    resizer = Resizer(mode, in_path, out_path, max_size, crop_borders, preprocessing, frame)

    if num_workers == 1:
        # iterate over the files
        for file in tqdm(files):
            resizer(file)
    else:
        if num_workers < 0:
            num_workers = mp.cpu_count()
        print(f"Start multiprocessing with {num_workers} workers.")

        with Pool(num_workers) as p:
            list(tqdm(p.imap(resizer, files), total=len(files)))

    print("Mask Fails: ", resizer.mask_fail_files, "#fails: ", len(resizer.mask_fail_files))
    print("Preprocessing Fails:", resizer.prep_fail_files, "#fails: ", len(resizer.prep_fail_files))



def check_and_create_out_dirs(out_paths):
    """
    expects a dict containing paths in to the train, val and test out_directories.
    It iterates over the contents and creates directories if necessary.
    """
    for top_level in out_paths:
        for nested_path in out_paths[top_level]:
            if not os.path.isdir(nested_path):
                os.makedirs(nested_path)



def train_test_val_split(data, train=0.8, val=0.1, test=0.1, ensure_not_empty=True):
    """
    expects a list and ratios.
    Returns three lists approximately conforming to the ratios.

    if ensure_not_empty
    then ensure all lists are not empty.
    """
    np.random.seed(42)
    assert train + val + test == 1.0

    if len(data) == 0:
        return [], [], []

    # get the number of images
    n = len(data)

    np.random.shuffle(data)

    # slice lists for train, val, and test
    tr = data[:int(train*n)]
    va = data[int(train*n):int(train*n)+int(test*n)]
    te = data[int(train*n)+int(test*n):]

    if ensure_not_empty:
        ns = [len(tr), len(va), len(te)]

        if not 0 in ns:
            return tr, va, te

        most_filled = np.argmax(ns)
        sets = [tr, va, te]

        for n, s in zip(ns, sets):
            if n == 0:
                s.append(sets[most_filled][-1])
                sets[most_filled] = sets[most_filled][0:-1]
        tr, va, te = sets
    return tr, va, te


def generate_out_names(n, prefix, leading_zeros=6, ext=".jpg"):
    """
    creates a list of output names given a prefix and the number n of files.
    It uses leading zeros
    """
    return [f"{prefix}{str(i).zfill(leading_zeros)}{ext}" for i in range(1,n+1)]


def sorted_nicely( l ):
    """
    https://stackoverflow.com/questions/2669059/how-to-sort-alpha-numeric-set-in-python
    Sort the given iterable in the way that humans expect.
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


def generate_overview_csv(out_path):
    """
    takes an outpath and generates a overview csv of the scheme:
    Dataset,Split,Class,Samples
    """
    dataset = os.path.split(out_path)[-1]

    csv_content = ["Dataset,Split,Class,Samples\n"]

    for split in ["train", "val", "test"]:
        for label in ["HE", "DR", "GL"]:
            samples = len(os.listdir(os.path.join(out_path, split, label)))
            csv_content.append( f"{dataset},{split},{label},{samples}\n" )

    with open(os.path.join(out_path, "data.csv"), "w") as f:
        f.writelines(csv_content)


def resize_img(img, max_size=512, mode='cv2'):
    """
    resizes and returns a single pil image to max_size
    """
    # find larger side
    idx = 0 if img.shape[0] > img.shape[1] else 1
    if img.shape[idx] <= max_size:
        return img
    else:
        # get width and height
        width = img.shape[1]
        height = img.shape[0]

        # calculate the new height and width
        if idx == 1:
            new_height = int(max_size*height/width)
            new_width = max_size
        elif idx == 0:
            new_height = max_size
            new_width = int(max_size*width/height)

        if mode == 'pil':
            new_img = img.resize((new_width, new_height))
        elif mode == 'cv2':
            new_img = cv2.resize(img, dsize=(new_width, new_height), interpolation=cv2.INTER_CUBIC)
        return new_img

def get_idx_split(list_len, split=(0.8, 0.1, 0.1)):
    """
    returns 3 lists of indices for splitting a data list default split:80/10/10 (trn, val, tst)
    """
    assert sum(split) == 1

    idx = np.arange(list_len)
    np.random.shuffle(idx)
    trn_idx = idx[:int(list_len*split[0])]
    val_idx = idx[int(list_len*split[0]):int(list_len*split[0])+int(list_len*split[1])]
    tst_idx = idx[int(list_len*split[0])+int(list_len*split[1]):]
    return trn_idx, val_idx, tst_idx

def load_images_from_path(path, resize=512):
    """
    loads and resizes images from a directory
    returning a list of the resized images.
    if resize = None / False / 0 -> dont resize
    """
    file_names = [f for f in sorted(os.listdir(path)) if os.path.isfile(os.path.join(path, f)) and f.split('.')[-1] in IMG_EXTENSIONS] # create a list of files in path
    return load_images_from_list(path, file_names, resize)

def load_images_from_list(base_path, file_names, resize=512, cut = False, tmp = True, cut_handle = False, crop_borders = True, preprocessing=''):
    """
    loads and resizes images from a list of filenames and the path to the corresponding directory
    returning a list of the resized images.
    if resize = None / False / 0 -> don't resize
    if cut is True cuts the image vertically in half and adds both images
    if tmp is True save the images to a tmp folder and return the tmp folder name instead of the image list (all in memory)
    if crop_borders crop away black borders of the image so we only have the actual image contents left
    preprocessing options: kdrd else no preprocessing will be done..
    """
    frame = not('ACRIMA' in base_path or 'RIM' in base_path)
    if tmp:
        if not os.path.isdir('.tmp'):
            os.makedirs('.tmp')
        subfolder_name = random.randint(0,100)
        while os.path.isdir(os.path.join('.tmp', str(subfolder_name))):
            subfolder_name += 1
        else:
            tmp_path = os.path.join('.tmp', str(subfolder_name))
            os.makedirs(tmp_path)
    images = [] 
    missing_files = []
    prep_fail_files = []
    mask_fail_files = []
    for file in tqdm(file_names):
        img_path = os.path.join(base_path, file)
        try:
            #img = Image.open(img_path)
            img = cv2.imread(img_path)
            if cut:
                img1 = img[:,img.shape[1]//2:,:]
                img2 = img[:,:img.shape[1]//2,:]
                if crop_borders:
                    img1 = crop_img_borders(img1)
                    img2 = crop_img_borders(img2)
                if resize:
                    img1 = resize_img(img1, resize)
                    img2 = resize_img(img2, resize)
                try:
                    img1, return_code = preprocess_img(img1, preprocessing, frame=frame)
                    img2, return_code = preprocess_img(img2, preprocessing, frame=frame)
                    if not return_code:
                        mask_fail_files.append(img_path)
                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    print(exc_tb.tb_lineno, e)
                    prep_fail_files.append(img_path)
                if tmp:
                    cv2.imwrite(os.path.join(tmp_path, f'{file}1.jpg'), img1)
                    cv2.imwrite(os.path.join(tmp_path, f'{file}2.jpg'), img2)
                    #img1.save(os.path.join(tmp_path, f'{file}1.jpg'))
                    #img2.save(os.path.join(tmp_path, f'{file}2.jpg'))
                else:
                    images.append(img1)
                    images.append(img2)
            else:
                if crop_borders:
                    img = crop_img_borders(img)
                if resize:
                    img = resize_img(img, resize)
                try:
                    img, return_code = preprocess_img(img, preprocessing, frame=frame)
                    if not return_code:
                        mask_fail_files.append(img_path)
                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    print(exc_tb.tb_lineno, e)
                    prep_fail_files.append(img_path)
                if tmp:
                    if cut_handle:
                        file_handle = file.split('/')[-1]
                        write_file = os.path.join(tmp_path, f'{file_handle}.jpg')
                    else:
                        write_file = os.path.join(tmp_path, f'{file}.jpg')
                    cv2.imwrite(write_file, img)    
                    #img.save(write_file)
                else:
                    images.append(img)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(exc_tb.tb_lineno, e)
            missing_files.append(img_path)
            continue
    if len(missing_files) > 0:
        print('Some files seem to be missing... Skipping them...:')
        print(missing_files)
    if len(prep_fail_files) > 0:
        print('For some files the preprocessing failed... Skipping preprocessing for them...:')
        print(prep_fail_files)
    if len(prep_fail_files) > 0:
        print('For some files the an ellipse could not be fit... used the simple watershed mask for them...:')
        print(mask_fail_files)
    if tmp:
        return tmp_path
    else:
        return images

def set_random_seed(seed=42):
    """
    sets the random seeds so we can get reproducable data splits
    """
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)
    np.random.seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 5)
    random.seed(seed + 6)
    cv2.setRNGSeed(seed + 7)
    print(f'Seed was set to {seed}')


def save_converted(data_dict, path, tmp=True):
    """
    takes a data_dict with keynames that represent the target folder structure and lists of images as *leaf* elements and saves the images is to the corresponding folders
    path gives the path to the folder to which the folder structure is to be saved
    e.g. {ACRIMA: {train:{HE:[i0.....i100], DR:[], GL:[]}, test:{HE:[], DR:[], GL:[]}, val:{HE:[], DR:[], GL:[]}}}
    if tmp move the dat from tmp folder to destination folder else all in memory
    """

    for k1 in data_dict.keys():
        #layer 1: dataset name
        ds_path = os.path.join(path, k1)
        #print(ds_path)
        if not os.path.isdir(ds_path):
            os.makedirs(ds_path)
        for k2 in data_dict[k1].keys():
            #layer 2: trn/val/tst
            type_path = os.path.join(ds_path, k2)
            #print(type_path)
            if not os.path.isdir(type_path):
                os.makedirs(type_path)
            for k3, images in data_dict[k1][k2].items():
                #layer 3: classes and their data
                class_path = os.path.join(type_path, k3)
                #print(class_path)
                if not os.path.isdir(class_path):
                    os.makedirs(class_path)
                format_str = f'06d' #{len(str(len(num_imgs)))} for variable leading zeros. Let's just use 6 here
                if tmp:
                    img_list = sorted(os.listdir(images))
                    for i, img_file in enumerate(img_list):
                        handle = f'{k1}_{k2}_{k3}_{i:{format_str}}'
                        os.replace(os.path.join(images, img_file), os.path.join(class_path, f'{handle}.jpg'))
                else:
                    for i, img in enumerate(images):
                        handle = f'{k1}_{k2}_{k3}_{i:{format_str}}'
                        cv2.imwrite(os.path.join(class_path, f'{handle}.jpg'), img)
    shutil.rmtree('.tmp')

def write_overview_csv(path):
    """
    creates a csv file that includes the amount of data that is available for Train/test/val in the corresponding HE/DR/GL sets
    needs the path to the processed dataset folder
    """
    dataset = os.path.basename(os.path.normpath(path))
    with open(os.path.join(path, 'data.csv'), 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(['Dataset', 'Split', 'Class', 'Samples'])
        l1_folders = [name for name in sorted(os.listdir(path)) if name in ['test', 'train', 'val']]
        for layer1 in l1_folders:
            l2_folders = [name for name in sorted(os.listdir(os.path.join(path, layer1))) if name in ['HE', 'DR', 'GL']]
            for layer2 in l2_folders:
                num_imgs = len(sorted(os.listdir(os.path.join(path, layer1, layer2))))
                csv_writer.writerow([dataset, layer1, layer2, num_imgs])


def preprocess_img(img, ptype = 'kdrd', **kwargs):
    """ preprocesses an image depending on ptype

    Args:
        img np array (opencv img): Image to process 
        ptype (str, optional): type of preprocessing (only kdrd implemented yet, else no further preprocessing is done.). Defaults to 'kdrd'.

    Returns:
        np array img, returncode int: processed img; returncode 0 if something went wrong, 1 if everything is okay
    """
    if ptype == 'kdrd':
        return kdrd_preprocessing(img, frame=kwargs['frame'])
    else:
        return img, 1

def kdrd_preprocessing(img, blur_kernel=(0,0), blur_sigma=10, mixing_strength=4, frame=True):
    """preprocessing like in https://storage.googleapis.com/kaggle-forum-message-attachments/88655/2795/competitionreport.pdf

    Args:
        img np array (opencv img): Image to process 
        blur_kernel (tuple, optional): size of the blur kernel, if (0,0) it will be automatically determined. Defaults to (0,0).
        blur_sigma (int, optional): blur variance. Defaults to 10.
        mixing_strength (int, optional): strength of the signal when adding blur to the image (stronger contrasts and harder veins for higher values). Defaults to 4.
        frame (bool, optional): if True add a frame so that only the main features of the eye are to be seen. Defaults to True.

    Returns:
        np array img, returncode int: processed img; returncode 0 if something went wrong (ellipse masking), 1 if everything is okay
    """
    return_code = 1
    blur = cv2.GaussianBlur(img, blur_kernel, blur_sigma)
    imgc = cv2.addWeighted(img, mixing_strength, blur, -mixing_strength, 128) # mixing_strength * img - mixing_strength * img + 128
    if frame:
        _, mask, return_code = find_mask_watershed(img)
        imgc = imgc * mask + 128 * (1-mask)
    return imgc, return_code

def find_mask_watershed(img, fitEll=True):
    """Creates a mask to build a frame so only the eye is seen in the image with cv2.watershed -> if possible fitting an ellipse around it

    Args:
        img ([np array (opencv)]): Image to get the mask for

    Returns:
        watershed mask, ellipse mask, returncode: returncode 0 if masking with an ellipse failed else 1 (successful)
    """
    return_code = 1
    imgc = img.copy()
    gray = cv2.cvtColor(imgc,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    markers = cv2.watershed(imgc,markers)
    imgc[markers == -1] = [255,0,0]

    mask = np.zeros(markers.shape)
    mask[markers == markers[markers.shape[0]//2][markers.shape[1]//2]] = 128 #around that marker we will now lay a ellipse mask

    ell_mask = None
    if fitEll:
        gray = mask.astype('uint8')

        _, thresh_gray = cv2.threshold(gray, thresh=5, maxval=255, type=cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh_gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) > 4:
                ellipse = list(cv2.fitEllipse(cnt))
                ellipse[1] = tuple(np.array(ellipse[1])*0.96)
                ellipse = tuple(ellipse)
                ell_length1 = ellipse[1][0]
                ell_length2 = ellipse[1][1]
                #print('Ell:', ellipse[1], imgc.shape)
                if ell_length1 > imgc.shape[1]*0.5 and ell_length1 < imgc.shape[1]*2 and ell_length2 > imgc.shape[0]*0.7 and ell_length2 < imgc.shape[0]*2:
                    ell_mask = np.zeros(imgc.shape)
                    cv2.ellipse(ell_mask, ellipse, color=(1, 1, 1), thickness=-1)
                    break
            

    w_mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)/128
    if ell_mask is None:
        #print('Could not fit an ellipse to the image. Using Watershed mask instead')
        return_code = 0
        ell_mask = w_mask
    return w_mask, ell_mask, return_code

def find_mask_ellipse(img, thresh=0):
    imgc = img.copy()
    gray = cv2.cvtColor(imgc, cv2.COLOR_BGR2GRAY)
    cnt = []
    ell_length1 = 0
    ell_length2 = 0
    while len(cnt) < 5 or ell_length1 < imgc.shape[1]*0.5 or ell_length1 > imgc.shape[1]*3 or ell_length2 < imgc.shape[0]*0.5 or ell_length2 > imgc.shape[0]*3:
        ell_length1 = 0
        ell_length2 = 0
        _, thresh_gray = cv2.threshold(gray, thresh=thresh, maxval=255, type=cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh_gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        #print('Thresh:',thresh)
        thresh += 5
        if len(cnt) > 4:
            ellipse = list(cv2.fitEllipse(cnt))
            ellipse[1] = tuple(np.array(ellipse[1])*0.96)
            ellipse = tuple(ellipse)
            #print(thresh)
            ell_length1 = ellipse[1][0]
            ell_length2 = ellipse[1][1]
            print('Ell:', ellipse[1], imgc.shape)
    ell = np.zeros(imgc.shape)
    cv2.ellipse(ell, ellipse, color=(1, 1, 1), thickness=-1)
    imgc = imgc * ell + 128 * (1-ell)
    return imgc, ell

def crop_img_borders(img):
    """crops one colored (black) borders from an image so main part of the image contains the eye scan

    Args:
        img (np array (opencv img)): Open CV standard img

    Returns:
        [np array (opencv img)]: cropped image 
    """
    imgc = img.copy() 
    m, _, _ = find_mask_watershed(imgc, fitEll=False)
    gray = m[:,:,0].astype('uint8')*128
    #gray = cv2.cvtColor(imgc, cv2.COLOR_BGR2GRAY)
    _, thresh_gray = cv2.threshold(gray, thresh=5, maxval=255, type=cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh_gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # take the contour with the largest area
    cnt = max(contours, key=cv2.contourArea)

    x,y,w,h = cv2.boundingRect(cnt)
    crop = imgc[y:y+h,x:x+w]
    
    return crop