import numpy as np
import converters.utils_custom as utils
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description='Convert RIGA dataset to consistent sizes and labels.')
    os.chdir('..')
    parser.add_argument('--in_path', default=os.path.join(os.getcwd(), 'dataset', 'raw', 'RIGA'), type=str,
                        help='Path to the raw dataset')
    parser.add_argument('--out_path', default=os.path.join(os.getcwd(), 'dataset', 'preprocessed'), type=str,
                        help='Path to where you want the converted dataset to be saved.')
    parser.add_argument('--preprocessing', default='', type=str,
                        help='Kind of preprocessing that should be applied on the images. Currently available: kdrd else there will be no preprocessing')
    args = parser.parse_args()

    BASE_PATH = args.in_path
    PROCESSED_PATH = args.out_path
    # print(BASE_PATH, PROCESSED_PATH)

    SUB_PATH_PREFIX = os.path.join('BinRushedcorrected', 'BinRushed')
    SUB_PATHS = [os.path.join(SUB_PATH_PREFIX, 'BinRushed1-Corrected'), os.path.join(SUB_PATH_PREFIX, 'BinRushed2'),
                 os.path.join(SUB_PATH_PREFIX, 'BinRushed3'), os.path.join(SUB_PATH_PREFIX, 'BinRushed4'),
                 os.path.join('Magrabia', 'Magrabia', 'MagrabiaMale'),
                 os.path.join('Magrabia', 'Magrabia', 'MagrabiFemale'), os.path.join('MESSIDOR', 'MESSIDOR')]

    GL_files = []
    for sub_path in SUB_PATHS:
        all_imgs = sorted(os.listdir(os.path.join(BASE_PATH, sub_path)))
        primes = []
        for img in all_imgs:
            if 'prime' in img:
                primes.append(os.path.join(sub_path, img))
        GL_files += primes

    GL_files = np.sort(np.array(GL_files))
    utils.set_random_seed()
    GL_trn_idx, GL_val_idx, GL_tst_idx = utils.get_idx_split(len(GL_files))
    GL_trn_files = GL_files[GL_trn_idx]
    GL_val_files = GL_files[GL_val_idx]
    GL_tst_files = GL_files[GL_tst_idx]

    RIGA_dict = {'RIGA':
        {'train': {
            'GL': utils.load_images_from_list(BASE_PATH, GL_trn_files, cut_handle=True,
                                              preprocessing=args.preprocessing)
        },
            'val': {
                'GL': utils.load_images_from_list(BASE_PATH, GL_val_files, cut_handle=True,
                                                  preprocessing=args.preprocessing)
            },
            'test': {
                'GL': utils.load_images_from_list(BASE_PATH, GL_tst_files, cut_handle=True,
                                                  preprocessing=args.preprocessing)
            }
        }}

    utils.save_converted(RIGA_dict, PROCESSED_PATH)
    utils.write_overview_csv(PROCESSED_PATH)


if __name__ == '__main__':
    main()
