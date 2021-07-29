import numpy as np
import converters.utils_custom as utils
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description='Convert RIM dataset to consistent sizes and labels.')
    os.chdir('..')
    parser.add_argument('--in_path', default=os.path.join(os.getcwd(), 'dataset', 'raw', 'RIM'), type=str, help='Path to the raw dataset')
    parser.add_argument('--out_path',  default=os.path.join(os.getcwd(), 'dataset', 'preprocessed', 'RIM'), type=str, help='Path to where you want the converted dataset to be saved.')
    parser.add_argument('--preprocessing', default='', type=str, help='Kind of preprocessing that should be applied on the images. Currently available: kdrd else there will be no preprocessing')

    args = parser.parse_args()

    BASE_PATH = args.in_path
    PROCESSED_PATH = args.out_path
    #print(BASE_PATH, DATA_PATH, PROCESSED_PATH)

    HE_PATH = os.path.join(BASE_PATH, 'Healthy', 'Stereo Images')
    GL_PATH = os.path.join(BASE_PATH, 'Glaucoma and suspects', 'Stereo Images')



    HE_files = np.sort(np.array([file for file in os.listdir(HE_PATH) if not file.startswith('.')]))
    GL_files = np.sort(np.array([file for file in os.listdir(GL_PATH) if not file.startswith('.')]))

    utils.set_random_seed()
    HE_trn_idx, HE_val_idx, HE_tst_idx = utils.get_idx_split(len(HE_files))
    HE_trn_files = HE_files[HE_trn_idx]
    HE_val_files = HE_files[HE_val_idx]
    HE_tst_files = HE_files[HE_tst_idx]


    GL_trn_idx, GL_val_idx, GL_tst_idx = utils.get_idx_split(len(GL_files))
    GL_trn_files = GL_files[GL_trn_idx]
    GL_val_files = GL_files[GL_val_idx]
    GL_tst_files = GL_files[GL_tst_idx]

    RIM_dict = {'RIM':
                 {'train': {
                     'HE':utils.load_images_from_list(HE_PATH, HE_trn_files, cut=True, preprocessing=args.preprocessing),
                     'GL':utils.load_images_from_list(GL_PATH, GL_trn_files, cut=True, preprocessing=args.preprocessing)
                 },
                  'val': {
                     'HE':utils.load_images_from_list(HE_PATH, HE_val_files, cut=True, preprocessing=args.preprocessing),
                     'GL':utils.load_images_from_list(GL_PATH, GL_val_files, cut=True, preprocessing=args.preprocessing)
                  },
                  'test': {
                     'HE':utils.load_images_from_list(HE_PATH, HE_tst_files, cut=True, preprocessing=args.preprocessing),
                     'GL':utils.load_images_from_list(GL_PATH, GL_tst_files, cut=True, preprocessing=args.preprocessing)
                  }
                 }}


    utils.save_converted(RIM_dict, PROCESSED_PATH)
    utils.write_overview_csv(os.path.join(PROCESSED_PATH, 'RIM'))




if __name__ == '__main__':
    main()
