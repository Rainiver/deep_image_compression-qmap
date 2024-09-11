import os
import cv2
from tqdm import tqdm
from glob import glob


ROOT = "/mnt/lustre/share/wangyuan/datasets"


DATA_DIR_LIST = {
    "OPEN_IMAGE_TRAIN": f"{ROOT}/openimage/train_oi",  # 337967 images
    "OPEN_IMAGE_VAL": f"{ROOT}/openimage/val_oi",  # 36298 images
    "OPEN_IMAGE_TEST": f"{ROOT}/openimage/val_oi_500_r",  # 500 images

    "CLIC2020_MOBILE_TRAIN": f"{ROOT}/CLIC2020/mobile_train_2020/train",  # 1048 images
    "CLIC2020_PRO_TRAIN": f"{ROOT}/CLIC2020/professional_train_2020/train",  # 585 images
    "CLIC2020_MOBILE_TEST": f"{ROOT}/CLIC2020/CLIC2020Mobile_test/mobile",  # 178 images
    "CLIC2020_PRO_TEST": f"{ROOT}/CLIC2020/CLIC2020Professional_test/professional",  # 250 images

    "IMAGE64_TRAIN": f"{ROOT}/imagenet64/train_64x64",  # 1281149 images
    "IMAGE64_VAL": f"{ROOT}/imagenet64/valid_64x64",  # 49999 images

    "DIV2K_VAL_HR": f"{ROOT}/DIV2K_valid_HR",  # 100 images
}


META_PATH_LIST = {
    "OPEN_IMAGE_TRAIN": f"{ROOT}/data_list/open_image_train_oi.txt",  # 337967 images
    "OPEN_IMAGE_VAL": f"{ROOT}/data_list/open_image_val_oi.txt",
    "OPEN_IMAGE_TEST": f"{ROOT}/data_list/open_image_val_oi_500_r.txt",

    "CLIC2020_MOBILE_TRAIN": f"{ROOT}/data_list/clic2020_mobile_train.txt",
    "CLIC2020_PRO_TRAIN": f"{ROOT}/data_list/clic2020_professional_train.txt",
    "CLIC2020_MOBILE_TEST": f"{ROOT}/data_list/clic2020_mobile_test.txt",
    "CLIC2020_PRO_TEST": f"{ROOT}/data_list/clic2020_professional_test.txt",

    "IMAGE64_TRAIN": f"{ROOT}/data_list/imagenet64_train.txt",
    "IMAGE64_VAL": f"{ROOT}/data_list/imagenet64_valid.txt",

    "DIV2K_VAL_HR": f"{ROOT}/data_list/div2k_valid_hr.txt"
}


if __name__ == "__main__":
    for _data in DATA_DIR_LIST.keys():
        _path = DATA_DIR_LIST[_data]
        image_list = glob(f"{_path}/*.png")
        print(f"processing {_data}, {len(image_list)} to be processed...")
        with open(META_PATH_LIST[_data], "w") as f:
            for fn in tqdm(image_list):
                # size = os.path.getsize(fn)
                # h, w, _ = cv2.imread(fn).shape
                # record = f"{fn} {h} {w} {size * 8 / h / w:.6f} \n"
                record = f"{fn} -1 -1 -1\n"
                f.write(record)
