import torch
import logging
import numpy as np
import cv2
import json
import time
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as dataset
from torchvision import transforms
from .data_utils import remove_duplicate, rotate_bbox, rotate_keypoints
from .transform import BaseTransform
from utils.distributed_utils import get_rank, get_world_size

logger = logging.getLogger('global')


def build_roiloader(cfg, training, distributed):
    phase = "train"
    if not training:
        cfg["batch_size"] = 1
        phase = "test"
    transform_fn = BaseTransform(cfg['data_augment'][phase], training)
    logger.info('building dataset from: {}'.format(cfg['meta_file_list']))
    dataset = ROIDataset(
        cfg['meta_file_list'],
        transform_fn=transform_fn,
        color_image=cfg.get('color_image', True),
        for_training=training,
        kept=cfg.get('kept', None),
    )
    dataset_size = len(dataset)
    index = list(range(dataset_size))
    if training:
        np.random.seed(int(time.time()) % 100)
        np.random.shuffle(index)
    rank = get_rank()
    world_size = get_world_size()
    data_l = rank * dataset_size // world_size
    data_r = data_l + dataset_size // world_size
    index = index[data_l:data_r]
    logger.info(f'rank: {rank}, data_sum: {len(index)}')
    dataset_after = torch.utils.data.SubsetRandomSampler(index)
    if not training:
        dataset_after = None
    batch_size = cfg['batch_size'] if training else 1
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=training,
                        num_workers=cfg['workers'],
                        pin_memory=True,
                        sampler=dataset_after)
    return loader


class ROIDataset(dataset):
    def __init__(self,
                 meta_file_list,
                 transform_fn=None,
                 color_image=True,
                 for_training=True,
                 kept=None):
        super(ROIDataset, self).__init__()
        self.for_training = for_training
        self.color_image = color_image
        self.transform_fn = transform_fn
        self.initialized = False

        logger.debug('not using server dataset loader...')
        self.metas = []
        for i, par_file_path in enumerate(meta_file_list):
            meta = parse_detect_meta(i, par_file_path)
            self.metas.extend(meta)
            logger.debug('Partition {} {} loaded.'.format(i, par_file_path))
        if not self.for_training:
            self.metas = remove_duplicate(self.metas)
        if kept is not None:
            self.metas = self.metas[0:kept]
        self.num = len(self.metas)
        logger.debug('Total number of images: %d' % self.num)
        self.to_tensor = transforms.ToTensor()

    def read_img_by_OpenCV(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            logger.info(img_path)
        # if self.color_image:
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # else:
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def __len__(self):
        return self.num

    def _fake_zero_data(self, *size):
        return np.zeros(size)

    def tensor2numpy(self, x):
        if x is None:
            return x
        if torch.is_tensor(x):
            return x.cpu().numpy()
        if isinstance(x, list):
            x = [_.cpu().numpy() if torch.is_tensor(_) else _ for _ in x]
        return x

    def __getitem__(self, idx):
        meta = self.metas[idx].to_dict()
        img_path = meta['path']
        bboxes = np.array(meta['gts'], dtype=int)
        if bboxes.shape[0] == 0:
            bboxes = np.ones((0, 4))
        labels = np.array(meta['labels'], dtype=int)
        ignores = np.array(meta['igns'], dtype=int)
        if ignores.shape[0] == 0:
            ignores = np.ones((0, 4))
        bboxes = bboxes.astype(np.float32)
        ignores = ignores.astype(np.float32)
        labels = labels.astype(np.float32)

        img = self.read_img_by_OpenCV(img_path)
        if img is None:
            logger.critical("Cannot load image:{}".format(img_path))
        if ignores.shape[0] != 0 and ignores is not None:
            ig_bboxes = np.hstack([ignores, -np.ones(ignores.shape[0])[:, np.newaxis]])
            targets = np.concatenate((np.concatenate((bboxes, labels[:, np.newaxis]), axis=1), ig_bboxes))
        else:
            targets = np.concatenate((bboxes, labels[:, np.newaxis]), axis=1)
        ig_bboxes = targets[targets[:, 4] == -1][:, :4]
        gt_bboxes = targets[targets[:, 4] != -1]
        image_h, image_w = img.shape[:2]
        roi = np.zeros((image_h, image_w, 1), dtype='float32')
        for box in gt_bboxes:
            xmin, ymin, xmax, ymax = [int(x) for x in box[1:]]
            roi[ymin:ymax, xmin:xmax, 0] = 1
        for box in ig_bboxes:
            xmin, ymin, xmax, ymax = [int(x) for x in box]
            roi[ymin:ymax, xmin:xmax, 0] = 0
        # to tensor
        img, roi = self.transform_fn((img, roi))
        if self.image_engine == "PIL":
            img = self.to_tensor(img)
        else:
            if len(img.shape) == 2:
                img = self.to_tensor(np.expand_dims(img.astype(np.uint8), axis=2))
            else:
                img = self.to_tensor(img)
        roi = self.to_tensor(roi)
        inputs = {
            'img': img,
            'roi': roi
        }
        return inputs


def parse_detect_meta(partition, meta_file_path, angle=0):
    metas = []
    with open(meta_file_path, encoding="utf-8") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        img_igs = []
        img_gts = []
        img_labels = []
        img_keypoints = []
        path = lines[i + 1].rstrip()
        img_info = lines[i + 2].strip().split()
        img_height = float(img_info[1])
        img_width = float(img_info[2])
        img_ig_size = int(lines[i + 3])
        i += 4
        for j in range(img_ig_size):
            sp = lines[i + j].split()
            igs = [float(sp[0]), float(sp[1]), float(sp[2]), float(sp[3])]
            igs = rotate_bbox(igs, (img_height, img_width), angle)
            img_igs.append(igs)
        i += img_ig_size
        img_gt_size = int(lines[i])
        i += 1
        for j in range(img_gt_size):
            sp = lines[i + j].split()
            img_labels.append(float(sp[0]))
            gts = [
                float(sp[1]),
                float(sp[2]),
                float(sp[3]),
                float(sp[4]),
            ]
            gts = rotate_bbox(gts, (img_height, img_width), angle)
            img_gts.append(gts)
            # if keypoints are provided
            if len(sp) > 5:
                keypoints = []
                assert (len(sp) - 5) % 2 == 0, "keypoints data error!!!"
                for k in range(5, len(sp)):
                    keypoints.append(float(sp[k]))
                rotate_keypoints(keypoints, (img_height, img_width), angle)
                img_keypoints.append(keypoints)
        i += img_gt_size
        metas.append(ImageData(partition, path, img_height, img_width, img_gts, img_labels, img_igs, img_keypoints))
    return metas


class ImageData():
    __slots__ = ['partition', 'path', 'height', 'width', 'gts', 'labels', 'igns', 'keypoints']

    def __init__(self, partition, path, height, width, gts, labels, igns, keypoints=[]):
        self.partition = partition
        self.path = path
        self.height = height
        self.width = width
        self.gts = gts
        self.igns = igns
        self.labels = labels
        self.keypoints = keypoints

    def to_json(self):
        res = {}
        res['partition'] = self.partition
        res['path'] = self.path
        res['height'] = self.height
        res['width'] = self.width
        res['gts'] = self.gts
        res['igns'] = self.igns
        res['labels'] = self.labels
        if len(self.keypoints) > 0:
            res['keypoints'] = self.keypoints
        return json.dumps(res)

    def to_dict(self):
        res = {}
        res['partition'] = self.partition
        res['path'] = self.path
        res['height'] = self.height
        res['width'] = self.width
        res['gts'] = self.gts
        res['igns'] = self.igns
        res['labels'] = self.labels
        if len(self.keypoints) > 0:
            res['keypoints'] = self.keypoints
        return res
