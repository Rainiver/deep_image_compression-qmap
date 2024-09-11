import os
import glob
import cv2

GLOB_PATTERN = '/mnt/lustre/share/DSK/datasets/mscoco2017/val2017/*.jpg'
OUTPUT_PATH = 'flist.txt'


flist = glob.glob(GLOB_PATTERN)

flist = sorted(list(flist))

meta = []
skipped = []
for i, f in enumerate(sorted(flist)):
    try:
        if ' ' in f:
            raise ValueError('unexpected space in filename: ' + f)
        img = cv2.imread(f)
        if img is None or img.shape is None:
            raise ValueError('broken image: ' + f)
    except:
        print('skip', f)
        skipped.append(f)
        continue
    m = os.path.abspath(f), img.shape[0], img.shape[1], 0.0
    m = ' '.join(map(str,m))
    print(i, '/', len(flist), m)
    meta.append(m)

with open(OUTPUT_PATH, 'w') as f:
    f.write('\n'.join(meta))

print('skipped:', skipped)
