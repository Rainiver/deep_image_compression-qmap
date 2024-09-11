def rotate_bbox(bbox, img_size, angle):
    h, w = img_size
    x1, y1, x2, y2 = bbox
    if angle == 0:
        return [x1, y1, x2, y2]
    if angle == 90:
        return [h - 1 - y2, x1, h - 1 - y1, x2]
    elif angle == 180:
        return [w - 1 - x2, h - 1 - y2, w - 1 - x1, h - 1 - y1]
    elif angle == 270:
        return [y1, w - 1 - x2, y2, w - 1 - x1]
    else:
        raise Exception("Rotate angle not supported")


def rotate_keypoints(keypoints, img_size, angle):
    h, w = img_size
    if angle == 0:
        return
    for i in range(0, len(keypoints), 2):
        keypoints[i:i + 2] = _rotate_xy(keypoints[i:i + 2], w, h, angle)


def _rotate_xy(xy, w, h, angle=0):
    x, y = xy[0], xy[1]
    if angle == 0:
        return xy
    elif angle == 90:
        return [h - 1 - y, x]
    elif angle == 180:
        return [w - 1 - x, h - 1 - y]
    elif angle == 270:
        return [y, w - 1 - x]
    else:
        raise Exception("Rotate angle not supported")


def read_img_by_PIL(path, color_image):
    return None


def remove_duplicate(metas):
    single_metas = list()
    single_set = set()
    for meta in metas:
        if meta.path not in single_set:
            single_metas.append(meta)
        single_set.add(meta.path)
    return single_metas
