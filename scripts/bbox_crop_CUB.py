import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import cv2


if __name__ == '__main__':
    image_type = 'CUB'
    cub_path = r'E:\PycharmProjects\ada_zero_shot_clf\data\CUB\CUB_200_2011\images'
    classes = os.listdir(cub_path)
    bbox_path = r'E:\PycharmProjects\ada_zero_shot_clf\data\CUB\CUB_200_2011\bounding_boxes.txt'
    imageid_path = r'E:\PycharmProjects\ada_zero_shot_clf\data\CUB\CUB_200_2011\images.txt'
    save_dir = r'E:\PycharmProjects\ada_zero_shot_clf\data\CUB\CUB_200_2011\cropped_image'


    with open(bbox_path, 'r') as f:
        id_bbox = f.readlines()
    with open(imageid_path, 'r') as f:
        id_imagename = f.readlines()
    id2bbox = {}
    imagename2id = {}

    for x in id_imagename:
        imageid, imagename = x.strip().split()
        imagename = imagename.split("/")[-1]
        imagename2id[imagename] = imageid
    for i in id_bbox:
        imageid, x, y, w, h = i.strip().split()
        x, y, w, h = int(eval(x)), int(eval(y)), int(eval(w)), int(eval(h))
        id2bbox[imageid] = [x, y, w, h]

    cnt = 0
    for c in classes:
        image_dir = os.path.join(cub_path, c)
        image_names = os.listdir(image_dir)
        for name in image_names:
            if name not in imagename2id:
                raise Exception
            imageid = imagename2id[name]
            if imageid not in id2bbox:
                raise Exception
            bbox = id2bbox[imageid]
            x, y, w, h = bbox
            x2, y2 = x + w, y + h

            image_path = os.path.join(image_dir, name)
            # img = np.array(Image.open(image_path))
            img = cv2.imread(image_path)
            new_img = img[y:y2, x:x2, :]

            img_save_dir = os.path.join(save_dir, c)
            os.makedirs(img_save_dir, exist_ok=True)
            img_save_path = os.path.join(img_save_dir, f'cropped_{name}')
            cv2.imwrite(img_save_path, new_img)
            # plt.imshow(new_img)
            # plt.savefig(img_save_path)
            print(f"Saving image {name} {cnt}")
            cnt += 1
