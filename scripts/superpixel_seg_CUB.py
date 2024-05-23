import os
import cv2
import pickle
import glob
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed, flood
from skimage.measure import regionprops


if __name__ == '__main__':
    image_type = 'CUB'
    cub_path = r'../data\CUB\CUB_200_2011\cropped_image'
    classes = os.listdir(cub_path)
    bbox_path = r'../data\CUB\CUB_200_2011\bounding_boxes.txt'
    imageid_path = r'../data\CUB\CUB_200_2011\images.txt'
    save_dir = r'../data\CUB\part_bbox'

    all_markers = [9, 15, 25, 50]
    all_compactness = [0, 0.001]

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

    imagename2bbox = {}

    cnt = 0
    for c in classes:
        image_dir = os.path.join(cub_path, c)
        image_names = os.listdir(image_dir)
        for name in image_names:
            ori_name = "_".join(name.split("_")[1:])
            if ori_name not in imagename2id:
                raise Exception
            imageid = imagename2id[ori_name]
            if imageid not in id2bbox:
                raise Exception
            bbox = id2bbox[imageid]
            x, y, w, h = bbox
            x2, y2 = x + w, y + h

            image_path = os.path.join(image_dir, name)
            # img = np.array(Image.open(image_path))
            img = cv2.imread(image_path)

            gradient = sobel(rgb2gray(img))

            search_dict = {}
            for markers in all_markers:
                for compactness in all_compactness:
                    segments_watershed = watershed(gradient, markers=markers, compactness=compactness)
                    props = regionprops(segments_watershed)
                    bboxes = [prop.bbox for prop in props]

                    new_bboxes = []
                    for _bbox in bboxes:
                        _y, _x, _y1, _x1 = _bbox
                        new_bbox = [_x, _y, _x1, _y1]
                        new_bbox[0] += x
                        new_bbox[1] += y
                        new_bbox[2] += x
                        new_bbox[3] += y
                        new_bboxes.append(new_bbox)
                    key = f'markers:{markers},compactness:{compactness}'
                    search_dict[key] = new_bboxes
            img_key = (image_path.replace("\\", "/").replace("cropped_image", "images")
                       .replace("cropped_", ""))
            imagename2bbox[img_key] = search_dict
            print(f"处理完了图片 {cnt}")
            cnt += 1

    with open(r'../data\CUB\imagename2bbox.pkl', 'wb') as f:
        pickle.dump(imagename2bbox, f)
