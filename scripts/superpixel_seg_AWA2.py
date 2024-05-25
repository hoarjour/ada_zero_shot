import os
import cv2
import pickle
import glob
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed, flood
from skimage.measure import regionprops


def downsample_image(image, scale_factor):
    height, width = image.shape[:2]
    new_dimensions = (int(width * scale_factor), int(height * scale_factor))
    return cv2.resize(image, new_dimensions, interpolation=cv2.INTER_LINEAR)


def upsample_bbox(bbox, scale_factor):
    return [int(coord / scale_factor) for coord in bbox]


if __name__ == '__main__':
    image_type = 'AWA2'
    all_img_paths = glob.glob('../data/AWA2/AWA2-data/Animals_with_Attributes2/JPEGImages/*/*.jpg')
    all_img_paths = [x.replace("\\", "/") for x in all_img_paths]

    all_markers = [9, 15, 25, 50]
    all_compactness = [0, 0.001]
    imagename2bbox = {}

    scale_factor = 0.5
    cnt = 0
    for p in all_img_paths:
        img = cv2.imread(p)
        try:
            downsampled_img = downsample_image(img, scale_factor)
            gradient = sobel(rgb2gray(downsampled_img))
        except Exception:
            print(f"处理图片{p} 失败，sobel阶段")
            continue

        search_dict = {}
        flag = 0
        for markers in all_markers:
            for compactness in all_compactness:
                try:
                    segments_watershed = watershed(gradient, markers=markers, compactness=compactness)
                    props = regionprops(segments_watershed)
                except Exception:
                    print(f"处理图片{p} 失败，watershed阶段")
                    flag = 1
                    continue

                bboxes = [prop.bbox for prop in props]
                new_bboxes = []
                for _bbox in bboxes:
                    _y, _x, _y1, _x1 = _bbox
                    new_bbox = [_x, _y, _x1, _y1]
                    # 将降采样后的bbox映射回原始图像尺寸
                    upsampled_bbox = upsample_bbox(new_bbox, scale_factor)
                    new_bboxes.append(upsampled_bbox)
                key = f'markers:{markers},compactness:{compactness}'
                search_dict[key] = new_bboxes
        if flag:
            continue
        imagename2bbox[p] = search_dict
        print(f"处理完了图片 {cnt}")
        cnt += 1

    with open(r'../data/AWA2/imagename2bbox.pkl', 'wb') as f:
        pickle.dump(imagename2bbox, f)
