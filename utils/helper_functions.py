import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pycocotools.mask as mask_utils
from vst_main.Testing import VST_test_once


def convert_to_json_serializable(data):
    if isinstance(data, torch.Tensor):
        return data.cpu().detach().numpy().tolist()
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {key: convert_to_json_serializable(value) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        return [convert_to_json_serializable(item) for item in data]
    else:
        return data

# this function comes from the SAM repository https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


# this function is taken from the SAM repository https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


# adapted from https://github.com/talshaharabany/AutoSAM/blob/main/train.py
def get_iou(predict, target):
    predict = predict + 1
    target = target + 1
    tp = np.sum(((predict == 2) * (target == 2)) * (target > 0))
    fp = np.sum(((predict == 2) * (target == 1)) * (target > 0))
    fn = np.sum(((predict == 1) * (target == 2)) * (target > 0))
    ji = float(np.nan_to_num(tp / (tp + fp + fn)))
    return ji


# adapted from https://github.com/talshaharabany/AutoSAM/blob/main/train.py
def open_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
    a = os.listdir(path)
    if a:
        a = sorted(a,key=lambda student: int(student.split('pu')[1]))
        nextfolder = int(a[-1].split('pu')[1])+1
    else:
        nextfolder=1
    return str(nextfolder)

def polys_to_mask(rle):
    mask = mask_utils.decode(rle)
    return mask

# Point Augmentation strategies from SAMAug
# the following point selection methods come from https://github.com/yhydhx/SAMAug
# SAMAug provides point augmentation methods to augment point prompts for standard SAM
"""Random Sample Point"""
def get_random_point(mask):
  indices = np.argwhere(mask==True)

  random_point = indices[np.random.choice(list(range(len(indices))))]
  random_point = [random_point[1], random_point[0]]
  return random_point

"""Max Entropy Point"""
def image_entropy(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate the histogram
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    # Normalize the histogram
    hist /= hist.sum()
    # Calculate the entropy
    entropy = -np.sum(hist * np.log2(hist + np.finfo(float).eps))

    return entropy

def calculate_image_entroph(img1, img2):
    # Calculate the entropy for each image
    entropy1 = image_entropy(img1)
    # print(img2)
    try:
        entropy2 = image_entropy(img2)
    except:
        entropy2 = 0
    # Compute the entropy between the two images
    entropy_diff = abs(entropy1 - entropy2)
    # print("Entropy Difference:", entropy_diff)
    return entropy_diff

def select_grid(image, center_point, grid_size):
    (img_h, img_w, _) = image.shape

    # Extract the coordinates of the center point
    x, y = center_point
    x = int(np.floor(x))
    y = int(np.floor(y))
    # Calculate the top-left corner coordinates of the grid
    top_left_x = x - (grid_size // 2) if x - (grid_size // 2) > 0 else 0
    top_left_y = y - (grid_size // 2) if y - (grid_size // 2) > 0 else 0
    bottom_right_x = top_left_x + grid_size if top_left_x + grid_size < img_w else img_w
    bottom_right_y = top_left_y + grid_size if top_left_y + grid_size < img_h else img_h

    # Extract the grid from the image
    grid = image[top_left_y: bottom_right_y, top_left_x: bottom_right_x]

    return grid


def get_entropy_points(input_point,mask,image):
    max_entropy_point = [0,0]
    max_entropy = 0
    grid_size = 9
    center_grid = select_grid(image, input_point, grid_size)

    indices = np.argwhere(mask ==True)
    for x,y in indices:
        grid = select_grid(image, [x,y], grid_size)
        entropy_diff = calculate_image_entroph(center_grid, grid)
        if entropy_diff > max_entropy:
            max_entropy_point = [x,y]
            max_entropy = entropy_diff
    return [max_entropy_point[1], max_entropy_point[0]]


"""Max Distance Point"""
def get_distance_points(input_point, mask):
    max_distance_point = [0,0]
    max_distance = 0
    # grid_size = 9
    # center_grid = select_grid(image,input_point, grid_size)

    indices = np.argwhere(mask ==True)
    for x,y in indices:
        distance = np.sqrt((x- input_point[0])**2 + (y- input_point[1]) ** 2)
        if max_distance < distance:
            max_distance_point = [x,y]
            max_distance = distance
    return [max_distance_point[1],max_distance_point[0]]


"""Saliency Point"""
def get_saliency_point(img, mask, img_name, save_img_path=None):
    (img_h, img_w, _) = img.shape

    coor = np.argwhere(mask > 0)
    ymin = min(coor[:, 0])
    ymax = max(coor[:, 0])
    xmin = min(coor[:, 1])
    xmax = max(coor[:, 1])

    xmin2 = xmin - 10 if xmin - 10 > 0 else 0
    xmax2 = img_w if xmax + 10 > img_w else xmax + 10
    ymin2 = ymin - 10 if ymin - 10 > 0 else 0
    ymax2 = img_h if ymax + 10 > img_h else ymax + 10

    vst_input_img = img[ymin2:ymax2, xmin2:xmax2, :]
    # VST mask
    vst_mask = VST_test_once(img_path=vst_input_img)
    # judge point in the vst mask
    vst_indices = np.argwhere(vst_mask > 0)
    random_index = np.random.choice(len(vst_indices), 1)[0]
    vst_roi_random_point = [vst_indices[random_index][1], vst_indices[random_index][0]]
    vst_random_point = [vst_roi_random_point[0] + xmin - 10, vst_roi_random_point[1] + ymin - 10]

    return vst_random_point
