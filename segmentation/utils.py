# import
import os, shutil
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import OrderedDict

from reder import COLOR_MAP, convert_to_color


#---------------------------------------------------------
# get img and polygons for the given image_path and label_path
#---------------------------------------------------------
def get_img_polygons(image_path, label_path):
    if not os.path.exists(label_path):
        assert False, f"Warning: Label not found for {image_path}"

    # print(f"Image: {image_path}")
    # print(f"Label: {label_path}")

    # read image
    img = Image.open(image_path)

    # show the size of im
    width, height = img.size
    # print(f'Image width: {width}')
    # print(f'Image height: {height}')

    # convert im to array
    img = np.array(img)

    # read the label txt file
    with open(label_path, 'r') as f:
        lines = f.readlines()

    # loop all line (line = polyon)
    polygons = []
    for line in lines:
        # remove next line
        polygon = line.replace('\n', '')

        # split by space (delimiter)
        polygon = polygon.split(' ')

        # convert all from string to float
        polygon = [float(i) for i in polygon]

        # first one is just class (all is building)
        polygon = polygon[1:]

        # multiply all odd number item in polygon with width, multiply all even number item in polygon with height
        polygon = [polygon[i] * width if i % 2 == 0 else polygon[i] * height for i in range(len(polygon))]

        # convert all item in polygon to int
        polygon = [int(i) for i in polygon]

        # group every 2 item together (x, y)
        polygon = [polygon[i:i+2] for i in range(0, len(polygon), 2)]

        # convert to array
        polygon = np.array(polygon)

        # append to polygons
        polygons.append(polygon)

    return img, polygons


#---------------------------------------------------------
# Convert YOLOv8-seg polygon format to mask
#---------------------------------------------------------
def polygons2mask(img, polygons):
    # get width and heigth of img array
    width, height = img.shape[1], img.shape[0]

    # create a mask using the polygon
    background = np.zeros((height, width, 3), dtype=np.uint8)
    mask = cv2.fillPoly(background, pts=polygons, color=(1, 1, 1))

    return mask


#---------------------------------------------------------
# Generate mask for the given image_dir and label_dir
#---------------------------------------------------------
def generate_mask(image_dir, label_dir, dataloader, mode, edge=False, plot=True, save_dir=None):
    #---------------------------------------------------------
    # Assertion
    #---------------------------------------------------------
    assert mode in ['train', 'val'], "mode must be either 'train' or 'val'"
    assert os.path.exists(image_dir), "image_dir does not exist"
    assert os.path.exists(label_dir), "label_dir does not exist"
    if save_dir is not None:
        if not os.path.exists(os.path.join(save_dir, 'data', mode)):
            os.makedirs(os.path.join(save_dir, 'data', mode))
            os.makedirs(os.path.join(save_dir, 'mask/all', mode))

    #---------------------------------------------------------
    # Generate the corrected mask
    #---------------------------------------------------------
    # image size
    imgsz = (512, 512)

    # loop
    idx = 0
    filenames = sorted(os.listdir(image_dir))
    for img, _ in tqdm(dataloader):

        #---------------------------------------------------------
        # HRNet (LoveDA) inference
        #---------------------------------------------------------
        # make prediction
        img = img.cuda()
        pred = model(img)
        pred = pred.argmax(dim=1).cpu()
        pred = pred[0].numpy()

        # rearrange img from (ch, r, c) to (r, c, ch)
        img = np.transpose(img[0].cpu(), (1, 2, 0))

        # show the image (processed)
        if plot:
            plt.imshow(img)
            plt.xticks([]), plt.yticks([])
            plt.show()

        # show the prediction
        if plot:
            plt.imshow(convert_to_color(pred))
            plt.xticks([]), plt.yticks([])
            plt.show()

        #---------------------------------------------------------
        # Building mask (ground truth)
        #---------------------------------------------------------
        # get image and label path
        filename = filenames[idx]
        image_path = os.path.join(image_dir, filename)
        label_filename = filename.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')  # Adjust as needed
        label_path = os.path.join(label_dir, label_filename)

        # get img and polygons
        img, polygons = get_img_polygons(image_path, label_path)

        # create mask
        building_mask = polygons2mask(img, polygons)
        building_mask = building_mask[:,:,0] # no need channel

        # resize
        img  = cv2.resize(img, imgsz, interpolation=cv2.INTER_LINEAR)
        building_mask = cv2.resize(building_mask, imgsz, interpolation=cv2.INTER_LINEAR)

        #---------------------------------------------------------
        # Remove building prediction
        #---------------------------------------------------------
        # remove building labels from prediction, then add ground truth building
        all_masks = []
        for i in range(len(CLASS)): # 7 labels
            # ignore the building labels (1) here
            if i == 1:
                continue
            all_masks.append(np.where(pred == i, i, 0))
        # recombine individual masks
        masked_array = sum(all_masks)

        #---------------------------------------------------------
        # Add canny edge as pseudo road
        #---------------------------------------------------------
        # canny edge
        if edge:
            img = np.asarray(img)
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
            edges = cv2.Canny(blurred_image, threshold1=100, threshold2=200)
            edges = edges / 255 # normalize to [0, 1] range
            edges = edges * 2  # road is class 2
            edges = np.array(edges, dtype=np.int32) # convert to integert

            # make sure the location with pseudp-road is set to 0 before adding
            masked_array[edges==2] = 0
            masked_array += edges

        #---------------------------------------------------------
        # Add building mask after all other labels are done
        #---------------------------------------------------------
        # make sure the location with building is set to 0 before adding
        masked_array[building_mask==1] = 0

        # add with the building ground truth mask
        masked_array = masked_array + building_mask

        if plot:
            plt.imshow(convert_to_color(masked_array))
            plt.xticks([]), plt.yticks([])
            plt.show()

        #---------------------------------------------------------
        # save
        #---------------------------------------------------------
        if save_dir is not None:
            basename = os.path.basename(image_path)

            # img = img.astype(np.uint8)  # Convert to uint8 format
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # cv2.imwrite(os.path.join(save_dir, 'data', mode, basename).replace('.jpg', '.png'), img)

            # masked_array = masked_array.astype(np.uint8)  # Convert to uint8 format
            # cv2.imwrite(os.path.join(save_dir, 'mask/all', mode, basename).replace('.jpg', '.png'), masked_array)

            img = Image.fromarray(img.astype(np.uint8))
            img.save(os.path.join(save_dir, 'data', mode, basename).replace('.jpg', '.jpg'), quality=100, subsampling=0)

            masked_array = Image.fromarray(masked_array.astype(np.uint8))#.convert("L")
            masked_array.save(os.path.join(save_dir, 'mask/all', mode, basename).replace('.jpg', '.jpg'), quality=100, subsampling=0)

        # increment index
        idx += 1
        # if idx >= 30:
        #     break
