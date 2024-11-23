# Segmentation-Guided Diffusion Models for Label-Efficient Fine-Tuning for Remote Sensing Building Segmentation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nwClwVQKWVAojyBgxPu51olI3NcKxmpu?usp=sharing)

#### By [Yi Jie WONG](https://github.com/yjwong1999) et al

This code is part of our solution for [2024 IEEE BigData Cup: Building Extraction Generalization Challenge (IEEE BEGC2024)](https://www.kaggle.com/competitions/building-extraction-generalization-2024/overview). Specifically, this repo provides a segmentation-guided diffusion model to transform the given input land cover segmentation masks into realistic images. These images, along with the segmentation masks (labels), can then be used to train a building extraction instance segmentation model. Essentially, this is a diffusion augmentation technique tailored for the building extraction task. We find that our YOLO model trained using both the original and synthetic dataset generated by our diffusion model is comparable to the YOLO model trained with the original dataset alone. Our approach ranked 1st globally in the IEEE Big Data Cup 2024 - BEGC2024 challenge! 🏅🎉🥳

![methodology](https://github.com/yjwong1999/RSGuidedDiffusion/blob/main/assets/Segmentation%20Guided%20Diffusion.jpg?raw=true)

## Instructions

We provided a [Jupyter Notebook](https://github.com/yjwong1999/RSGuidedDiffusion/blob/main/BEGC2024_Segmentation_Guided_Diffusion_Model.ipynb) for easy reimplementation of our method. The entire notebook is run via lightning.ai studio. You might need to edit the filepath if run it on colab. Otherwise, you should be able to run it even in your local Jupyter notebook or in lightning.ai studio. Otherwise, you can follow the instructions below using command prompt.

Conda environment
```bash
conda create --name diffusion python=3.10.12 -y
conda activate diffusion
```

Clone this repo
```bash
# clone this repo
git clone https://github.com/yjwong1999/RSGuidedDiffusion.git
cd RSGuidedDiffusion
```

Install dependencies
```bash
# Please adjust the torch version accordingly depending on your OS
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121

# Install Jupyter Notebook
pip install jupyter notebook==7.1.0

# Remaining dependencies (for segmentation)
pip install opendatasets==0.1.22
pip install ever-beta==0.2.3
pip install git+https://github.com/qubvel/segmentation_models.pytorch
pip install pycocotools requests click

# Remaining dependencies (for diffusion)
pip install diffusers==0.21.4
pip install datasets==2.14.5
pip install transformers==4.33.2
pip install tensorboard==2.14.0
pip install safetensors==0.4.4
```


Setup the dataset
```bash
# Make sure you have your Kaggle authentication
# {"username":<USERNAME>,"key":<YOUR KEY>}

# run the code
python setup_data.py
```


Setup Segmentation
```bash
# Get Pretrained HRNet weights
cd segmentation
curl -L -o "hrnetw32.pth" "https://www.dropbox.com/scl/fi/5au20lvw3yb5y3btnlamg/hrnetw32.pth?rlkey=eoqio6mlxtq4ykdnaa8n4dp4l&st=d4tg641s&dl=0"

# move the pretrained weights to the designated directory
mkdir -vp ./log/
mv "hrnetw32.pth" "./log/hrnetw32.pth"

# make a soft link from "building-extraction-generalization-2024" to get the image data into LoveDA
ln -s "../building-extraction-generalization-2024" ./LoveDA

# copy the label data into LoveDA
cp -r "../detect/train/label" ./LoveDA/train
cp -r "../detect/val/label" ./LoveDA/val

# run the segmentation code
python3 run.py
cd ../
```

##  Train the diffusion model
Train Diffusion Model
```bash
current_dir=$(pwd)
CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --mode train \
    --model_type DDIM \
    --img_size 256 \
    --num_img_channels 3 \
    --dataset BEGC \
    --img_dir ${current_dir}/segmentation/diffusion_data/data \
    --seg_dir ${current_dir}/segmentation/diffusion_data/mask \
    --segmentation_guided \
    --segmentation_channel_mode single \
    --num_segmentation_classes 7 \
    --train_batch_size 2 \
    --eval_batch_size 2 \
    --num_epochs 200
```

Test Diffusion Model
```bash
current_dir=$(pwd)
CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --mode eval_many \
    --model_type DDIM \
    --img_size 256 \
    --num_img_channels 3 \
    --dataset BEGC \
    --eval_batch_size 1 \
    --eval_sample_size 1584 \
    --seg_dir ${current_dir}/segmentation/diffusion_data/mask \
    --segmentation_guided \
    --segmentation_channel_mode single \
    --num_segmentation_classes 7 
```

## Download Pretrained Model (prefered)
Download via curl command, or manually via [this link](https://www.dropbox.com/scl/fi/86i7mvr3fe1rkgejdewcj/ddim-BEGC-256-segguided.zip?rlkey=eugkdfero832mecdu9mdk0fio&st=n7tixbaa&dl=0).
```bash
curl -L -o "ddim-BEGC-256-segguided.zip" "https://www.dropbox.com/scl/fi/86i7mvr3fe1rkgejdewcj/ddim-BEGC-256-segguided.zip?rlkey=eugkdfero832mecdu9mdk0fio&st=k245vc5h&dl=0"
```

## Download the Generated Synthetic Dataset (prefered)
Download via curl command, or manually via [this link](https://www.dropbox.com/scl/fi/slq3qcg0qhzpj9cc22ws4/generated_images.zip?rlkey=npgj3v4ki6o7sogrca742ubt3&st=fjuxt1vn&dl=0)
```bash
curl -L -o "generated_images.zip" "https://www.dropbox.com/scl/fi/slq3qcg0qhzpj9cc22ws4/generated_images.zip?rlkey=npgj3v4ki6o7sogrca742ubt3&st=fjuxt1vn&dl=0"
```

## Optional: Custom Dataset
Please put your training images in some dataset directory `DATA_FOLDER`, organized into train, validation and test split subdirectories. The images should be in a format that PIL can read (e.g. `.png`, `.jpg`, etc.). For example:
``` 
DATA_FOLDER
├── train
│   ├── train_1.jpg
│   ├── train_2.jpg
│   └── ...
├── val
│   ├── val_1.jpg
│   ├── val_2.jpg
│   └── ...
└── test
    ├── test_1.jpg
    ├── test_2.jpg
    └── ...
```

Please put your segmentation masks in a similar directory structure in a separate folder `MASK_FOLDER`, with a subdirectory `all` that contains the split subfolders, as shown below. **Each segmentation mask should have the same filename as its corresponding image in `DATA_FOLDER`, and should be saved with integer values starting at zero for each object class, i.e., 0, 1, 2,...**. Note that class 0 should represent the background (if available).
``` 
MASK_FOLDER
├── all
│   ├── train
│   │   ├── train_1.jpg
│   │   ├── train_2.jpg
│   │   └── ...
│   ├── val
│   │   ├── val_1.jpg
│   │   ├── val_2.jpg
│   │   └── ...
│   └── test
│       ├── test_1.jpg
│       ├── test_2.jpg
│       └── ...
```

## Samples of Generated Images

![Sample Images](https://github.com/yjwong1999/RSGuidedDiffusion/blob/main/assets/Samples%20of%20Diffusion%20Data.png?raw=true)


## Comparison with Existing Diffusion Models

![Comparison](https://github.com/yjwong1999/RSGuidedDiffusion/blob/main/assets/Ours%20vs%20Seg2Sat.png?raw=true)

Our segmentation-guided diffusion model generates a more faithful reconstruction of the buildings’ shape and texture, with the buildings generated at the precise location, as opposed to [Seg2Sat](https://github.com/RubenGres/Seg2Sat). Please note that the images generated by Seg2Sat look clearer because the model is trained using a 512 x 512 image resolution. Meanwhile, our segmentation-guided diffusion model is trained with a 256 x 256 image resolution due to GPU resource constraints. Nevertheless, our diffusion model still generates a more faithful reconstruction of the buildings' shape and texture based on the input segmentation mask!

## Using the Synthetic Data to train YOLOv8-Seg
<table border="1">
  <tr>
    <th rowspan=2>Solution</th>
    <th rowspan=2>FLOPS (G)</th>
    <th colspan="2">F1-Score</th>
  </tr>
  <tr>
    <td>Public</td>
    <td>Private</td>
  </tr>
  <tr>
    <td>YOLOv8m-seg + BEGC 2024</td>
    <td rowspan=4>110.2</td>
    <td>0.64926</td>
    <td>0.66531</td>
  </tr>
  <tr>
    <td>YOLOv8m-seg + BEGC 2024 + Redmond Dataset</td>
    <td>0.65951</td>
    <td>0.67133</td>
  </tr>
  <tr>
    <td>YOLOv8m-seg + BEGC 2024 + Las Vegas Dataset</td>
    <td>0.68627</td>
    <td>0.70326</td>
  </tr>
  <tr>
    <td>YOLOv8m-seg + BEGC 2024 + Diffusion Augmentation</td>
    <td>0.67189</td>
    <td>0.68096</td>
  </tr>
  <tr>
    <td>2nd place (RTMDet-x + Alabama Buildings Segmentation Dataset)</td>
    <td>141.7</td>
    <td>0.6813</td>
    <td>0.68453</td>
  </tr>
  <tr>
    <td>3rd Place (Custom Mask-RCNN + No extra Dataset)</td>
    <td>124.1</td>
    <td>0.59314</td>
    <td>0.60649</td>
  </tr>
</table>

Please refer [this repo](https://github.com/yjwong1999/RSBuildingExtraction) to see how we can use Microsoft Building Footprint dataset to enhance the F1-score of our YOLO instance segmentation model.

Observations
- Using an additional dataset, whether it is an open-sourced dataset or a synthetic dataset, helps improve the training of YOLOv8-Seg.
- However, you might sample high-quality or low-quality additional datasets from open-sourced databases without careful engineering. For instance, using the Redmond dataset only slightly improves the F1 score compared to using the BEGC 2024 dataset alone. On the other hand, using the Las Vegas dataset significantly improves the F1 score, achieving the top F1 score among all methods.
- On the other hand, using our diffusion augmentation, we can generate a synthetic dataset to train YOLOv8m-Seg without needing an additional dataset (which means no extra manual annotation is required). Using BEGC2024 combined with the synthetic dataset, our YOLOv8m-Seg model achieved an F1 score that is significantly higher than the baseline and close to our top-1 score (using the Las Vegas dataset) and the 2nd-place solution.
- Note that the 2nd-place solution uses a bigger model (higher FLOPs) with an additional dataset to reach a high F1 score, whereas our diffusion augmentation pipeline allows our model (lower FLOPs) to achieve a surprisingly close F1 score without an additional dataset.

## Limitations
1. The segmentation mask generated using the LoveDA's [pretrained hrnetw32](https://drive.google.com/drive/folders/1xFn1d8a4Hv4il52hLCzjEy_TY31RdRtg?usp=sharing) is not perfect. In fact, it missed a lot of segmentation, most likely due to the different data distributions between the LoveDA dataset and the IEEE BEGC2024 dataset. However, it is still better than nothing. Currently, this is the best pretrained model that can at least segment the IEEE BEGC2024 dataset at an acceptable level. Feel free to try other/better segmentation models!
2. Due to resource constraints, we are only able to train our diffusion model using a batch size of 2 (accumulated to a batch size of 16) with a 256 image size. We suggest those interested in trying our code use a better GPU with higher RAM to train the model using a higher batch size and higher image resolution.
3. There might be class imbalance for the land cover segmentation mask, which could affect the training of the segmentation-guided diffusion model. Future work may explore how to mitigate this via adaptive sampling or upsampling.


## Troubleshooting/Bugfixing
- You might receive an error of `module 'safetensors' has no attribute 'torch'`. This appears to be an issue with the `diffusers` library itself in some environments, and may be mitigated by [this proposed solution](https://github.com/mazurowski-lab/segmentation-guided-diffusion/issues/11#issuecomment-2251890600).
- It is preferable to save your images/masks in .jpg/.tif format. I tried saving the masks in .png format, but the code automatically converts the values (object class) from integer to float. Hence, the easiest workaround is to use .jpg format, although this comes with a slight sacrifice in image quality.


## Acknowledgement
We thank the following works for the inspiration of our repo!
1. 2024 IEEE BigData Cup: Building Extraction Generalization Challenge [link](https://www.kaggle.com/competitions/building-extraction-generalization-2024/overview)
2. Remote Sensing Segmentation [code](https://github.com/Junjue-Wang/LoveDA/tree/master/Semantic_Segmentation)
3. Segmentation-Guided Diffusion [code](https://github.com/mazurowski-lab/segmentation-guided-diffusion)
4. COCO2YOLO format [original code](https://github.com/tw-yshuang/coco2yolo), [modified code](https://github.com/yjwong1999/coco2yolo)

## Cite this repository

Our paper has been accepted by IEEE BigData 2024! Please cite our paper if this repo helps your research. The preprint is available [here](https://doi.org/10.36227/techrxiv.173091008.80781383/v1)

```
@InProceedings{Wong2024,
title = {Cross-City Building Instance Segmentation: From More Data to Diffusion-Augmentation},
author = {Yi Jie Wong and Yin-Loon Khor and Mau-Luen Tham and Ban-Hoe Kwan and Anissa Mokraoui and Yoong Choon Chang},
booktitle={2024 IEEE International Conference on Big Data (Big Data)},
year={2024}}
```
