# Remote Sensing Guided Diffusion with Limited Domain Data

This code is part of our solution for [2024 IEEE BigData Cup: Building Extraction Generalization Challenge (BEGC)](https://www.kaggle.com/competitions/building-extraction-generalization-2024/overview). Specifically, this repo provides a segmentation-guided diffusion model to transform the given input land cover segmentation masks into realistic images. These images, along with the segmentation masks (labels), can then be used to train a building extraction instance segmentation model. Essentially, this is a diffusion augmentation technique tailored for the building extraction task.

![methodology](https://github.com/yjwong1999/EY-challenge-2024/blob/main/Team%20Double%20Y%20-%20Methodology.jpg?raw=true)

## Instructions
Conda environment
```bash
conda create --name diffusion python=3.10.12 -y
conda activate diffusion
```

Clone this repo
```bash
# clone this repo
git clone https://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@github.com/DoubleY-BEGC2024/RSGuidedDiffusion.git
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
# {"username":"yjwong99","key":"5708a20c5643d05827b398dce05031bc"}

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
    --num_epochs 100
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
    --eval_sample_size 100 \
    --seg_dir ${current_dir}/segmentation/diffusion_data/mask \
    --segmentation_guided \
    --segmentation_channel_mode single \
    --num_segmentation_classes 7 
```

Download Pretrained Model (prefered)
```bash
curl -L -o "ddim-BEGC-256-segguided.zip" "https://www.dropbox.com/scl/fi/86i7mvr3fe1rkgejdewcj/ddim-BEGC-256-segguided.zip?rlkey=eugkdfero832mecdu9mdk0fio&st=k245vc5h&dl=0"
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


## Troubleshooting/Bugfixing
- You might receive an error of `module 'safetensors' has no attribute 'torch'`. This appears to be an issue with the `diffusers` library itself in some environments, and may be mitigated by [this proposed solution](https://github.com/mazurowski-lab/segmentation-guided-diffusion/issues/11#issuecomment-2251890600).
- It is preferable to save your images/masks in .jpg format. I tried saving the masks in .png format, but the code automatically converts the values (object class) from integer to float. Hence, the easiest workaround is to use .jpg format, although this comes with a slight sacrifice in image quality.


## Acknowledgement
We thank the following works for the inspiration of our repo!
1. 2024 IEEE BigData Cup: Building Extraction Generalization Challenge [link](https://www.kaggle.com/competitions/building-extraction-generalization-2024/overview)
2. Remote Sensing Segmentation [code](https://github.com/Junjue-Wang/LoveDA/tree/master/Semantic_Segmentation)
3. Segmentation-Guided Diffusion [code](https://github.com/mazurowski-lab/segmentation-guided-diffusion)
4. COCO2YOLO format [original code](https://github.com/tw-yshuang/coco2yolo), [modified code](https://github.com/yjwong1999/coco2yolo)
