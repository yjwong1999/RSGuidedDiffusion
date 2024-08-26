# Remote Sensing Guided Diffusion with Limited Domain Data

This code is part of our solution for 2024 IEEE BigData Cup: Building Extraction Generalization Challenge (BEGC).

## Setup
Conda environment
```bash
conda create --name diffusion python=3.10.12 -y
conda activate diffusion
```

Clone this repo
```bash
# clone this repo
git clone https://github.com/DoubleY-BEGC2024/RSGuidedDiffusion.git
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
    --num_epochs 400
```

## Troubleshooting/Bugfixing
- Some users have reported a [bug](https://github.com/mazurowski-lab/segmentation-guided-diffusion/issues/11) when the model attempts to save during training and they receive an error of `module 'safetensors' has no attribute 'torch'`. This appears to be an issue with the `diffusers` library itself in some environments, and may be remedied by [this proposed solution](https://github.com/mazurowski-lab/segmentation-guided-diffusion/issues/11#issuecomment-2251890600).


## Acknowledgement
1. 2024 IEEE BigData Cup: Building Extraction Generalization Challenge [link](https://www.kaggle.com/competitions/building-extraction-generalization-2024/overview)
2. Remote Sensing Segmentation [code](https://github.com/Junjue-Wang/LoveDA/tree/master/Semantic_Segmentation)
3. Segmentation-Guided Diffusion [code](https://github.com/mazurowski-lab/segmentation-guided-diffusion)
4. COCO2YOLO format [original code](https://github.com/tw-yshuang/coco2yolo), [modified code](https://github.com/yjwong1999/coco2yolo)
