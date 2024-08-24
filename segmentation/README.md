# BEGC2024_Diffusion

## Getting Started

#### Requirements:
- pytorch >= 1.1.0
- python >=3.6

### Install Ever + Segmentation Models PyTorch
```bash
pip install ever-beta==0.2.3
pip install git+https://github.com/qubvel/segmentation_models.pytorch
```

### Prepare LoveDA Dataset

```bash
ln -s </path/to/LoveDA> ./LoveDA
```

### Evaluate Model on the test set
#### 1. Download the pre-trained [<b>weights</b>](https://drive.google.com/drive/folders/1xFn1d8a4Hv4il52hLCzjEy_TY31RdRtg?usp=sharing)
#### 2. Move weight file to log directory
```bash
mkdir -vp ./log/
mv ./hrnetw32.pth ./log/hrnetw32.pth
```
#### 3. Predict on test set
```bash 
bash ./scripts/predict_test.sh
```
Submit your test results on [LoveDA Semantic Segmentation Challenge](https://codalab.lisn.upsaclay.fr/competitions/421) and you will get your Test score.


### Train Model
```bash 
bash ./scripts/train_hrnetw32.sh
```
Evaluate on eval set 
```bash
bash ./scripts/eval_hrnetw32.sh
```



