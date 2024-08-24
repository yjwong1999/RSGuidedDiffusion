# Remote Sensing Guided Diffusion with Limited Domain Data

This code is part of our solution for 2024 IEEE BigData Cup: Building Extraction Generalization Challenge (BEGC).

## Setup
Conda environment
```bash
conda create --name diffusion python=3.10.12 -y
conda activate diffusion
```

Install Pytorch
```bash
# Please adjust accordingly depending on your OS
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
```

Install Jupyter Notebook
```
pip install jupyter notebook==7.1.0
```

## Acknowledgement
1. 2024 IEEE BigData Cup: Building Extraction Generalization Challenge [link](https://www.kaggle.com/competitions/building-extraction-generalization-2024/overview)
2. Remote Sensing Segmentation [code](https://github.com/Junjue-Wang/LoveDA/tree/master/Semantic_Segmentation)
3. Segmentation-Guided Diffusion [code](https://github.com/mazurowski-lab/segmentation-guided-diffusion)
