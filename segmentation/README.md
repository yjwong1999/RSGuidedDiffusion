# Pretrained Remote Sensing Semantic Segmentation

We use LoveDA's [pretrained hrnetw32](https://drive.google.com/drive/folders/1xFn1d8a4Hv4il52hLCzjEy_TY31RdRtg?usp=sharing). For simplicity's sake, we have uploaded the pretrained weigths to our project's dropbox, which is easier to be downloaded using curl command. We modified the codebase such that it fits into our pipeline. Please refer the [README](https://github.com/DoubleY-BEGC2024/RSGuidedDiffusion/tree/main#) in the homepage.

## Class Map

We follow the class mapping similar to [LoveDA Semantic Segmentation Challenge](https://github.com/Junjue-Wang/LoveDA/tree/master/Semantic_Segmentation)

```python
# color map
COLOR_MAP = OrderedDict(
    Background=(60,20,90),         # index 0
    Building=(255, 0, 0),          # index 1
    Road=(255, 255, 0),            # index 2
    Water=(0, 0, 255),             # index 3
    Barren=(159, 129, 183),        # index 4
    Forest=(0, 255, 0),            # index 5
    Agricultural=(255, 195, 128),  # index 6
)
CLASS = list(COLOR_MAP.keys())
```
