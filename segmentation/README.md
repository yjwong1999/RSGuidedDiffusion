

## Class Map

We follow the class mapping similar to [LoveDA dataset](https://github.com/Junjue-Wang/LoveDA/tree/master/Semantic_Segmentation)

```python
# color map
COLOR_MAP = OrderedDict(
    Background=(60,20,90),
    Building=(255, 0, 0),
    Road=(255, 255, 0),
    Water=(0, 0, 255),
    Barren=(159, 129, 183),
    Forest=(0, 255, 0),
    Agricultural=(255, 195, 128),
)
CLASS = list(COLOR_MAP.keys())
```
