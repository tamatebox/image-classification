# image-classification

## How to use

```
python valid.py --path="path to data(default is'./data')" --photo_size="Reading image size(default is 224*224)" --aug_num="Augument number（default is 0）"
max_photo:Maximum number of read images per label
rate:Weighting percentage
```

cnn.py
Network design


## Directory structure

```
├── data
│   ├── label a
│   ├── label b
│   └── 
├── cnn_model.py
├── models
├── resnet50.py
└── valid.py
```
