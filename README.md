# PiSA
## Data Prepare
Follow the `datasets/README.md` to generate format_label like:
```
PiSA/
├── data/
│   ├── coco/
│   ├── pascal/
│   └── nuswide/
├── format_label/
│   ├── coco/
│   │   └── formatted_x.npy
│   ├── pascal/
│   │   └── formatted_x.npy
│   └── nuswide/
│       └── formatted_x.npy
```
## Salient Single Postive Multi Label Learnning
### Generating Single Positive Annotations
#### confidence-based:
First train a model to get salient_value in `format_label/{dataset}/salient_value.npy`  and generate observation label in `format_label/{dataset}/single_salient.npy` 
```
python preproc/setting.py --dataset {DATASET}
```
1. `{DATASET}`: The adopted dataset. (*default*: `pascal` | *available*: `pascal`, `coco`, `nuswide`, or `vg`)

#### size_based
```
python preproc/FromSeg.py --dataset {DATASET}
```

### Generating ML-PL
When get `salient_value.npy`, run:
```
python preproc/missing.py --left_rate {leftRate} --dataset {DATASET}
```
1. `{DATASET}`: The adopted dataset. (*default*: `pascal` | *available*: `pascal`, `coco`, `nuswide`, or `vg`)
2. `{leftRate}`: select `[0.3,0.5,0.7]` positive label to save


## Training and Evaluation
### SPML
Run `main.py` to train and evaluate a model:
```
python main.py -dataset {DATASET} --seg {SEG}
```
1. `{DATASET}`: The adopted dataset. (*default*: `pascal` | *available*: `pascal`, `coco`, `nuswide`, or `vg`)
2. `{SEG}`: `False` means confidence-based and `True` means sized_based.

For example, to train and evaluate a model on the PASCAL VOC dataset using our confidence-based, please run:
```
python main.py -dataset pascal --seg False
```

### ML-PL 

```
python main.py -dataset pascal --left_rate {leftRate}
```
# PiSA
