# SCORED
code and data for SCORED conference

```
├── assets
│   ├── data
│   │   ├── model_hubs.json
│   │   ├── top20model_downloads.json
│   │   ├── top20npm_downloads.json
│   │   └── top20pypi_downloads.json
│   └── figures
│       ├── err.png
│       └── registry_monthly_downloads.png
├── reproducibility
│   ├── results
│   │   ├── COCO
│   │   │   └── **.txt
│   │   ├── EMOTION
│   │   │   ├── emotion-models.json
│   │   │   ├── sentiment_analysis.json
│   │   │   └── sentiment_err.json
│   │   └── IMAGENET
│   │       └── image_classification.json
│   ├── detection.py
│   ├── image_classification.py
│   ├── results.csv
│   ├── results.json
│   ├── results.py
│   └── sentiment_analysis.py
├── LICENSE
├── README.md
├── audit.py
├── err.png
├── err.py
├── ignore.json
├── modelinfo.json
├── modelinfo.py
├── ptm.py
└── requirements.txt
```

## COCO 

Contains output of object detection models evaluated on the COCO2017 dataset.

- facebook/detr-resnet-101-dc5
- facebook/detr-resnet-101
- facebook/detr-resnet-50-dc5
- facebook/detr-resnet-50
- hustvl/yolos-base
- hustvl/yolos-small-dwr
- hustvl/yolos-small
- hustvl/yolos-tiny

## assets



### modelinfo.json 

- contains model metadata for 67K+ repos
- used in audit.py

### ignore.json

- some models require a sign in to get access and others are restricted for various reasons 
- we ignore those

### modelinfo.py

updates data in the modelinfo.json to contain most up to date info on 67K+ repos.  

### audit.py

- loads all models in modelinfo.json and associated metadata as PTM objects (ptm.py)
- finds trends among tasks, downloads, claims, etc

## err.py

Reads validation data and generates `err.png` which shows the fraction of models that have > 1% discrepancy.

## reproducibility

Contains code to reproduce the results of HF models. Many models are not reproducible.

### results

Contains output of:
- image classification models evaluated on the imagenet dataset
- text classification models evaluated on the emotion dataset
