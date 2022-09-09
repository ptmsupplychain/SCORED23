"""detection.py"""

from argparse import ArgumentParser as AP
import json
import os
from pprint import pprint

from PIL import Image
from datasets import list_datasets, list_metrics, load_dataset, load_metric
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import requests
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
from transformers import AutoFeatureExtractor, pipeline
import uutils as UU


def predict(args):

    print("pred")
    coco_path = os.path.join(args.root, "COCOdataset2017")
    annfile = os.path.join(coco_path, "annotations/instances_val2017.json")
    coco = COCO(annfile)

    catIds = coco.getCatIds()
    cats = coco.loadCats(catIds)

    imgIds = coco.getImgIds()  # [:8]
    img_infos = coco.loadImgs(imgIds)

    img2path = lambda img: os.path.join(coco_path, f'images/val/{img["file_name"]}')
    read_img = lambda path: plt.imread(path)
    dataset = [img2path(img) for img in img_infos]

    print(args.model)

    "check if file exists already"

    pipe = pipeline("object-detection", model=args.model, device=-1)

    legend = UU.coco.id2l(coco)
    l2id = lambda l: legend["l2id"][l]
    xyxy2xywh = lambda b: [b["xmin"], b["ymin"], b["xmax"] - b["xmin"], b["ymax"] - b["ymin"]]

    results_path = f'{coco_path}/results/{args.model.replace("/","_")}'
    results_json = results_path + ".json"
    results_txt = results_path + ".txt"

    print("inference")

    dt = []
    for i, item in tqdm(enumerate(dataset), total=len(dataset)):

        d = pipe(item, threshold=0.5)
        d = [
            {
                "score": x["score"],
                "image_id": imgIds[i],
                "category_id": l2id(x["label"]),
                "bbox": xyxy2xywh(x["box"]),
            }
            for x in d
        ]
        dt += d

    with open(results_json, "w") as file:
        json.dump(dt, file)

    dt = coco.loadRes(results_json)
    eval = COCOeval(coco, dt, "bbox")
    eval.params.imgIds = imgIds

    eval.evaluate()
    eval.accumulate()

    """ pipe stdout to file """
    UU.deco.stdout(results_txt)(lambda: eval.summarize())()


def main():

    ap = AP()
    ap.add_argument("-i", "--inference", action="store_true")
    ap.add_argument("-e", "--eval", action="store_true")
    # ap.add_argument('-m','--models',nargs='+',type=str)
    ap.add_argument("-m", "--model", type=str)
    ap.add_argument("-r", "--root", type=str)
    args = ap.parse_args()

    try:
        predict(args)
    except Exception as ex:
        quit() if ex is KeyboardInterrupt else print(ex, "\n")
        # UU.imagenet.ignore(name=model,path=os.path.join(args.root,'IMAGENET'))


if __name__ == "__main__":
    main()
