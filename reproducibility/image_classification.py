"image classification"

from argparse import ArgumentParser as AP
import os
from pprint import pprint

from PIL import Image
from datasets import list_datasets, list_metrics, load_dataset, load_metric
import matplotlib.pyplot as plt
import requests
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, pipeline
import uutils as UU



def get_args():

    ap = AP()
    ap.add_argument("-m", "--models", type=str, nargs="+")
    ap.add_argument('-r','--root',type=str) # the root dir for datasets (in progress)

    args = ap.parse_args()
    return args

def predict():

    args = get_args()
    task = __file__.split("/")[-1].split(".")[0]

    try:
        with open(f"results/{task}.json", "r") as file:
            data = json.load(file)
    except:
        data = {"task": task, "results": {}, 'ignore':[]}

    dataset = "imagenet-1k" # could change w argument
    dataset = load_dataset(dataset, split="test")

    print(dataset[0])
    quit()

    done = [k for k in data["results"].keys()] + data['ignore']
    models = [m for m in args.models if m not in done]

    root = args.root if args.root else [print("need root"), quit()]
    imagenet_path = os.path.join(root, "IMAGENET")

    # paths = [p for p in os.listdir(os.path.join(imagenet_path, "val"))]
    # dataset = [os.path.join(imagenet_path, "val", p) for p in paths]

    for model in models:
        print(model)

        try:
            pipe = pipeline("image-classification", model=model, return_all_scores=True)

            synset_map = UU.imagenet.get_synset_map(path=imagenet_path)
            id2l = lambda id: synset_map["id2l"][id]
            l2id = lambda l: synset_map["l2id"][l]

            Y = []
            for x in tqdm(dataset):

                y = pipe(x)
                y.sort(key=lambda x: x["score"], reverse=True)
                y = [l2id(i["label"]) for i in y[:5]]
                y = {"top1": y[0], "top5": y}
                Y.append(y)

                prediction = pipe(x["text"])
                y = np.array([i["score"] for i in prediction[0]]).argmax()
                Y.append(y)

            print('eval')

            Yh = [y["label"] for y in dataset]
            f1 = evaluate.load("f1")
            eval = lambda a: f1.compute(references=Yh, predictions=Y, average=a)["f1"]
            result = {f"f1-{a}": eval(a) for a in ["macro", "micro", "weighted"]}
            data["results"][model] = result
        
        except Exception as ex:
            quit() if ex is KeyboardInterrupt else print(ex)
            data['ignore'] = data['ignore'] + [model]

        # inside loop so you dont lose intermediate progress
        with open(f"results/{task}.json", "w") as file:
            json.dump(data, file)


def main():
    pass

if __name__ == '__main__': 
    predict()
