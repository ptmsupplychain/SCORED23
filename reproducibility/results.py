"""one timer ... concat all data into results.json and results.csv nicely"""

import json
import os

import pandas as pd


mode = input('do you want to overwrite the previous results.json? (this is a bad idea) (y/n) ')
if mode != 'y':
    quit()


results = ["image_classification.json", "sentiment_analysis.json"]

R = []
for r in results:

    with open(f"results/{r}", "r") as file:
        data = json.load(file)

    task = data["task"].replace('_','-')

    for name, results in data["results"].items():
        model = {"name": name, "task": task, "results": results}
        R.append(model)

claims = {
    "facebook_detr-resnet-50": 0.420,
    "hustvl_yolos-tiny": 0.287,
    "facebook_detr-resnet-101": 0.435,
    "hustvl_yolos-small": 0.361,
    "hustvl_yolos-base": 0.420,
    "facebook_detr-resnet-101-dc5": 0.449,
    "nickmuchi_yolos-small-finetuned-masks": None,  
    "facebook_detr-resnet-50-dc5": 0.433,
    "hustvl_yolos-small-300": 0.361,  
    "hustvl_yolos-small-dwr": 0.376,
    "SamMorgan_yolo_v4_tflite": None,  
}

d = 'results/COCO'
for f in os.listdir(d):

    with open(os.path.join(d,f),'r') as file:
        results = [line for line in file.readlines()][0].strip().split(' ')[-1]
        results = {'mAP':results}

    name = f.split('.')[0]
    task = 'object-detection'
    claim = {'claims':claims[name]} if name in claims else {}
    model = {'name':name,'task':task,'results':results, **claim}
    R.append(model)

df = pd.DataFrame(data=R)
print(df['task'])

df.to_csv('results.csv',index=False)
df.to_json('results.json',orient='records')
