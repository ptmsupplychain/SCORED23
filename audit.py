import json

import matplotlib.pyplot as plt

from ptm import PTM


def fig11():
    """generates figure 11"""

    PTM.load([m.id for m in PTM.api.list_models()])

    "Figure 11"

    with open("../reproducibility/results/sentiment_analysis.json", "r") as file:
        results = json.load(file)["results"]
        models = list(results.keys())
        results = {k: v["f1-micro"] for k, v in results.items()}

    PTM.ptms = [p for p in PTM.ptms if p.id in models]
    # PTM.ptms = [p for p in PTM.ptms if p.claims]

    data = {m: {"results": results[m]} for m in models}
    for p in PTM.ptms:
        data[p.id]["ptms"] = p

    n = 0
    errs = []
    for k, d in data.items():
        r = d["results"]
        try:
            c = [c.value for c in d["ptms"].claims if "accuracy" == c.type]
            if len(c) == 1:
                c = c[0]
                errs.append([k, c - r])
                if c - r > 0.1:
                    print(k)
                    print(c, r)
        except:
            pass

    errs.sort(key=lambda x: x[1])


def fig5():
    """generates figure 5"""

    PTM.load([m.id for m in PTM.api.list_models()])

    tasks = set([p.task for p in PTM.ptms])

    hist = {t: {"claimed": 0, "claimless": 0} for t in tasks}

    for p in PTM.ptms:
        hist[p.task]["claimed" if p.claims else "claimless"] += 1

    if None in hist:
        hist["None"] = hist[None]
        del hist[None]

    for k, v in hist.items():
        ratio = v["claimed"] / (v["claimed"] + v["claimless"])
        total = v["claimed"] + v["claimless"]
        hist[k] = [ratio, total]

    others, claims, delete = [], [], []
    for k, v in hist.items():
        if v[0] < 0.001:
            claims.append(v[0] * v[1])
            others.append(v[1])
            delete.append(k)
    hist["other"] = [sum(claims) / sum(others), sum(others)]
    for d in delete:
        del hist[d]

    keys, values = [x for x in zip(*sorted(hist.items(), key=lambda x: x[1][0]))]
    totals = [v[1] for v in values]
    values = [v[0] for v in values]

    ax = plt.gca()
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    bars = plt.barh(keys, values)
    for rect, ratio, total in zip(bars, values, totals):
        h, w = rect.get_height(), rect.get_width()
        # w = 0-plt.gcf().get_figwidth()
        x, y = rect.get_x(), rect.get_y()
        plt.text(x + w, y, f"{int(ratio*total)}/{total}", ha="left", va="bottom", fontsize=12)

    plt.xticks([0.05, 0.1, 0.15, 0.2], ["5%", "10%", "15%", "20%"])
    plt.tight_layout()

    plt.show()


def sentiment_claims():
    """get sentiment analysis models (emotion dataset) that make claims"""

    PTM.load([m.id for m in PTM.api.list_models(filter="text-classification")])
    PTM.ptms = [p for p in PTM.ptms if p.claims]

    cond = lambda p: any([(c.dataset['type'] == 'emotion' ) for c in p.claims])
    # cond = lambda p: any([(c.ds_type == 'emotion' and c.type == 'accuracy') for c in p.claims])
    PTM.ptms = [p for p in PTM.ptms if cond]

    print(len(PTM.ptms))
    print(PTM.ptms[0])


def main():
    """docstring"""

    PTM.load([m.id for m in PTM.api.list_models()])
    # PTM.load([m.id for m in PTM.api.list_models(filter="image-classification")])

    PTM.ptms = [p for p in PTM.ptms if p.claims]
    condition = lambda p: any([(c.value > 1 and c.type == "accuracy") for c in p.claims])
    PTM.ptms = [p for p in PTM.ptms if condition]

    print(len(PTM.ptms))
    print(PTM.ptms)


if __name__ == "__main__":
    sentiment_claims()
    # main()
