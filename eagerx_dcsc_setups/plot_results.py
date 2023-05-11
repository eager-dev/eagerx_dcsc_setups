# Common imports
import os
import yaml
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# Import seaborn
import pandas as pd

# Apply the default theme
import seaborn as sns
sns.set(style="darkgrid", font_scale=1.5)
cmap = plt.rcParams['axes.prop_cycle'].by_key()['color']
# for key, value in plt.rcParams.items():
# if "font.size" not in key:
# continue
# print(key, value)
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['font.size'] = 12
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['xtick.major.pad'] = -2.0
plt.rcParams['ytick.major.pad'] = -2.0
plt.rcParams['lines.linewidth'] = 1.3
plt.rcParams['axes.xmargin'] = 0.0

plt.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

if __name__ == "__main__":
    # d = {}

    # Get root path
    root = Path(__file__).parent.parent

    # Load config
    cfg_path = root / "cfg" / "eval.yaml"
    with open(str(cfg_path), "r") as f:
        cfg = yaml.safe_load(f)
    train_cfg_path = root / "cfg" / "train.yaml"
    with open(str(train_cfg_path), "r") as f:
        train_cfg = yaml.safe_load(f)

    # Get parameters
    repetitions = cfg["eval"]["repetitions"]
    cfg["settings"] = train_cfg["settings"]

    # fig = plt.figure()
    fig, ax = plt.subplots()
    ax.set_ylabel("Episodic Cost")
    # ax.set_xlabel("Setting")
    # ax.set_title("Pendulum Swing-Up: Zero-Shot Transfer")
    width = 0.25
    xtickslabels = []
    # df = pd.DataFrame(data={})
    
    for idx, setting in enumerate(cfg["settings"].keys()):
        engine = cfg["settings"][setting]["engine"]
        # d[setting] = {}
        name = cfg["settings"][setting]["name"]
        # xtickslabels.append(f"sim/real\n{engine}\n" + name)
        xtickslabels.append(name)
        for mode in ["sim", "real"]:
            results = []
            for repetition in range(repetitions):
                eval_log_dir = root / "exps" / "eval" / "runs" / f"{setting}_{repetition}"
                eval_file = eval_log_dir / "eval.yaml"

                # Check if evaluation already done
                if os.path.exists(eval_file):
                    eval_results = yaml.safe_load(open(str(eval_file), "r"))
                else:
                    eval_results = {}
                if eval_results is not None and mode in eval_results.keys() and "results" in eval_results[mode].keys():
                    results.extend(eval_results[mode]["results"])
            if len(results) > 0:
                results = np.array(results)
                mean = -np.mean(results, axis=0)
                std = np.std(results, axis=0)
                x = idx - width / 2 if mode == "sim" else idx + width / 2
                color = "c" if mode == "sim" else "m"
                color = "b" if mode == "real" and engine == "gym" else color
                ax.bar(x, mean, label=f"{engine}\n{setting}_{mode}", yerr=std, width=width, color=color)
                # ax.boxplot(results, positions=[x], widths=width, showfliers=False, patch_artist=True)  #, boxprops=dict(facecolor=color))
                # d[setting][mode] = mean
                # for result in results:
                    # df = df.append({"name": f"{engine}\n{name}", "mode": mode, "Episodic Reward": result, "engine": engine}, ignore_index=True)
            # else:
            #     d[setting][mode] = 0
    # sns.catplot(data=df, kind="bar", x="name", y="Episodic Reward", hue="mode", errorbar="sd", palette="Set2")
    # plt.show()
    ax.set(xticks=np.arange(len(cfg["settings"].keys())), xticklabels=xtickslabels)
    ax.grid(axis="x")
    sim_patch = mpatches.Patch(color='c', label='sim')
    real_patch = mpatches.Patch(color='b', label='real')
    real_delayed_patch = mpatches.Patch(color='m', label='real delayed')
    std_patch = mlines.Line2D([], [], color='k', marker='_', linestyle='None', markersize=10, label='$\sigma$')
    ax.legend(handles=[sim_patch, real_patch, real_delayed_patch, std_patch])
    # ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    [label.set_fontweight('bold') for label in ax.get_xticklabels()]

    plt.show()
