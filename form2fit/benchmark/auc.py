import argparse
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from form2fit import config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Curves")
    parser.add_argument("pkl_files", nargs='+', type=str)
    args, unparsed = parser.parse_known_args()

    dump_dir = "../code/dump/"
    plot_dir = "./plots/"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    accs = [pickle.load(open(os.path.join(dump_dir, fp), "rb")) for fp in args.pkl_files]
    names = [pkl_file.split("_")[0] for pkl_file in args.pkl_files]

    all_res = {name: {} for name in names}
    aucs = {}
    for method_name, method_dict in zip(names, accs):
        aucs[method_name] = {}
        res = {
            "err_tr": [],
            "err_rot": [],
            "err_add": [],
            "err_reproj": [],
        }
        for k, v in method_dict.items():
            aucs[method_name][k] = {}

            # load metrics for current item
            err_tr = np.array(v['err_trans'])
            err_rot = np.array(v['err_rot'])
            err_add = np.array(v['err_add'])
            err_reproj = np.array(v['err_reproj'])

            acc_lists = [err_tr, err_rot, err_add, err_reproj]
            acc_names = ["err_tr", "err_rot", "err_add", "err_reproj"]
            x_names = [
                "Translation Threshold [m]",
                "Rotation Angle Threshold [deg]",
                "Average Distance Threshold [m]",
                "Reprojection Threshold [heightmap pix]",
            ]

            for acc_list, acc_name, x_name in zip(acc_lists, acc_names, x_names):
                # if NaN encountered, make it bigger than the threshold
                idxs = np.argwhere(np.isnan(acc_list))
                acc_list[idxs] = 10000000
                accuracies = []
                if acc_name == "err_rot":
                    thresholds = np.linspace(0, config.MAX_DEG_THRESH, 30)
                elif acc_name == "err_reproj":
                    thresholds = np.linspace(0, config.MAX_PIX_THRESH, 20)
                elif acc_name == "err_tr":
                    thresholds = np.linspace(0, config.MAX_TR_THRESH, 20)
                else:
                    thresholds = np.linspace(0, config.MAX_LENGTH_THRESH, 20)
                for thresh in thresholds:
                    accuracies.append(np.mean(acc_list < thresh))
                aucs[method_name][k][acc_name] = np.trapz(accuracies, thresholds)
                res[acc_name].append(accuracies)
        all_res[method_name] = res

    with open("per_auc.json", "w") as fp:
        json.dump(aucs, fp, indent=4)

    aucs = {}
    fig, axes = plt.subplots(1, 4, sharey=True, figsize=(25, 6))
    for i, (metric_name, metric_xlabel) in enumerate(zip(acc_names, x_names)):
        aucs[metric_name[4:]] = {}
        for method_name, method_dict in all_res.items():
            accs = np.mean(np.array(method_dict[metric_name]), axis=0)
            if metric_name == "err_rot":
                thresholds = np.linspace(0, config.MAX_DEG_THRESH, 30)
            elif metric_name == "err_reproj":
                thresholds = np.linspace(0, config.MAX_PIX_THRESH, 20)
            elif metric_name == "err_tr":
                thresholds = np.linspace(0, config.MAX_TR_THRESH, 20)
            else:
                thresholds = np.linspace(0, config.MAX_LENGTH_THRESH, 20)
            aucs[metric_name[4:]][method_name] = np.trapz(accs, thresholds)
            axes.flatten()[i].plot(thresholds, accs, label=method_name)
        axes.flatten()[i].set_xlabel(metric_xlabel)
        if i == 0:
            axes.flatten()[i].set_ylabel("Accuracy")
        axes.flatten()[i].legend(loc='lower right')
        axes.flatten()[i].grid()

    fig.tight_layout()
    plt.savefig(os.path.join(plot_dir, "4x4.pdf"), format="pdf", dpi=200)
    with open("avg_auc.json", "w") as fp:
        json.dump(aucs, fp, indent=4)