# Benchmark Evaluation

**Disclaimer.** Evaluation can currently only be done on the train partition. It will be updated to support the entire benchmark when the generalization partition is added to the dataset in the coming weeks.

The model-agnostic code for evaluating benchmark performance resides in the `benchmark` folder. It reads a pickle file of estimated poses and outputs a pickle file containing computed metrics.

## Input Format

The input must be saved as a pickle file. The expected content of the pickle file is a dictionary. The keys of the dictionary are each of the kits in that specific partition of the benchmark. For example, on the train partition, you should have 5 keys, one for each of the 5 kits. The value of each key is a list containing the estimated pose of each data sample in that particular kit. In the case of the train partition, we have 25 test points per kit. For the generalization benchmark, there are 20 test points per configuration.

```
estimated_poses = {
    'deodorants': [pose_0, pose_1, ..., pose_24],
    'black-floss': [pose_0, pose_1, ..., pose_24],
    ...
}
```

## Evaluate

To evaluate the performance of your model, run the following command, replacing `{ALGORITHM}_poses.pkl` with the name of your saved estimated poses pickle file. Make sure it ends with `_poses.pkl`.

```
python eval.py {ALGORITHM}_poses.pkl --debug=False
```

This will output a pickle file of computed accuracies named `{ALGORITHM}_acc.pkl`. The computed metrics are:

* translational error
* rotational error
* reprojection error
* average distance (ADD)

## Visualize

You can now plot accuracy vs threshold curves with:

```
python auc.py ALGORITHM_acc.pkl
```