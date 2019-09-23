"""A meta-dataloader for iterating over combinations of datasets.
"""

import random

from form2fit.code.ml.dataloader.suction import *
from form2fit.code.ml.dataloader.placement import *
from form2fit.code.ml.dataloader.correspondence import *


class MetaLoader:
    def __init__(self, names, dtype, stype, kwargs):
        self.names = names
        self.dtype = dtype
        self.kwargs = kwargs

        if stype == "corr":
            func = get_corr_loader
        elif stype == "suction":
            func = get_suction_loader
        elif stype == "place":
            func = get_placement_loader
        else:
            raise ValueError("{} not supported.".format(stype))

        self.dsets = []
        for name, kwarg in zip(self.names, self.kwargs):
            self.dsets.append(func(name, dtype=self.dtype, **kwarg))
        self.loaders = [iter(d) for d in self.dsets]
        self.num_loaders = len(self.loaders)

        self.iter = 0

    def __iter__(self):
        return self

    def __next__(self):
        """Samples from each dataset consecutively.
        """
        dset_idx = self.iter % self.num_loaders
        try:
            imgs, labels = next(self.loaders[dset_idx])
        except StopIteration:
            self.loaders[dset_idx] = iter(self.dsets[dset_idx])
            imgs, labels = next(self.loaders[dset_idx])
        self.iter += 1
        return imgs, labels