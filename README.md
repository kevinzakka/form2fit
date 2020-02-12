# Form2Fit

Code for the paper

**[Form2Fit: Learning Shape Priors for Generalizable Assembly from Disassembly][1]**<br/>
*Kevin Zakka, Andy Zeng, Johnny Lee, Shuran Song*<br/>
[arxiv.org/abs/1910.13675][2]<br/>
ICRA 2020

<p align="center">
<img src="./assets/teaser.gif" width=100% alt="Drawing">
</p>

This repository contains:

- The [Form2Fit Benchmark](docs/about_benchmark.md)
  - Code to [download and process](#data) the benchmark datasets.
  - Code to [evaluate](docs/evaluate_benchmark.md) any model's performance on the benchmark test set.
- Code to [reproduce](docs/paper_code.md) the paper results:
	- Architectures, dataloaders and losses for suction, place and matching networks.
	- Planner module for intergrating all the outputs.
  - Baseline implementation.

If you find this code useful, consider citing our work:

```
@inproceedings{zakka2019form2fit,
  title={Form2Fit: Learning Shape Priors for Generalizable Assembly from Disassembly},
  author={Zakka, Kevin and Zeng, Andy and Lee, Johnny and Song, Shuran},
  journal={arXiv preprint arXiv:1910.13675},
  year={2019}
}
```

### Documentation

- [setup](docs/setup.md)
- [about the Form2Fit benchmark](docs/about_benchmark.md)
- [reproducing paper results](docs/paper_code.md)
- [evaluating a trained model](docs/evaluate_benchmark.md)
- [model weights](docs/model_weights.md)
- [conventions](docs/conventions.md)

### Todos

- [ ] Add processed generalization partition (combinations, mixtures and unseen) to benchmark.
- [ ] Add code for training the different networks.

### Note

This is not an officially supported Google product.

[1]: https://form2fit.github.io/
[2]: https://arxiv.org/abs/1910.13675
