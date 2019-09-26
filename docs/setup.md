# Setup

We're going to install `form2fit` as a local python package. The only requirement is to have python3 which you can install through [Anaconda](https://docs.continuum.io/anaconda/install/).

## Creating a Conda Environment

First, create a new conda environment:

```
conda create -n form2fit python=3.6
conda activate form2fit
```

## Installing Walle

Install `walle` with pip:

```
pip install walle
```

If it fails to import `python -c "import walle"`, you can install it locally with:

```
git clone https://github.com/kevinzakka/walle.git
cd walle
pip install -e .
```

## Installing Form2Fit

Finally, clone this repository and install locally with pip.

```
git clone https://github.com/kevinzakka/form2fit.git
cd form2fit
pip install -e .
```