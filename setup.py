from setuptools import setup


setup(
    name="form2fit",
    version="0.0.1",
    description="Form2Fit Code and Benchmark",
    author="Kevin Zakka",
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.0.0",
        "scikit-image>=0.15.0",
        "open3d-python==0.4.0.0",
        "PyQt5==5.9.2",
        "torch",
        "torchvision",
        "ipdb",
        "opencv-python",
        "tqdm",
    ],
)
