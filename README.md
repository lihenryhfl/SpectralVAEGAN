# Variational Diffusion Autoencoder (VDAE)
![diagram](https://user-images.githubusercontent.com/9156971/91104612-1a2ffc00-e63c-11ea-8e04-33bac1e2a397.png)

VDAE is a python library that performs spectral clustering with deep neural networks. See https://arxiv.org/abs/1905.12724.

## requirements
To run our package, you'll need Python 3.x and the following python packages:
- scikit-learn
- tensorflow==1.15
- munkres
- annoy
- h5py
- POT

## installing the package
For the most painless installation process, you can run:
```
pip install vdae
```

However, to install the most up-to-date version, instead run:
```
git clone https://github.com/lihenryhfl/SpectralVAEGAN.git; cd SpectralVAEGAN; python setup.py install
```

## usage
For example usage, please see the example script `vdae_examples.ipynb`
