import setuptools


long_description="""Python package for
Variational Diffusion Autoencoders with Random Walk Sampling. See https://arxiv.org/abs/1905.12724.
"""

setuptools.setup(
    name="vdae", # Replace with your own username
    version="0.0.2",
    author="Henry,Ofir",
    author_email="henry.li@yale.edu",
    description="python package for variational diffusion autoencoders",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lihenryhfl/SpectralVAEGAN",
    packages=setuptools.find_packages(),
    extra_packages = {
    'tensorflow': ['tensorflow==1.15'],
    'tensorflow with gpu': ['tensorflow-gpu==1.15']
    },
    install_requires=[
        'POT',
        'annoy',
        'scipy',
        'sklearn',
        'keras==2.3.0',
        'matplotlib'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)

