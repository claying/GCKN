# Graph convolutional kernel networks

The repository implements Graph Convolutional Kernel Networks (GCKNs) described in the following paper

>Dexiong Chen, Laurent Jacob, Julien Mairal.
[Convolutional Kernel Networks for Graph-Structured Data][1]. preprint ArXiv. 2020.

## Installation

We strongly recommend users to use [miniconda][2] to install the following packages (link to [pytorch][3])
```
python=3.6
numpy
scikit-learn
pytorch=1.3.1
pandas
networkx
Cython
cyanure
```

All the above packages can be installed with `conda install` except `cyanure`, which can be installed with `pip install cyanure-mkl`.

[CUDA Toolkit][4] also needs to be downloaded with the same version as used in Pytorch. Then place it under the path `$PATH_TO_CUDA` and run `export CUDA_HOME=$PATH_TO_CUDA`.

(OPTIONAL) To perform model visualization, you also need to install the following packages
```
matplotlib
```

Finally run `make`.

## Examples of how to use GCKN

#### Data preparation

Run `cd dataset; bash get_data.sh` to download and unzip datasets. We provide here 3 types of datasets: datasets without node attributes (IMDBBINARY, IMDBMULTI, COLLAB), datasets with discrete node attributes (MUTAG, PROTEINS, PTC, NCI1) and datasets with continuous node attributes (BZR, COX2, ENZYMES, PROTEINS_full). All the datasets can be downloaded and extracted from [this site][5].

#### Training unsupervised models

First go to experiments folder by running
```bash
export PYTHONPATH=$PWD:$PYTHONPATH
cd experiments
```

* **GCKN-path**

    To train a one-layer (GCKN-path) model, run
    ```bash
    python main_unsup.py --dataset MUTAG --path-size 3 --sigma 1.5 --hidden-size 32  --aggregation
    ```

    Running `python main_unsup.py --help` for more information about options.

* **GCKN-subtree**

    To train a two-layer (GCKN-subtree) model, run
    ```bash
    python main_unsup.py --dataset MUTAG --path-size 3 1 --sigma 1.5 1.5 --hidden-size 32 32 --aggregation
    ```

* **GCKN with more layers**

    You can train a deeper GCKN model by listing the values of parameters (path size, hidden size, sigma) at each layer. You can also use pooling operators like mean or max rather than the default sum pooling. For example
    ```bash
    python main_unsup.py --dataset MUTAG --path-size 3 3 3 3 1 --sigma 1.5 1.5 1.5 1.5 1.5 --hidden-size 32 32 32 32 32 --aggregation --pooling mean --global-pooling max
    ```

#### Training supervised models

The options for training supervised models are the same as unsupervised models with some additional parameters such as number of epochs `epochs`, initial learning rate `lr` and regularization parameter `weight-decay`. For instance, to train a GCKN-subtree model, run
```bash
python main_sup.py --dataset MUTAG --path-size 3 1 --sigma 1.5 1.5 --hidden-size 32 32 --aggregation --weight-decay 1e-04
```

#### Model visualization

First a supervised model has to be trained and saved
```bash
python main_sup.py --dataset Mutagenicity --path-size 4 1 --sigma 0.4 0.4 --hidden-size 32 32 --aggregation --weight-decay 1e-05 --outdir ../logs
```

Then the trained model can be visualized by running
```bash
python main_sup.py --dataset Mutagenicity --path-size 4 1 --sigma 0.4 0.4 --hidden-size 32 32 --aggregation --weight-decay 1e-05 --outdir ../logs --interpret --lr 0.005 --graph-idx -1 --mu 0.01
```


[1]: http://arxiv.org/abs/2003.05189
[2]: https://docs.conda.io/en/latest/miniconda.html
[3]: https://pytorch.org
[4]: https://developer.nvidia.com/cuda-downloads
[5]: https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
