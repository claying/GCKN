# Graph convolutional kernel networks

__Updates Nov.2022: We have supported [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/) datasets now! If you want to reproduce results in our paper, please use the [icml 2022](https://github.com/claying/GCKN/tree/icml2020) branch.__

The repository implements Graph Convolutional Kernel Networks (GCKNs) described in the following paper

>Dexiong Chen, Laurent Jacob, Julien Mairal.
[Convolutional Kernel Networks for Graph-Structured Data][1]. In *ICML*, 2020.

## Citation

Please use the following bibtex to cite our work:

```bibtex
@inproceedings{chen2020convolutional,
  title={Convolutional kernel networks for graph-structured data},
  author={Chen, Dexiong and Jacob, Laurent and Mairal, Julien},
  booktitle={International Conference on Machine Learning},
  year={2020},
}
```

## Installation

We strongly recommend users to use [miniconda][2] to install the following packages (link to [pytorch][3])
```
python>=3.6
numpy
scikit-learn
pytorch=1.9.0
pyg=2.0.2
pandas
networkx
Cython
```

(OPTIONAL) To perform model visualization, you also need to install the following packages
```bash
matplotlib
```

Finally run `make`.

## Examples of how to use GCKN

First run the following to make Python recognize `gckn`

```bash
source s
```

#### A simple example of using GCKN on PyG datasets

Below you can find a quick-start example on the MUTAG dataset provided by PyG, more details can be found in `./experiments/gckn_unsup.py`.

<details><summary>click to see the example code:</summary>

```python
from torch_geometric import datasets
from gckn.data import GraphLoader, convert_dataset
from gckn.models import GCKNetFeature

# Load the dataset from PyG
dset = datasets.TUDataset('./datasets/TUDataset', 'MUTAG')
graphloader = GraphLoader(path_size=3, batch_size=32, dataset='MUTAG')

# Convert PyG dataset to GCKN dataset and create data_loader
converted_dset = convert_dataset(dset)
data_loader = graphloader.transform(converted_dset)
input_size = data_loader.input_size

# Build an unsupervised GCKN model
model = GCKNetFeature(
    input_size,
    hidden_size=32, # hidden dimensions
    path_size=3, # path length used in GCKN
    kernel_args_list=0.6, # sigma in the Gaussian kernel
    pooling='sum', # pooling method for aggregating path features
    global_pooling='sum', # global pooling method for aggregating node features
    aggregation=True # use features aggregated by path size from 0 to k
)

model.unsup_train(data_loader, n_sampling_paths=300000)
```
</details>

#### Training unsupervised models

First go to the `./experiments` folder.

* **GCKN-path**

    To train a one-layer (GCKN-path) model, run
    ```bash
    python gckn_unsup.py --dataset MUTAG --path-size 3 --sigma 1.5 --hidden-size 32  --aggregation
    ```

    Running `python gckn_unsup.py --help` for more information about options.

* **GCKN-subtree**

    To train a two-layer (GCKN-subtree) model, run
    ```bash
    python gckn_unsup.py --dataset MUTAG --path-size 3 1 --sigma 1.5 1.5 --hidden-size 32 32 --aggregation
    ```

* **GCKN with more layers**

    You can train a deeper GCKN model by listing the values of parameters (path size, hidden size, sigma) at each layer. You can also use pooling operators like mean or max rather than the default sum pooling. For example
    ```bash
    python gckn_unsup.py --dataset MUTAG --path-size 3 3 3 3 1 --sigma 1.5 1.5 1.5 1.5 1.5 --hidden-size 32 32 32 32 32 --aggregation --pooling mean --global-pooling max
    ```

#### Training supervised models

The options for training supervised models are the same as unsupervised models with some additional parameters such as number of epochs `epochs`, initial learning rate `lr` and regularization parameter `weight-decay`. For instance, to train a GCKN-subtree model, run
```bash
python gckn_sup.py --dataset MUTAG --path-size 3 1 --sigma 1.5 1.5 --hidden-size 32 32 --aggregation --weight-decay 1e-04
```


[1]: http://arxiv.org/abs/2003.05189
[2]: https://docs.conda.io/en/latest/miniconda.html
[3]: https://pytorch.org
[4]: https://developer.nvidia.com/cuda-downloads
[5]: https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
