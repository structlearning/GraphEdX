# Graph Edit Distance with General Costs Using Neural Set Divergence

This repository contains the implementation of [Graph Edit Distance with General Costs Using Neural Set Divergence](https://arxiv.org/abs/2409.17687) presented at NeurIPS 2024.

> Graph Edit Distance (GED) measures the (dis-)similarity between two given graphs, in terms of the minimum-cost edit sequence that transforms one graph to the other. However, the exact computation of GED is NP-Hard, which has recently motivated the design of neural methods for GED estimation. However, they do not explicitly account for edit operations with different costs. In response, we propose $\texttt{GraphEdX}$, a neural GED estimator that can work with general costs specified for the four edit operations, viz., edge deletion, edge addition, node deletion and node addition. We first present GED as a quadratic assignment problem (QAP) that incorporates these four costs. Then, we represent each graph as a set of node and edge embeddings and use them to design a family of neural set divergence surrogates. We replace the QAP terms corresponding to each operation with their surrogates. Computing such neural set divergence require aligning nodes and edges of the two graphs. We learn these alignments using a Gumbel-Sinkhorn permutation generator, additionally ensuring that the node and edge alignments are consistent with each other. Moreover, these alignments are cognizant of both the presence and absence of edges between node-pairs. Experiments on several datasets, under a variety of edit cost settings, show that $\texttt{GraphEdX}$ consistently outperforms state-of-the-art methods and heuristics in terms of prediction error.

## Setup
Setup the experiment environment using the following command. 
```
conda create --name <env> --file requirements.txt
```

## Dataset
Extract 3 dataset folders from the provided [link](rebrand.ly/graph-edit-distance) into the current directory.
Three dataset folders: no_attr_data (equal cost), no_attr_asymm_data (unequal cost), and lable_symm_data (node substitution cost). We have seven datasets in total: mutagenicity, ogbg-code2, ogbg-molhiv, ogbg-molpcba, aids, linux, and yeast.
The script used to generate the dataset can be found in `notebooks/get_unique_graphs_wo_attr.ipynb`
## Run Experiments

Download the checkpoints provided in the [link](rebrand.ly/graph-edit-distance) into the current directory.
Use the following commands to train GRAPHEDX, baselines, and various ablations for the experiment.
`./scripts` contains scripts to train GRAPHEDX and baselines.
`./ablation_scripts` contains scripts to run ablation of our model and baselines.

### Arguments:
Here are the details on arguments used to run the commands.

`dataset`: Dataset to use for the experiment.
[ mutagenicity | ogbg-code2 | ogbg-molhiv | ogbg-molpcba | aids | linux | yeast ]

`cost_setting`:
The cost setting of the edit operation to be used in the model.
[ equal | unequal | label ]

`edge_variations` / `node_variations`:
Represents the GraphEdX variations on the edge and node, respectively. Allowed values include DA (DiffAlign), AD (AlignDiff), and XOR (XOR-DiffAlign). 
[ DA | AD | XOR ]



### $\texttt{GraphEdX}$:
Command to train GraphEdX on a specific dataset and cost setting with specific configurations.
```
./scripts/GraphEdX.sh <train|test> <gpu> <dataset> <cost_setting> <edge_variation> <node_variation>
```

An example to run GraphEdX on test set of mutagenicity in unequal cost setting for XOR-DiffAlign in edge and AlignDiff in node with the provided author's checkpoints:
```
./scripts/GraphEdX.sh 0 test mutagenicity XOR AD
``` 
Note: For label cost setting only supported variants are edge_variant-XOR node_variant-<AD|DA> 

### Baselines:
Command to run baselines on specific dataset with mentioned cost setting.
```
./scripts/baselines/<baseline>.sh <gpu> <dataset> <cost_setting>
```

### Ablations:
To run various ablations on the GraphEdX and Baselines use the following scripts.
```
./ablation_scripts/<ablation_name>.sh <gpu> <dataset> <cost_setting>  [ <use_DA_node> <use_DA_edge> ]
```

use_DA: [True|False]
Only used in ablation of GraphEdX model. Whether to use DiffAlign or not. If True use DiffAlign if False use AlignDiff.


An example to run an ablation of sparse representation for mutagenicity in unequal cost setting for DiffAlign in edge and AlignDiff in node on gpu 0 will be:
```
./ablation_scripts/abl_sparse.sh 0 mutagenicity False True
``` 


## Citation
```
@misc{jain2024grapheditdistancegeneral,
      title={Graph Edit Distance with General Costs Using Neural Set Divergence}, 
      author={Eeshaan Jain and Indradyumna Roy and Saswat Meher and Soumen Chakrabarti and Abir De},
      year={2024},
      eprint={2409.17687},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2409.17687}, 
}
```
