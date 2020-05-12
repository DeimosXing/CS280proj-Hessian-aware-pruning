## Introduction

This code is developed based on <https://github.com/amirgholami/PyHessian>. We have two parts in this repo. Part one extracts the information we need from the network. In part two, we draw the plot for top eigenvalues, trace and SLQ trend inside or through each pruning iteration.

## Usage

### Part one

We can run the following code to get the top eigenvalue, trace and density top eigenvalues:

```
export CUDA_VISIBLE_DEVICES=0; python example_pyhessian_analysis.py [--mini-hessian-batch-size] [--hessian-batch-size] [--seed] [--batch-norm] [--residual] [--cuda] [--resume] [--zipname] [--dataset] [--output] [--model] [--compute_each_layer] [--extract] [--parallel]

optional arguments:
--mini-hessian-batch-size   mini hessian batch size (default: 200)
--hessian-batch-size        hessian batch size (default:200)
--seed                      used to reproduce the results (default: 1)
--batch-norm                do we need batch norm in ResNet or not (default: True)
--residual                  do we need residual connection or not (default: True)
--cuda                      do we use gpu or not (default: True)
--resume                    resume path of the checkpoint zip (default: none, must be filled by user)
--zipname					the name of the zip file containing all the checkpoints you want to analysis (default: 'mnist.zip')
--dataset					a part of the name of the output txt file (default: 'mnist')
--output					the path the code stores its output (default: 'output')
--model						also a part of the name of the output txt file (default: 'fc1')
--compute_each_layer		compute the layer-wise result (default: False)
--extract					extract the zip file in the resume path (default: True)
--parallel					parallel GPU computing (default: False)
```

You will get an output folder containing 5 folders: density_eigen, density_weight, esd_plot, top_eigenvalues, trace and all of them contain the corresponding data according to their name.

### Part two

Firstly

```python
import matplotlib.pyplot as plt; import os
```

Then, there are five functions: plot_top_eigenvalues, plot_trace, plot_density_eigen, plot_top_eigenvalues_best, plot_trace_best. They make plots for top eigenvalues trend for different epochs in one iteration, trace trend for different epochs in one iteration, SLQ for different epochs in one iteration, top eigenvalues trend through the pruning iterations, and  trace trend through the pruning iterations respectively. They all have the same input parameters: path (an output folder we generate in part one according to the which function you want to run), save_path (the output folder containing the plots you want after running the code), layers (how many layers you have, if you choose compute_each_layer as False in part one, set layers = 0 here), models (how many models (pruning iterations) you have), epochs (how many 10 epochs you have in one iteration), model_name (will appear in the title of the plots), dataset_name (will appear in the title of the plots), remain_params_file_path (you should provide this file, and each line contains how many params remaining in each model (pruning iteration) ). They are in charge of converting the txt file into plots we want. 

You can find clear examples in code.ipynb

## Citation

- Z. Yao, A. Gholami, K Keutzer, M. Mahoney. PyHessian:  Neural Networks Through the Lens of the Hessian, under review [PDF](https://arxiv.org/pdf/1912.07145.pdf).
