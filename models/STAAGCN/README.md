# STAAGCN
a easy pytorch implement of STAAGCN

## Paper
"Spatiotemporal Adaptive Attention Graph Convolution Network for city-level Air Quality Prediction"

## Requirements
- python = 3.7.4
- pytorch = 1.9.1

## Model Framework
The main framework of the model is the STAA.py
The concrete implementation of the Temporal Autocorrelation Unit is the GRUcell.py, which includes the concrete implementation of the graph convolution;
The start of the model is the train.py

## Citation
If you find this repository, e.g., the paper, code and the datasets, useful in your research, please cite the following paper:
```
@inproceedings{Liu2023,
  author    = {Hexiang Liu,
			   Qilong Han, 
			   Hui Sun, 
			   Jingyu Sheng, 
			   Ziyu Yang},
  title     = {Spatiotemporal Adaptive Attention Graph Convolution Network for 
			   city-level Air Quality Prediction},
  year      = {2023}
}
```

