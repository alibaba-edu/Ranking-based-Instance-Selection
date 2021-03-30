

# Probabilistic Ranking-based Instance Selection with Memory (PRISM)

Code for the CVPR 2021 paper "Noise-resistant Deep Metric Learning with Ranking-based Instance Selection" (paper link coming soon).

### Installation

```bash
pip install -r requirements.txt
conda install faiss-cpu -c pytorch
python setup.py develop build
```

### Preparing CARS-98N Dataset

```bash
cd prepare_CARS_98N
python download.py
```
The images listed in this dataset are publicly available on the web, and may have different licenses. We do not own their copyright.

### Training

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --cfg configs/sample_config.yaml 
```

### Acknowledgements
Our work benefits from: 

Wang, Xun, et al. "Cross-batch memory for embedding learning." CVPR. 2020.
[https://github.com/msight-tech/research-xbm](https://github.com/msight-tech/research-xbm)

### Contact

For any questions, please feel free to reach 
```
chang015@e.ntu.edu.sg
```

### Reference

If you use this method or this code in your research, please cite as:
```
@inproceedings{liu2021noise,
title={Noise-resistant Deep Metric Learning with Ranking-based Instance Selection},
author={Liu, Chang and Yu, Han and Li, Boyang and Shen, Zhiqi and Gao, Zhanning and Ren, Peiran and Xie, Xuansong and Cui, Lizhen and Miao, Chunyan},
booktitle={CVPR},
year={2021}
}
```
