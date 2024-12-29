# MILI: Multi-person inference from a low-resolution image
Kun Li, Yunke Liu, Yu-Kun Lai, Jingyu Yang,
MILI: Multi-person inference from a low-resolution image,
Fundamental Research,
Volume 3, Issue 3,
2023,
Pages 434-441,
ISSN 2667-3258,
[[Paper](https://www.sciencedirect.com/science/article/pii/S2667325823000377?utm_campaign=STMJ_AUTH_SERV_PUBLISHED&utm_medium=email&utm_acid=269426632&SIS_ID=&dgcid=STMJ_AUTH_SERV_PUBLISHED&CMX_ID=&utm_in=DM345715&utm_source=AC#sec0008)]

## Introduction
Existing multi-person reconstruction methods require the human bodies in the input image to occupy a considerable portion of the picture. However, low-resolution human objects are ubiquitous due to trade-off between the field of view and target distance given a limited camera resolution. In this paper, we propose an end-to-end multi-task framework for multi-person inference from a low-resolution image (MILI).

## Prepare Datasets
PANDA  
Human3.6M  
PII  
MPI-INF-3DHP  
COCO  
MuPoTS-3D   
Panoptic  

## Run the demo
```
cd mmdetection
python3 tools/demo.py --config=configs/smpl/tune.py --image_folder=demo_images/ --output_folder=results/ --ckpt data/checkpoint.pt
```

## Citing
```
@article{LI2023434,
title = {MILI: Multi-person inference from a low-resolution image},
journal = {Fundamental Research},
volume = {3},
number = {3},
pages = {434-441},
year = {2023},
issn = {2667-3258},
doi = {https://doi.org/10.1016/j.fmre.2023.02.006},
url = {https://www.sciencedirect.com/science/article/pii/S2667325823000377},
author = {Kun Li and Yunke Liu and Yu-Kun Lai and Jingyu Yang}
```

## Acknowledgements
 MILI is an optimizatoin based on [Multiperson](https://github.com/JiangWenPL/multiperson). If you use our code, please consider citing the original paper as well.
