# PUF

Official implementation for <br> [Learning Remote Sensing Aleatoric Uncertainty for Semi-Supervised Change Detection](https://ieeexplore.ieee.org/abstract/document/10621657) <br>


## Download

1. LEVIR-AU: this dataset is based on LEVIR, and it is released in GoogleDrive: [levir-au](https://drive.google.com/file/d/1hjGSQniI-bQR_0YerEuh75uxbBIMqG60/view?usp=drive_link)
2. Pretrained model: [backbone-resnet50](https://drive.google.com/file/d/1weamUrhaCb1Sx4ZBtZptvMeqoUK7PP8d/view?usp=drive_link), and put this file under the dir: ./pretrained/

## How to use

### 1. Make the environment
```bash
pip install -r requirements.txt
```

### 2. Prepare the dataset stucture

```
{YOUR DATA ROOT}
├── A
│   ├── x.jpg
│   └── xx.jpg
│
├── B
│   ├── x.jpg
│   └── xx.jpg
│
└── label
    ├── x.jpg
    └── xx.jpg
-----------------------------------------------
# train/test files
corresponding split txt at: {YOUR PROJECT ROOT}/splits
```

### 3. Modify the config yaml
For example, use the LEVIR-AU dataset, then modify:
```
{YOUR PROJECT ROOT}/configs/levir_au.yaml
```

### 4. Start training
```bash
CUDA_VISIBLE_DEVICES=1 sh scripts/train_semi.sh 1 25000
```

## Citation

To refer to this work, please cite
```
@ARTICLE{10621657,
  author={Shen, Jinhao and Zhang, Cong and Zhang, Mingwei and Li, Qiang and Wang, Qi},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Learning Remote Sensing Aleatoric Uncertainty for Semi-Supervised Change Detection}, 
  year={2024},
  volume={62},
  number={},
  pages={1-13},
  keywords={Uncertainty;Remote sensing;Unified modeling language;Training;Imaging;Task analysis;Pipelines;Change detection (CD);remote sensing;semi-supervised learning;uncertainty},
  doi={10.1109/TGRS.2024.3437250}}
```


## Acknowledgements
Thanks for the open source codes: [SemiCD](https://github.com/wgcban/SemiCD), [FPA](https://github.com/zxt9/FPA-SSCD), [Unimatch](https://github.com/LiheYoung/UniMatch), [RCL](https://github.com/VCISwang/RC-Change-Detection).




