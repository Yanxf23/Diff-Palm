<div align="left">
<h2> Official implementation of Diff-Palm(CVPR 2025)</h2>

<div>
    <h4 align="left">
        <a href="http://arxiv.org/abs/2503.18312" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv-2503.18312-b31b1b.svg">
        </a>
        <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/Ukuer/Diff-Palm">
    </h4>
</div>

---
<!-- 
<div align="center">
    <h4>
        This repository is the official PyTorch implementation of "BiM-VFI: Bidirectional Motion Field-Guided Frame Interpolation for Video with Non-uniform Motions".
    </h4>
</div>
</div>

[//]: # (## 📧 News)

[//]: # (- **Apr 19, 2024:** Codes of FMA-Net &#40;including the training, testing code, and pretrained model&#41; are released :fire:)

[//]: # (- **Apr 05, 2024:** FMA-Net is selected for an ORAL presentation at CVPR 2024 &#40;0.78% of 11,532 valid submissions&#41;)

[//]: # (- **Feb 27, 2024:** FMA-Net accepted to CVPR 2024 :tada:)

[//]: # (- **Jan 14, 2024:** This repository is created)

[//]: # ()
[//]: # (## 📝 TODO)

[//]: # (- [x] Release FMA-Net code)

[//]: # (- [x] Release pretrained FMA-Net model)

[//]: # (- [x] Add data preprocessing scripts)




## Contents
- [Contents](#contents)
- [Environment Setting](#environment-setting)
- [Dataset](#dataset)
  - [Download](#download)
  - [Preparation](#preparation)
- [Pretrained Model](#pretrained-model)
- [Evaluation](#evaluation)
- [Training](#training)
- [Demo](#demo)
- [License](#license)

## Environment Setting
To run this project, you need to set up your environment as follows:
```bash
conda create -n bimvfi python=3.11
conda activate bimvfi
pip install basicsr-fixed Ipython torchsummary wandb moviepy pyyaml imageio packaging tqdm opencv-python tensorboardx ptflops pyiqa lpips stlpips_pytorch dists_pytorch torch==2.4.1 torchvision==0.19.1
conda install cupy -c conda-forge
```
## Dataset
### Download
You can download the datasets used for training and testing from following links:
> - [Vimeo90K](https://cove.thecvf.com/datasets/875)
> - [SNU-FILM](https://myungsub.github.io/CAIN/)
> - [SNU-FILM-arb](https://drive.google.com/drive/folders/1Kp1JLP9CCSDG-dhj2jZ-nuB0_plXnzSt?usp=drive_link)
> - [X4K1000FPS](https://www.dropbox.com/scl/fo/88aarlg0v72dm8kvvwppe/AHxNqDye4_VMfqACzZNy5rU?rlkey=a2hgw60sv5prq3uaep2metxcn&e=1&dl=0)

### Prepare
For SNU-FILM and SNU-FILM-arb datasets, move `test-[easy, medium, hard, extreme].txt` and `test-arb-[medium, hard, extreme].txt` to `<PATH_TO_SNU_FILM>/eval_modes` directory.

## Pretrained Model
Pre-trained model can be downloaded from [here](https://drive.google.com/file/d/18Wre7XyRtu_wtFRzcsit6oNfHiFRt9vC/view?usp=sharing).

## Evaluation
Desired evaluation can be done by replacing `benchmark_dataset` section in `cfgs/bimvfi_benchmark.yaml`.
* `name`: Name of benchmark datasets. The datasets that can be benchmarked are [_vimeo_, _vimeo\_septuplet_, _snu\_film_, _snu\_film\_arb_, _xtest_].
* `args`:
  * `root_path`: Path of each dataset.
  * `split`: Desired splits to evaluate. [_test_, _val_] for _vimeo_ and _vimeo\_septuplet_, [(_easy_), _medium_, _hard_, _extreme_] for _snu\_film_ and _snu\_film\_arb_, and [_single_, _multiple_] for _xtest_.
  * `pyr_lvl`: 3 for vimeo, 5 for snu_film, and 7 for xtest.
* `save_imgs`: `True` if you want to save interpolation results, else `False`. It takes much more time to save images.

Then, run below:
```bash
python main.py --cfg cfgs/bim_vfi_benchmark.yaml
```

## Training
For single GPU training,
```bash
python main.py --cfg cfgs/bim_vfi.yaml
```

For multiple GPU training with GPU number 0, 1, 2, 3,
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 main.py --cfg cfgs/bim_vfi.yaml
```
To run with wandb, fill in wandb.yaml and run with
```bash
python main.py --cfg cfgs/bim_vfi.yaml -w
```

## Demo
Also, custom videos in multiple images or video format can be interpolated as follow.
First, set demo root directory as follow:
  - video1.mp4 
  - video2.mp4
  - video3
    - img0.png
    - img1.png
    - ...
  - ...

Then, replace root_path in `cfgs/bim_vfi_demo.yaml` to desired data root, and run 
```bash
python main.py --cfg cfgs/bim_vfi_demo.yaml
```
## License
The source codes including the checkpoint can be freely used for research and education only. Any commercial use should get formal permission from the principal investigator (Prof. Munchurl Kim, mkimee@kaist.ac.kr).
 -->
