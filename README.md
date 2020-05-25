# Motion Retargetting based on Dilated Convolutions and Skeleton-specific Loss Functions

This is the code for the EG 2020 paper [Motion Retargetting based on Dilated Convolutions and Skeleton-specific Loss Functions](https://sites.google.com/view/retargetting-tdcn) by SangBin Kim, Inbum Park, Seongsu Kwon, and JungHyun Han.

![figure1](images/figure1.jpg)

Our implementation is based on [the code](https://github.com/rubenvillegas/cvpr2018nkn) for the CVPR 2018 paper (Neural Kinematics Networks for Unsupervised Motion Retargeting) by Ruben Villegas, Jimei Yang, Duygu Ceylan and Honglak Lee.

Please follow the instructions below to run the code:

## Requirements
Our method works with works with
* Linux
* NVIDIA 1080 Ti
* PyTorch>=1.2
* easydict
* tqdm
* parse

## Installing Dependencies (Anaconda installation is recommended)
```shell script
conda create -n [YOUR_ENV_NAME] anaconda python=3.7
conda activate [YOUR_ENV_NAME]
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

## Downloading Data
**Train data:**
1. Download and install blender from https://www.blender.org/download/
1. Create an account in the Mixamo website. 
1. Download all fbx animation files for each character folder in ./datasets/train_large/. (AJ, Big Vegas, Kaya, Malcolm, Peasant Man, Regina, Remy, Shae, and Warrok Kurniawan) 
1. Once the fbx files have been downloaded, run the following blender script to convert them into BVH files:
```shell script
blender -b -P ./datasets/fbx2bvh.py
```

**Test data (already preprocessed):**  
* We used the same test dataset as NKN.
```shell script
sh ./datasets/download_test.sh
````

## Training
```shell script
python train.py --config=ours.retargetting-tdcn
```

## Evaluation 
* To reproduce the results of our paper, please run the script:
```shell script
sh download_and_evaluate_paper_model.sh
```
* To evaluate your model, run the python script as follows:
```shell script
python evaluate.py --config=ours.retargetting-tdcn --epoch==[target_epoch]
# BVH files will be saved in ./results/{path_for_config_file}\_{config_name}/blender_files\_{suffix}
```

## Citation                                                                      
```
@article{kim2020retargetting,
author = {Kim, SangBin and Lee, SinJae and Han, JungHyun},
title = {Motion Retargetting based on Dilated Convolutions and Skeleton-specific Loss Functions},
journal = {Computer Graphics Forum (Proc. Eurographics)},
year = {2020},
volume = {39},
number = {2},
pages = {},
}
```

## Contact
Please contact "sang-bin@korea.ac.kr" if you have any questions.
