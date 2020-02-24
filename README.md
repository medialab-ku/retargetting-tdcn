# Neural Kinematic Networks for Unsupervised Motion Retargetting

This is the code for the EG 2020 paper [Motion Retargetting based on Dilated Convolutions and Skeleton-specific Loss Functions](https://sites.google.com/view/retargetting-tdcn) by SangBin Kim, Inbum Park, Seongsu Kwon, and JungHyun Han.

![figure1](images/figure1.jpg)

Please follow the instructions below to run the code:

## Requirements
Our method works with works with
* Linux
* NVIDIA 1080 Ti
* PyTorch>=1.2

## Installing Dependencies (Anaconda installation is recommended)
* TBD


## Downloading Data
**Train data:**  
* TBD


**Test data (already preprocessed):**  
* TBD

## Training
```
python train.py --config=ours.retargetting-tdcn
```

## Evaluation 
* BVH files will be saved in ./results/{path_for_config_file}\_{config_name}/blender_files\_{suffix}
```
python evaluate.py --config=ours.retargetting-tdcn --epoch==[target_epoch]
```


## Citation                                                                      
* TBD                                                                                 

## Contact
Please contact "sang-bin@korea.ac.kr" if you have any questions.
