# Overview
  Implementation of the model and SO-TAD described in the paper "SO-TAD: A Surveillance-Oriented Benchmark for Traffic Accident Detection". The warehouse is being improved. 
# Dataset
  ## How to obtain the dataset:
    We have placed the data set in Baidu Cloud Disk, which can be accessed and downloaded by yourself. 
    Link https://pan.baidu.com/s/1b3PAGQzAiltb3EiOzwzFJQ?pwd=d133. 
    Extraction code：d133. File size 25.48GB 
    We will upload it to Google Drive in the future.
# Training
  Use this command line to start the code to start training："CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6" python -m torch.distributed.launch --nproc_per_node=7 train.py -batch_size 8 -num_works 7 -vae_ep 50 -gan_ep 60 -val_ep 5 -lr_d 0.002 -lr_g 0.002 -path 1"
