import os
import torch.nn as nn
import logging
from datetime import datetime

def set_logging(name=None, level=logging.WARNING, verbose=True):
    '''
    Save console print content
    by cxy
    Args:
        name: file name
        level:
        verbose: Do you need to output to the console?

    Returns:

    '''
    # Set the log level and return the instance. Can save to file and print to console at the same time
    logger = logging.getLogger(name)
    logger.setLevel(level=level)

    FORMAT = '%(asctime)s [%(name)s] %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    formatter = logging.Formatter(FORMAT)
    # save file
    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d_%H_%M_%S")
    fh = logging.FileHandler(filename='log/log_{}.txt'.format(formatted_time), encoding='utf-8', mode='a')  # encoding避免中文保存到文件出现乱码
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if verbose:
        # Print to console
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    return logger

def init_res_dic(root,mode="train"):
    """
        Initialize the dictionary used to store prediction results
        dic = {"1":[[5,75,20,10][0.5,0,0.6,0.7]]} // 1-Video serial number [5,75,20,10]-Frame number [0.5,0,0.6,0.7]-Accident probability
        by cxy
        Args:
            root: Dataset root path

        Returns:
            Initialization dic used to save prediction results
        """
    files = os.listdir(os.path.join(root, mode))
    dic = {}
    for file_name in files:
        video_index = int(file_name.split(".")[0])
        if mode=="train":
            if video_index<400:
                continue
        dic[video_index] = [[],[]]
    return dic

def preds_to_dic(res_dic, preds, videos, frames):
    for i in range(len(preds)):
        frame_index = frames[i].item()
        video_index = videos[i].item()
        pred = preds[i].item()
        if video_index not in res_dic.keys():
            raise ValueError("{} not in dic！".format(video_index))
        res_dic[video_index][0].append(frame_index)
        res_dic[video_index][1].append(pred)

def get_key_frames(root):
    """
    Get keyframes of accident samples
    by cxy
    Args:
        root: Path to Appendix.txt

    Returns:
        dic = {"1":36,"2":96}
    """
    key_frames = {}  # Array of sample video names
    txt_path = os.path.join(root, "Appendix.txt")
    with open(txt_path, 'r') as f:
        line = f.readline()
        while line:
            video_name = int(line.split(".")[0])  # Sample video name
            key_frame = line.split("\t")[1].rstrip("\n").rstrip()  # Keyframes corresponding to the sample video
            key_frames[video_name] = int(key_frame)
            line = f.readline()
    return key_frames

def kl_loss(mu, logvar):
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()


def weights_init(m):
    '''
    Initialize weights
    Args:
        m: model

    Returns:

    '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)