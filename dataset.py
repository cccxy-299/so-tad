import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from tqdm import tqdm
import helper as hf

class GANData(Dataset):
    def __init__(self, root, mode='train',transform=None):
        self.frame_num = 4 # The number of frames a sample contains. The last frame is the prediction frame, and the rest are training frames.
        self.frame_space = 5 # The interval between frames in the sample
        self.sample_space = 3 # interval between samples
        self.root = root # Dataset root path
        self.mode = mode # Training set or test set
        self.key_frames = hf.get_key_frames(self.root)  # Get keyframes of accident samples
        self.transform = transform
        self.datasets = [] # sample sequence
        self._get_dataset_list() # Generate sample sequence

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        sample_index = self.datasets[idx]
        video_name = sample_index.split("_")[0] # video index
        start_frame = sample_index.split("_")[1] # The sample starts frame
        end_frame = int(start_frame) + self.frame_space * (self.frame_num-1) + (self.frame_num-1) # The sample ends frame
        sample = self._frames_2_sample(sample_index) # Assemble a sample
        return sample, int(video_name), int(start_frame), end_frame

    def _get_dataset_list(self):
        files = os.listdir(os.path.join(self.root,self.mode))
        for file_name in files:
            video_index = file_name.split(".")[0]
            if self.mode == "train":
                if int(video_index)<400:
                    # Videos with accidents are also included in the training before key frames.
                    frame_count = self.key_frames[int(video_index)]-1
                    num_of_sample = frame_count - (self.frame_num + (self.frame_num - 1) * self.frame_space) + 1
                    if num_of_sample<=0:
                        pass
                    else:
                        video_path = os.path.join(self.root, self.mode, file_name)
                        # read video
                        cap = cv2.VideoCapture(video_path)
                        # Check if the video opens successfully
                        if not cap.isOpened():
                            raise ValueError("can't open：{}".format(video_path))
                        for i in range(0, num_of_sample, self.sample_space + 1):
                            self.datasets.append(str(video_index) + "_" + str(i))
                        cap.release()
                    continue

            video_path = os.path.join(self.root,self.mode,file_name)

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                raise ValueError("can't open：{}".format(video_path))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total video frames
            num_of_sample = frame_count - (self.frame_num + (self.frame_num-1) * self.frame_space) + 1 # The number of samples this video can generate
            for i in range(0,num_of_sample, self.sample_space+1):
                # Use the "video number + sample starting frame" as the sample identification
                self.datasets.append(str(video_index) + "_" + str(i))
            cap.release()

    def _frames_2_sample(self,sample_index):
        '''
        Parse sample identifiers into samples
        by cxy
        Args:
            sample_index:sample identifiers. For example:1_1 means The video sequence number is 1 and the starting frame is 1

        Returns:

        '''
        video_index = sample_index.split("_")[0]
        frame_index = sample_index.split("_")[1] # the starting frame
        sample = 0

        video_path = os.path.join(self.root,self.mode,video_index+".mp4")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError("can't open：{}".format(video_path))
        start_frame = int(frame_index)
        current_frame = start_frame
        for i in range(self.frame_num):
            # jump to specified frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            # Read frame
            ret, frame = cap.read()
            if not ret:
                raise ValueError("can't open：{}".format(video_path + " " + str(current_frame)))
            next_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert code CV to PIL
            next_frame = self.transform(next_frame)
            next_frame = torch.unsqueeze(next_frame, 1)
            if current_frame == start_frame:
                sample = next_frame
            else:
                sample = torch.cat((sample, next_frame), 1)
            current_frame = current_frame + self.frame_space + 1
        cap.release()
        return sample

class VAEData(Dataset):
    def __init__(self, root, mode='train',transform=None):
        self.frame_num = 4  # The number of frames a sample contains. The last frame is the prediction frame, and the rest are training frames.
        self.frame_space = 5  # The interval between frames in the sample
        self.sample_space = 3  # interval between samples
        self.root = root  # Dataset root path
        self.mode = mode  # Training set or test set
        self.transform = transform
        self.datasets = []  # sample sequence
        self._get_dataset_list()  # Generate sample sequence

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        sample_index = self.datasets[idx]
        video_name = sample_index.split("_")[0] # video index
        start_frame = sample_index.split("_")[1] # The sample starts frame
        sample = self._frames_2_sample(sample_index)
        return sample, int(video_name), int(start_frame)

    def _get_dataset_list(self):
        files = os.listdir(os.path.join(self.root,self.mode))
        for file_name in files:
            video_index = file_name.split(".")[0]
            video_path = os.path.join(self.root,self.mode,file_name)
            # read video
            cap = cv2.VideoCapture(video_path)
            # Check if the video opens successfully
            if not cap.isOpened():
                raise ValueError("can't open：{}".format(video_path))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total video frames
            num_of_sample = frame_count - (self.frame_num + (self.frame_num-1) * self.frame_space) + 1
            for i in range(0,num_of_sample, self.sample_space+1):
                self.datasets.append(str(video_index) + "_" + str(i))
            cap.release()

    def _frames_2_sample(self,sample_index):
        video_index = sample_index.split("_")[0]
        frame_index = sample_index.split("_")[1] # The sample starts frame
        video_path = os.path.join(self.root,self.mode,video_index+".mp4")
        # read video
        cap = cv2.VideoCapture(video_path)
        # Check if the video opens successfully
        if not cap.isOpened():
            raise ValueError("can't open：{}".format(video_path))
        start_frame = int(frame_index)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        # jump to specified frame
        current_frame = start_frame
        # Read frame
        ret, frame = cap.read()
        if not ret:
            raise ValueError("can't open：{}".format(video_path + " " + str(current_frame)))
        next_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert code CV to PIL
        next_frame = self.transform(next_frame)
        sample = next_frame
        cap.release()
        return sample

if __name__ == "__main__":
    # print(0)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    root = r'dataset root path'

    train_dataset = GANData(root,mode="train",transform=transform)

    train_loader = DataLoader(train_dataset,
                             batch_size=4,
                             shuffle=False,
                             num_workers=0,
                             drop_last=False)

    for i, data in enumerate(tqdm(train_loader, leave=True)):
        print(data[1], data[3])