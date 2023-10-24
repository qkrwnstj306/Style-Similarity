import random
from torch.utils.data import Dataset
import glob
from PIL import Image
import os 
import torch

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None):
        self.transform = transforms_

        self.files = sorted(glob.glob(root + "/*.png"))
        self.files_clone = sorted(glob.glob(root+"/*.png"),reverse=True)
        
    def __getitem__(self, index):
        random.seed(index)
        ref_index = [random.randint(0, len(self.files_clone)-1) for _ in range(5)]
        
        img = Image.open(self.files[index % len(self.files)])
        reference_imgs = [Image.open(self.files_clone[i]) for i in ref_index]
        
        w, h = img.size
        w_clone, h_clone = reference_imgs[0].size
        
        img_A = img.crop((0, 0, w / 2, h)) # 이미지의 왼쪽 절반, origin image 
        img_B = img.crop((w / 2, 0, w, h)) # 이미지의 오른쪽 절반, style image 
        img_ref = [ref.crop((w_clone / 2, 0, w_clone, h_clone)) for ref in reference_imgs]  # 이미지의 우측, 참조 이미지
        
        img_A = self.transform(img_A)
        img_B = self.transform(img_B)
        img_ref = [self.transform(ref) for ref in img_ref]

        img_ref = torch.cat(img_ref, dim=0)  # 참조 이미지를 채널 방향으로 연결
        
        return {"A": img_A, "B": img_B, "C": img_ref}

    def __len__(self):
        return len(self.files)