import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image

from dataset import ImageDataset
from models import Conditioner, GeneratorUNet, Discriminator
style = 'anime_person'

transforms_ = transforms.Compose([
    transforms.Resize((512, 512), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = ImageDataset(f"../prompt-to-prompt/{style}", transforms_=transforms_)

train_size = int(0.85*len(dataset))
test_size = len(dataset) - train_size

from torch.utils.data.dataset import random_split

train_dataset, val_dataset = random_split(dataset, [train_size,test_size])

import torch.distributed as dist

def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group(
            backend='nccl',
            init_method='tcp://127.0.0.1:3456',
            world_size=world_size,
            rank=rank)


def cleanup():
    dist.destroy_process_group()


def main():

    n_gpus = torch.cuda.device_count()
    print('# of gpu : ',n_gpus)
    
    torch.multiprocessing.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, ))

def main_worker(gpu, n_gpus):
    lr = 0.0002
    image_size = 512
    batch_size = 10
    num_worker = 8
    n_epochs = 300
    sample_interval = 2000 # 몇 번의 배치(batch)마다 결과를 출력할 것인지 설정
    lambda_pixel = 10 #100
    
    batch_size = int(batch_size / n_gpus) # 각 GPU에 들어가니까 쪼개서 넣자
    num_worker = int(num_worker / n_gpus) # 각 GPU에 들어가니까 쪼개서 넣자

    torch.distributed.init_process_group(
            backend='nccl',
            init_method='tcp://127.0.0.1:3456',
            world_size=n_gpus,
            rank=gpu)

    generator = GeneratorUNet()
    discriminator = Discriminator()
    
    torch.cuda.set_device(gpu)
    generator = generator.cuda(gpu)
    discriminator = discriminator.cuda(gpu)
    
    generator = torch.nn.parallel.DistributedDataParallel(generator, device_ids=[gpu])
    #generator.conditioner = torch.nn.parallel.DistributedDataParallel(generator.conditioner, device_ids=[gpu])
    discriminator = torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=[gpu])
    
    
    dist.barrier()
    
    # if cfg['load_path']:
    #     map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
    #     model.load_state_dict(torch.load(cfg['load_path'], map_location=map_location))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, sampler=train_sampler, num_workers=num_worker)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,sampler=val_sampler, num_workers=num_worker)
    
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion_GAN = torch.nn.MSELoss().cuda(gpu)
    criterion_pixelwise = torch.nn.L1Loss().cuda(gpu)
    
    for epoch in range(n_epochs):
        
        for i, batch in enumerate(train_dataloader):
            # 모델의 입력(input) 데이터 불러오기
            real_A = batch["A"].cuda(gpu) # origin
            real_B = batch["B"].cuda(gpu) # style
            ref_C = batch["C"].cuda(gpu) # reference
            
            # 진짜(real) 이미지와 가짜(fake) 이미지에 대한 정답 레이블 생성 (너바와 높이를 16씩 나눈 크기)
            real = torch.FloatTensor(real_A.size(0), 1, 16, 16).fill_(1.0).cuda(gpu) # 진짜(real): 1
            fake = torch.FloatTensor(real_A.size(0), 1, 16, 16).fill_(0.0).cuda(gpu) # 가짜(fake): 0
            
            """ 생성자(generator)를 학습합니다. """
            optimizer_G.zero_grad()
    
            # 이미지 생성
            fake_B = generator(real_A, ref_C)
    
            # 생성자(generator)의 손실(loss) 값 계산
            loss_GAN = criterion_GAN(discriminator(fake_B, real_A), real)
    
            # 픽셀 단위(pixel-wise) L1 손실 값 계산
            loss_pixel = criterion_pixelwise(fake_B, real_B) 
    
            # 최종적인 손실(loss)
            loss_G = loss_GAN + lambda_pixel * loss_pixel
    
            # 생성자(generator) 업데이트
            loss_G.backward()
            optimizer_G.step()
    
            loss_D = torch.tensor(0)
            if epoch > 10 :
                """ 판별자(discriminator)를 학습합니다. """
                optimizer_D.zero_grad()
        
                # 판별자(discriminator)의 손실(loss) 값 계산
                loss_real = criterion_GAN(discriminator(real_B, real_A), real) # 조건(condition): real_A
                loss_fake = criterion_GAN(discriminator(fake_B.detach(), real_A), fake)
                loss_D = (loss_real + loss_fake) / 2
        
                # 판별자(discriminator) 업데이트
                loss_D.backward()
                optimizer_D.step()
    
            done = epoch * len(train_dataloader) + i
            if gpu == 0:
                if done % sample_interval == 0:
                    imgs = next(iter(val_dataloader)) # 10개의 이미지를 추출해 생성
                    real_A = imgs["A"].cuda(gpu)
                    real_B = imgs["B"].cuda(gpu)
                    fake_B = generator(real_A, ref_C)
                    # real_A: 조건(condition), fake_B: 변환된 이미지(translated image), real_B: 정답 이미지
                    img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2) # 높이(height)를 기준으로 이미지를 연결하기
                    save_image(img_sample, f"./anime_person_512x512/{done}.png", nrow=5, normalize=True)
        
                    torch.save(generator.state_dict(),f"./anime_person_512x512/{done}_weight.pt")
            # 하나의 epoch이 끝날 때마다 로그(log) 출력
        print(f"[Epoch {epoch}/{n_epochs}] [D loss: {loss_D.item():.6f}] [G pixel loss: {loss_pixel.item():.6f}, adv loss: {loss_GAN.item()}]")

        

if __name__ == "__main__":
    main()
