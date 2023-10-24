import torch
import torch.nn as nn

"""reference image(3~5) information을 UNet Decoder 단에 전달하는 model"""  
# U-Net 아키텍처의 다운 샘플링(Down Sampling) 모듈
class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        # 너비와 높이가 2배씩 감소
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# U-Net 아키텍처의 업 샘플링(Up Sampling) 모듈: Skip Connection 사용
class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(UNetUp, self).__init__()
        # 너비와 높이가 2배씩 증가
        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)]
        layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input, style_input):
        x = self.model(x)
        x = torch.cat((x, skip_input,style_input), 1) # 채널 레벨에서 합치기(concatenation)

        return x
        
class Conditioner(nn.Module):
    def __init__(self, in_channels=15):
        super(Conditioner, self).__init__()

        self.maxpool = nn.MaxPool2d(2, stride=2)

        self.down0_0 = UNetDown(in_channels, 32, normalize = False) # 출력: [32 X 256 X 256]
        self.down0_1 = nn.Sequential(
            UNetDown(32, 32) # 출력: [32 X 256 X 256]
        )
        
        self.down1 = UNetDown(32, 32) # 출력: [64 X 256 X 256]
        self.down2 = nn.Sequential(
            UNetDown(32, 32) # 출력: [32 X 256 X 256]
        )
        
        self.down3 = UNetDown(32, 64) # 출력: [64 X 128 X 128]
        self.down4 = nn.Sequential(
            UNetDown(64, 64) # 출력: [128 X 128 X 128]
        )
        self.down5 = UNetDown(64, 128) # 출력: [128 X 64 X 64]
        self.down6 = nn.Sequential(
            UNetDown(128, 128) # 출력: [256 X 64 X 64]
        )
        self.down7 = UNetDown(128, 256) # 출력: [256 X 32 X 32]
        self.down8 = nn.Sequential(
            UNetDown(256, 256) # 출력: [256 X 32 X 32]
        )
    def forward(self, x):
        d0_0 = self.down0_0(x)
        d0_1 = self.down0_1(d0_0) #connect
        d0_2 = self.maxpool(d0_1)
        
        d1 = self.down1(d0_2)
        d2 = self.down2(d1) #connect
        d3 = self.maxpool(d2)

        d4 = self.down3(d3)
        d5 = self.down4(d4) #connect
        d6 = self.maxpool(d5)

        d7 = self.down5(d6)
        d8 = self.down6(d7) #connect
        d9 = self.maxpool(d8)

        d10 = self.down7(d9)
        d11 = self.down8(d10) #connect

        return d0_1, d2, d5, d8, d11

# U-Net 생성자(Generator) 아키텍처
class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()
        
        self.conditioner = Conditioner()
        
        self.maxpool = nn.MaxPool2d(2, stride=2)

        self.down0_0 = UNetDown(in_channels, 32, normalize = False) 
        self.down0_1 = UNetDown(32, 32) # 출력: [32 x 512 x 512]
        
        self.down1 = UNetDown(32, 32) 
        self.down2 = UNetDown(32, 32) # 출력: [32 X 256 X 256]
        
        
        self.down3 = UNetDown(32, 64) # 출력: [64 X 128 X 128]
        self.down4 = UNetDown(64, 64) # 출력: [64 X 128 X 128]
        
        self.down5 = UNetDown(64, 128) # 출력: [128 X 64 X 64]
        self.down6 = UNetDown(128, 128) # 출력: [128 X 64 X 64]
        
        self.down7 = UNetDown(128, 256) # 출력: [256 X 32 X 32]
        self.down8 = UNetDown(256, 256) # 출력: [256 X 32 X 32]

        self.middle1 = UNetDown(256,512) # 출력: [512 x 16 x 16]
        self.middle2 = UNetDown(512,512) # 출력: [512 x 16 x 16]
        
        # Skip Connection 사용(출력 채널의 크기 X 2 == 다음 입력 채널의 크기)
        self.up1 = UNetUp(512, 256) # 출력: [256*3 X 32 X 32] #upsampling -> relu -> concat
        self.up2 = UNetDown(256*3,256*2) # 출력: [256*2 x 32 x 32] 
        self.up3 = UNetDown(256*2,256) # 출력: [256 x 32 x 32]
        
        self.up4 = UNetUp(256, 128) # 출력: [128*3 X 64 X 64] 
        self.up5 = UNetDown(128*3, 128*2) # 출력: [128*2 x 64 x 64]
        self.up6 = UNetDown(128*2,128) # 출력: [128 x 64 x 64]

        self.up7 = UNetUp(128, 64) # 출력: [64*3 x 128 x 128]
        self.up8 = UNetDown(64*3, 64*2) # 출력: [64*2 x 128 x 128]
        self.up9 = UNetDown(64*2, 64) # 출력: [64 x 128 x 128]

        self.up10 = UNetUp(64,32) # 출력: [32*3 x 256 x 256]
        self.up11 = UNetDown(32*3,32*2) # 출력: [32*2 x 256 x 256]
        self.up12 = UNetDown(32*2,32) # 출력: [32 x 256 x 256]

        self.up13 = UNetUp(32,32)
        self.up14 = UNetDown(32*3, 32*2)
        self.up15 = UNetDown(32*2, 32) # 출력: [16 x 512 x 512]
        self.final = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size = 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x, style_x):
        # 인코더부터 디코더까지 순전파하는 U-Net 생성자(Generator)
        c0,c1,c2,c3,c4 = self.conditioner(style_x)

        d0_0 = self.down0_0(x)
        d0_1 = self.down0_1(d0_0) # connect
        d0_2 = self.maxpool(d0_1)
        
        d1 = self.down1(d0_2)
        d2 = self.down2(d1) #connect
        d3 = self.maxpool(d2)

        d4 = self.down3(d3)
        d5 = self.down4(d4) #connect
        d6 = self.maxpool(d5)

        d7 = self.down5(d6)
        d8 = self.down6(d7) #connect
        d9 = self.maxpool(d8)

        d10 = self.down7(d9)
        d11 = self.down8(d10) #connect
        d12 = self.maxpool(d11)

        d13 = self.middle1(d12)
        d14 = self.middle2(d13)
    
        d15 = self.up1(d14,d11,c4)
        d16 = self.up2(d15)
        d17 = self.up3(d16)

        d18 = self.up4(d17,d8,c3)
        d19 = self.up5(d18)
        d20 = self.up6(d19)

        d21 = self.up7(d20, d5,c2)
        d22 = self.up8(d21)
        d23 = self.up9(d22)

        d24 = self.up10(d23, d2,c1)
        d25 = self.up11(d24)
        d26 = self.up12(d25)

        d27 = self.up13(d26, d0_1,c0)
        d28 = self.up14(d27)
        d29 = self.up15(d28)
        
        return self.final(d29)


# U-Net 판별자(Discriminator) 아키텍처
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_channels, out_channels, normalization=True):
            # 너비와 높이가 2배씩 감소
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # 두 개의 이미지(실제/변환된 이미지, 조건 이미지)를 입력 받으므로 입력 채널의 크기는 2배
            *discriminator_block(in_channels * 2, 64, normalization=False), # 출력: [64 X 256 X 256]
            *discriminator_block(64, 64), # 출력: [128 X 128 X 128]
            *discriminator_block(64, 128), # 출력: [128 X 64 X 64]
            *discriminator_block(128, 256), # 출력: [256 X 32 X 32]
            *discriminator_block(256, 512), # 출력: [512 X 16 X 16]
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, kernel_size=4, padding=1, bias=False) # 출력: [1 X 16 X 16]
        )

    # img_A: 실제/변환된 이미지, img_B: 조건(condition)
    def forward(self, img_A, img_B):
        # 이미지 두 개를 채널 레벨에서 연결하여(concatenate) 입력 데이터 생성
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)