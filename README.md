# Style Similarity Model 구축
본 프로젝트에서 하고자 하는 목표는 style 간의 유사도를 측정하는 것이다. 즉, style feature의 새로운 평가 지표를 제안.

이미지를 input으로 받은 model이 style information이 담긴 style vector/feature map을 뽑아내는 것을 기대.

해당 style vector/feature map으로 classification & distance loss 를 통해 style 간의 유사도를 측정하고자 함.
![image1](./imgs/image1.png)
- Dataset 
Kaggle의 WikiArt에서 제공하는 약 100,000장의 이미지를 사용 for 33 styles.

+) Mid-journey dataset : initial approach에서만 사용.
<a href="https://www.kaggle.com/datasets/succinctlyai/midjourney-texttoimage">mid-journey datsaet</a>

## First experiment 
cosine similarity를 loss fucntion으로 같은 style이면 유사도가 높게, 다른 style이면 유사도가 낮게 학습
![image](./imgs/image.png)

### problem
이 경우, positive pair의 부족(Server 상의 GPU 한계) & cosine similarity로 미는 척도(유사한 style이면 조금 밀고, 관련 없는 style이면 많이 밀어내는)를 정할 수 없는 문제로 인해서 실패

## Second experiment
그럼, classification은 잘 되나 실험해 봤는데 test accuary가 10% 였다. 즉, 아무런 근거 없이 classificaiton을 하는 것은 style information을 뽑아내지 못한다. 
![image2](./imgs/image2.png)

그럼에도, 최대한 성능을 끌어내고자 다양한 실험을 거쳤고 최종적으로 efficientNet을 사용하여 test accuary 50%를 달성했다.
![image3](./imgs/image3.png)
![image4](./imgs/image4.png)

## Third experiment
classification만으론 model이 온전한 style info를 뽑아낸다는 근거가 부족하다.

style information을 온전히 뽑아내기 위해, 구도가 같지만 style이 다른 이미지를 이용하여 style info를 뽑아내고자 했다. (image + style = style image)
![image5](./imgs/image5.png)

<strong>reference paper</strong>
- Neural Style Transfer
- Pix2Pix
- Control-Net
- InstructPix2Pix
- GAN
- UNet
- Conditioning GAN
- Diffusion
- DreamBooth
- CLIP
- WikiArt
- StyleGAN
- Prompt-to-prompt
- AdaIN
- Etc.

<strong>Dataset generation</strong>
![image6](./imgs/image6.png)
![image7](./imgs/image7.png)

(1) Pix2Pix paper implementation
Pix2Pix에서 제안하는 method를 그대로 사용했다. 색깔의 변화만 존재
![image8](./imgs/image8.png)
![image9](./imgs/image9.png)
![image10](./imgs/image10.png)

(2) +UNet
Generator architecture를 UNet으로 바꿨다. 색깔의 변화만 존재
![image11](./imgs/image11.png)
![image12](./imgs/image12.png)

(3) +reference image 
UNet decoder 단에 style reference image를 VGG network를 통해 전달함으로써, style info를 뽑는 동시에 정보 전달을 하려고 노력함. 

    1) 정답 이미지(style image)를 reference image로 주었을 때, 완벽히 따라하는 걸 보고 가능성이 있다고 판단
    2) 같은 style의 다른 구도 이미지를 제공. 즉, reference image 1장. generator가 학습을 포기
    3) 같은 style의 다른 구도 이미지를 제공. 이때, reference image는 3장. 색깔 변화 + object 생성의 시작.
    4) 같은 style의 다른 구도 이미지를 제공. 이때, reference image는 5장. 확연한 변화가 눈에 보인다.

1)
![image13](./imgs/image13.png)
![image14](./imgs/image14.png)

2)
![image15](./imgs/image15.png)
![image16](./imgs/image16.png)

3)
![image17](./imgs/image17.png)

4)
![image18](./imgs/image18.png)

### Problem
많은 노력을 했음에도 불구하고, 여전히 생성된 이미지에 noise가 심하다.

Style의 범주가 워낙 크다보니, 작은 모델 & 적은 데이터 셋으로 학습하기가 매우 어렵다

### Solution
Style의 범주를 사람에 대한 style로 좁혀서 정의한다. 사람과 관련된 prompt로 다시 데이터셋 구성(대략 2000장)
![image19](./imgs/image19.png)

## Fourth experiment
reference image 5장 (같은 style의 다른 구도 이미지)를 제공했다. Style의 범주를 좁혀서 학습하니, 성능이 나오는 걸 확인.
![image21](./imgs/image21.png)

### Problem
    (1) 이미지가 여전히 blurry하다. 
    (2) Discriminator가 classification할 때의 근거가 부족하다.
    (3) 이미지가 정확히 같은 구도는 아니다.
        - 없던 object가 생기거나
        - 모양이 바뀌거나 etc.

#### Upon in the future
만일, 학습이 잘 되었다면 이후에 할 step은 reference image를 generator decoder에 줄 때 사용한 feature map/vector로 classification 및 distance loss를 주어 학습을 시키는 것이다. 

그렇게 하면, style information으로 classification을 하는 동시에 같은 class면 두 vector가 같은 분포를 가지도록 학습이 되기 때문에, 추후에 분포상 거리에 따라 style similarity를 측정할 수 있게 된다.
![image20](./imgs/image20.png)

#### 추가적인 개선사항
- StyleGAN2에서 style을 주입하는 방식
- Model 구조
- Dataset 
- GAN에서 사용하는 regularization loss
- 512x512 image로 upscaling (o)

#### Problem
이 새로운 평가지표가, 기존에 존재하는 FID & KID 등 과의 차별점이 존재하는가?

적은 이미지로도 측정 가능하지만, 엄청나게 많은 style을 학습시켜야 하기 때문에 cost가 비싸다.


### Additional result
512x512 image로 재실험을 했을 때에도 여전히 blurry하다.

# Folder/Files
- 10000.png
기존의 Pix2Pix result

- initial approach
ResNet & ViT & EfficientNet & VGG 등으로 classification 을 진행한 code.

WikiArt & Mid-journey image dataset를 download하는 코드.

- anime_result(pix2pix)
기존의 Pix2Pix model을 그대로 사용

- anime_unet_result(+UNet)
Generator model 구조를 UNet으로 교체

- anime_unet_with_condition_result(+style image)
Reference image에 정답을 제공

- anime_unet_with_condition2(1ref)_result
구도가 다른 같은 style image 한 장

- anime_unet_with_condition2(3ref)_result
Reference image 3장 사용

- anime_unet_with_condition2(5ref)_result
Reference image 5장 사용

- anime_unet_with_condition2(5ref_l1_1)_result
Reference image 5장 사용 + L1 loss의 strength를 1로

- anime_person_unet_with_condition2(5ref_l1_1)_result
Reference image 5장 사용 + L1 loss의 strength를 1 + only person in dataset 

- anime_person_unet_with_condition2(5ref_l1_1)_result
Reference image 5장 사용 + L1 loss의 strength를 10 + only person in dataset 

- classification
Reference image의 style info를 추출하여 각 feature map을 plot 해봤다. style infor가 확연히 보이지는 않다.

- anime_person_512x512
512 x 512 upscaling -> additional result

- .py
512 x 512를 입력으로 사용을 하면, 단일 GPU로는 학습이 되지 않기 때문에 multiGPU를 활용하여 학습하는 코드.