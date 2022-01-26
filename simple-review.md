## List
### I. Matching problem
1. Convolutional neural network architecture for geometric matching
2. Spatial Transformer Networks
3. Neighbourhood Consensus Networks
4. DGC-Net: Dense Geometric Correspondence Network
5. GLU-Net: Global-Local Universal Network for Dense Flow and Correspondences
### II. representation learning
1. A Simple Framework for Contrastive Learning of Visual Representations
2. Momentum Contrast for Unsupervised Visual Representation Learning
3. Bootstrap your own latent: A new approach to self-supervised Learning
4. Exploring Simple Siamese Representation Learning
### III. image transformer
1. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
2. COTR: Correspondence Transformer for Matching Across Images
3. LoFTR: Detector-Free Local Feature Matching with Transformers

## I. Matching problem
### 1. Convolutional neural network architecture for geometric matching
#### Introduction
기존의 detecting, matching local features를 이용한 geometry model (such as epipolar geometry or planar affine transformation) 통해 estimating correspondence를 수행한 SIFT나 HOG같은 방법들은 대상의 appearance나 scene에 large changes가 있는 many parameters를 가진 complex geometric model이 필요한 tasks에서는 좋은 성능을 내지 못했다. 
#### Contribution
geometric matching을 위한 CNN architecture를 포함한 model

never seen before images에 대해서도 잘 동작할 수 있고 synthetically generated image를 이용한 self-supervised learning이 가능한 model

challenging dataset에 대해 instance level과 category level matching에서 둘 다 좋은 성능을 내는 model

#### Key ideas
CNN for feature extraction, small to large geometric transformation, 3D correlation map, self supervised learning, siamese model

#### Details
![image](https://user-images.githubusercontent.com/67745456/150939326-e3d009cd-3425-468d-81c5-6507beed0dc9.png)
geometric matching을 수행할 두 image를 각각 siamese CNN으로 구성된 feature extractor에 input으로 넣고 두 개의 feature map을 얻는다. 

두 개의 feature map을 correlation layer에서 받아 3D correlation map(w × h × wh)을 만든다.

correlation map을 affine regression module에 통과시켜 affine transformation parameters(6)를 얻는다.

얻은 affine transformation parameters로 source image를 warp하고 warped source image와 target image로 위의 과정을 반복하고 affine transformation parameters 대신 TPS transformation parameters(18)을 얻어서 두 image에 geometric matching을 수행한다.

![image](https://user-images.githubusercontent.com/67745456/150939370-b12948e7-c8d2-4e2b-a9a8-fbbdf23efa98.png)

한 개의 image에 randomly sampled transformation을 수행하여 pixel-wise matching의 ground truth을 가진 data들을 얻어 학습한다.

### 2. Spatial Transformer Networks
#### Introduction
computer vision분야에 CNN의 도입으로 많은 발전이 있었지만 CNN에서 spatially invariant한 extracted feature를 얻기 위해 사용하는 max-pooling layer는 small receptive field를 가지기 때문에 spatially invariance를 위해 깊은 구조를 만들 필요가 있고 input data의 large transformation에 대해 invariant하지 않다.
#### Contribution
CNN을 사용하지 않고 얕은 구조로 spatial transformation capabilities를 제공하는 module,

standard neural network architecture에 end-to-end 방식으로 들어갈 수 있는 differentiable module

#### Key ideas
transformation with parameters, sampler, generalization, modularization, differentiable end-to-end

#### Details
![image](https://user-images.githubusercontent.com/67745456/150918613-7fbd9bd0-020a-4777-ae0a-3495e75e21d3.png)

Localisation network가 input image의 feature map을 입력받아 parameters of the transformation을 출력한다. 이때, network는 fully connected net이나 convolutional net같은 임의의 form으로 구성할 수 있고 output size는 원하는 transformation 종류에 따라 조정할 수 있다.

Grid generator가 Localisation net의 output을 input으로 받고 transformation 연산을 통해 normalised coordinates로부터 sampling할 coordinates를 출력한다.

Sampler가 input으로 input feature map과 Grid generatior의 output인 sampling coordinates를 받아 input feature map의 sampiling coordinates를 전체 feature map으로 펴서 최종적으로 출력한다.

![image](https://user-images.githubusercontent.com/67745456/150920969-b4064284-560d-4903-bdbe-7d64b42f41f2.png)

#### Conclusion
spatially invariance를 이용해야하는 tasks (ex.찌그러진 이미지 구분 등)에서 좋은 성능을 보였고 기존 모델에 end-to-end 방식으로 적용할 수 있어 training speed도 별로 줄지 않았다.

### 3. Neighbourhood Consensus Networks

#### Introduction
matching problem에서 local patch descriptors을 이용해 individual image features를 matching하는 방법은 textureless region이나 repetitive pattern이 많은 이미지 처리에 대해 자신의 patch와 근처의 patch의 차이를 구분하지 못하는 근본적인 한계가 있다.
#### Contribution
4D CNN으로 구성된 dense matching과 local geometric constraints learning을 위한 neighbourhood consensus network

image pair의 negative, positive 구분 정도의 weakly supervised loss로 처음부터 학습가능한 network

category level과 instance level matching 둘 다에서 넓은 범위의 matching tasks에서 활용가능한 model

#### Key ideas
4D correlation map, 4D convolution layer, soft mutual nearest neighbour filter

#### details
![image](https://user-images.githubusercontent.com/67745456/150980399-7c28bbcd-0a07-44d1-a2b0-789a25e327fe.png)

두 이미지의 feature map을 4D correlation layer에서 input으로 받아 4D correlation map을 만든다.

4D correlation map은 soft mutual nearest neighbour filter를 거쳐 상호간의 matching되지 않는 점들의 value를 떨어트리고 neighbourhood consensus network를 거쳐 high confidence score를 가진 match로부터 주변 match들이 information을 얻을 수 있게 하고 다시 soft mutual nearest neighbour filter를 거쳐 최종적인 4D dense correlation map을 얻는다. 

![image](https://user-images.githubusercontent.com/67745456/150986815-bb47c94c-94c1-4914-9f08-d9b83a7893b4.png)

이때, neighbourhood consensus network의 4D convolution layer는 서로 근처의 matching 정보를 확인하는 역활을 하고 2D convolution layer처럼 점점 깊어지며 점점 넓은 범위의 정보를 이용할 수 있게 된다.

training은 positive image pair에서는 mean matching score가 높은 값을 가지도록 negative image pair에서는 mean matching score가 낮은 값을 가지도록 update를 진행한다.


### 4. DGC-Net: Dense Geometric Correspondence Network
#### Introduction
matching problem에서 pixel-wise correspondence field를 얻는 방법은 크게

(i) image pair에 대해 feature descriptors를 적용하여 얻은 match keypoints의 주변에서 match를 시켜 얻은 sparse map을 통해 pixel-wise correspondences를 얻는 방법,
(ii) image patches를 비교하여 feature space로부터 바로 dense correspondences를 얻는 방법

두 가지로 나눌 수 있다. 그러나 (i) 방법은 sparse map에 의존하기 때문에 충분한 key point를 찾지 못할 수 있고 (ii) 방법은 멀리 떨어진 point간의 correspondence, large transformation을 처리하지 못한다는 한계점이 있다.

#### Contribution
strong geometric transformations를 이용하여 pixel-wise dense correspondence map을 만들 수 있는 end-to-end CNN-based method

synthetic transformations를 이용한 unsupervised learning만으로 real data에 대해 잘 작동할 수 있는 network model

original DGC-Net을 수정하여 low confidence score를 가진 tentative correspondence를 제거하고 계산 효율을 높였다.

#### Key ideas
pyramid feature map, iterative architecture, coarse-to-fine method

#### Details
![image](https://user-images.githubusercontent.com/67745456/150994219-c4a00faf-739e-4bc1-a51e-30958e56ab5b.png)

image pair가 siamese CNN feature extractor에 input으로 들어가고 extractor의 각 단계에서 fine~coarse하게 구성된 pyramid feature map을 얻는다. 

coarsest feature map부터 차례대로 correspondence map을 만들어 다음 level의 source feature map을 warp시키고 다음 correpondence module에 warped source feature map, target feature map, 전 level correspondence map을 dimension 방향으로 concatenation 하여 전달한다.

위의 연산을 finest feature map까지 반복하고 마지막 finest 영역의 dense correspondence map과 matchability map을 얻는다.

투박한 영역에서부터 조금씩 source를 target으로 matching 시켜 matching 정확도를 높인다. 

training에서는 ground truth dense correspondence map과의 차이를 줄이고 matchability 영역의 차이를 줄이는 방향으로 update된다.


### 5. GLU-Net: Global-Local Universal Network for Dense Flow and Correspondences

#### Introduction
estimating correspondence problem은 geometric matching, optical flow, semantic matching같은 다양한 tastks에서 사용될 수 있지만 대부분의 current methods는 일반적인 correspondence problem과 크게 관련되지 않은 특별한 구조를 사용해 한 가지씩만을 다룬다.

optical flow같은 small displacements를 예측하는 architecture는 large viewpoints changes를 예측하지 못하고 반대로 geometric, semantic matching을 수행하는 architecture는 longrange matches는 처리할 수 있지만 high resolution에서 correlation layer를 적용시킬 수 없다.
#### Contribution
geometric matching, semantic matching, optical flow에서 모두 활용될 수 있는 single unified architecture

large, small displacements를 모두 다룰 수 있도록 global, local correlation layer를 적절히 조합한 network

global correlation layer를 사용함으로써 생기는 fixed input resolution 문제를 피하기 위한 adaptive resolution strategy

synthetic warps of real images에 의한 self-supervised learning


#### Key ideas
pyramid feature map, coarse-to-fine method, using both global and local correlation

#### Details
![image](https://user-images.githubusercontent.com/67745456/151071387-3b312854-0f97-415b-921b-c650372c2ec9.png)


## II. representation learning
### 1. A Simple Framework for Contrastive Learning of Visual Representations
#### Introduction
#### Contribution
#### Key ideas
#### Details
![image](https://user-images.githubusercontent.com/67745456/151148509-a8a3e80c-3004-4ae0-aeac-17cdb73cd6f0.png)


### 2. Momentum Contrast for Unsupervised Visual Representation Learning
#### Introduction
#### Contribution
#### Key ideas
#### Details
![image](https://user-images.githubusercontent.com/67745456/151148885-69082d41-4ddc-402e-bcd0-8c0722339572.png)
![image](https://user-images.githubusercontent.com/67745456/151148894-73173df2-a078-4ba5-b0ab-3b010f8d8b64.png)



### 3. Bootstrap your own latent: A new approach to self-supervised Learning
#### Introduction
#### Contribution
#### Key ideas
#### Details
![image](https://user-images.githubusercontent.com/67745456/151149715-83e224d6-6d45-4f6f-9232-0ed6de2d8519.png)


### 4. Exploring Simple Siamese Representation Learning
#### Introduction
#### Contribution
#### Key ideas
#### Details
![image](https://user-images.githubusercontent.com/67745456/151149553-ac1a666f-b05b-44d0-bf0e-e79c8807ccd8.png)
![image](https://user-images.githubusercontent.com/67745456/151149581-a9def428-2e67-4c64-b370-a6e0790c314e.png)



## III. image transformer
### 1. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
#### Introduction
#### Contribution
#### Key ideas
#### Details
![image](https://user-images.githubusercontent.com/67745456/151150710-22beb005-4b8b-4788-88be-cb9e0655e06b.png)


### 2. COTR: Correspondence Transformer for Matching Across Images
#### Introduction
#### Contribution
#### Key ideas
#### Details
![image](https://user-images.githubusercontent.com/67745456/151150754-d43de8bd-a108-4354-85f0-957a29a5e05b.png)
![image](https://user-images.githubusercontent.com/67745456/151150767-af3ed24d-a370-4ca9-8fcf-b3fc77094f84.png)


### 3. LoFTR: Detector-Free Local Feature Matching with Transformers
#### Introduction
#### Contribution
#### Key ideas
#### Details
![image](https://user-images.githubusercontent.com/67745456/151150791-701e39b6-ccfc-4e63-acdb-3a4612531cd5.png)
![image](https://user-images.githubusercontent.com/67745456/151150800-825a2f37-d7ce-4388-81f5-ee238906c576.png)



















