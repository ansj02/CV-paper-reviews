## List
### I. Matching problem
1. Convolutional neural network architecture for geometric matching
2. Spatial Transformer Networks
3. Neighbourhood Consensus Networks
4. DGC-Net: Dense Geometric Correspondence Network
5. GLU-Net: Global-Local Universal Network for Dense Flow and Correspondences - writing
6. Learning Correspondence from the Cycle-consistency of Time - writing
### II. representation learning
1. A Simple Framework for Contrastive Learning of Visual Representations
2. Momentum Contrast for Unsupervised Visual Representation Learning
3. Bootstrap your own latent: A new approach to self-supervised Learning
4. Exploring Simple Siamese Representation Learning
### III. image transformer
1. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
2. COTR: Correspondence Transformer for Matching Across Images
3. LoFTR: Detector-Free Local Feature Matching with Transformers
### IV. 3D reconstruction
1. Occupancy Networks: Learning 3D Reconstruction in Function Space - writing
2. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis - writing

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

coarest feature map에서 correlation을 수행하여 dense correlation map을 얻고 coarsest feature map부터 차례대로 correspondence map을 만들어 다음 level의 source feature map을 warp시키고 다음 correpondence module에 warped source feature map, target feature map, 전 level correspondence map을 dimension 방향으로 concatenation 하여 전달한다.

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

### 6. Learning Correspondence from the Cycle-consistency of Time
#### Introduction
visual correspondence를 위한 representation learning에는 주로 large amounts of labelled data를 이용한 supervised learning을 이용했고 이는 prohibitively expensive했다.
#### Contribution
self-supervised method for learning visual correspondence from unlabeled video
self-supervised method using cycle consistent tracking

#### Key ideas
cycle consistency, obtain unlimited supervision by tracking backward and then forward in video
#### Details
![image](https://user-images.githubusercontent.com/67745456/151367287-0f478c2e-3861-47bb-86b1-6349f9d85978.png)



## II. representation learning
### 1. A Simple Framework for Contrastive Learning of Visual Representations
#### Introduction
Representation learning에 주로 사용되는 방법은 크게 generative or discriminative로 나눌 수 있다. Generative approach는 불필요하게 복잡한 연산을 포함할 수 있고 discriminative approach는 pretext tasks에 의한 학습이 learned representations의 generality를 제한할 수 있다는 문제가 있다. 최근 contrastive learning을 base로 한 discriminative approaches가 좋은 성능을 내고 있다.
#### Contribution
a simple framework for contrastive learning of visual representations, which outperforms previous work
#### Key ideas
large batch size, long training, suitable augmentation operation, constrative learning, nonlinear projection
#### Details
![image](https://user-images.githubusercontent.com/67745456/151148509-a8a3e80c-3004-4ae0-aeac-17cdb73cd6f0.png)
image pair를 siamese feature extractor를 통과시켜 얻은 representations를 마찬가지로 weights를 공유하는 projection에 통과시켜 얻은 최종 outputs의 dot product 방식으로 simirality를 계산한다.

따로 negative samples를 만들지 않고 large size의 batch에 여러 image에 서로 다른 두 augmentation을 적용하여 만든 2 sets의 views를 만들고 loss의 분자에 같은 image에서 만들어진 views의 similarity, 분모에 다른 image에서 만들어진 views의 similarity를 넣고 -1을 곱해 같은 image에서 만들어진 representaion만 가까워지게 하고 나머진 다 멀어지게 하는 방향으로 학습한다.

augmentation으로는 random cropping과 random color distortion을 같이 사용한 것이 global to local prediction이나 neighboring view prediction의 효과를 얻어 가장 좋은 성능을 보였다고 한다. 


### 2. Momentum Contrast for Unsupervised Visual Representation Learning
#### Introduction
unsupervised representation learning이 nlp분야에서는 상당히 성공적으로 사용되고 있지만 computer vision 분야는 여전히 supervised pre-training 방식이 우세하다. 최근 contrastive learning을 사용한 방식으로 진전이 있었지만 negative sample을 제공하는 방식에서 large and consistent한 dictionary를 제공하지 못했다.
대표적으로 SimCLR은 batch size에 dictionary size가 종속되었고 memory size에 제약을 받았고 memory bank방식은 training 중에 dictionary 내부의 sample의 representation을 추출하는 model이 바뀌며 일관된 training이 힘들었다.
#### Contribution
a way of building large and consistent dictionaries for unsupervised learning with a contrastive loss
#### Key ideas
dictionary as a queue of data samples, momentum encoder
#### Details
![image](https://user-images.githubusercontent.com/67745456/151148885-69082d41-4ddc-402e-bcd0-8c0722339572.png)

일반적인 contrastive learning의 방식을 따르지만 key views를 처리하는 encoder가 loss에 의해 학습되지 않고 query encoder의 momentum에 의해 학습된다는 점과 key encoder의 결과가 queue에 저장되어 일정한 size의 negative samples로 학습에 활용된다는 점이 일반적인 방식과의 차이점이다.

![image](https://user-images.githubusercontent.com/67745456/151148894-73173df2-a078-4ba5-b0ab-3b010f8d8b64.png)



### 3. Bootstrap your own latent: A new approach to self-supervised Learning
#### Introduction
representation learning 분야에서 contrastive methods가 좋은 성능을 보여주었지만 carful treatment of negative pairs가 필요했기 때문에 이를 위해 large batch size나 memory bank나 customized mining strategy같은 별도의 처리법이 필요했고 performance도 image augmentation 방법에 크게 의존했다.
#### Contribution
negative pair를 사용하지 않아도 collapsed representation(모든 input에 대해 같은 상수를 출력하는 등의 잘못된 representation)을 학습하지 않는 method

contrastive methods에 비해 batch size나 augmentation 선택에 자유롭고 더 좋은 성능을 내는 method

#### Key ideas
training on positive pairs only, momentum learning(moving exponential average), asymmetric architecture with predictor
#### Details
![image](https://user-images.githubusercontent.com/67745456/151149715-83e224d6-6d45-4f6f-9232-0ed6de2d8519.png)

feature extractor(CNN), projector(MLP), predictor(MLP)로 이루어진 online network와 feature extractor(CNN), projector(MLP)로만 이루어져 있고 출력단에 stop gradient가 적용된 target network에 각각 하나의 image에 다른 augmentation을 적용하여 만든 두 view를 하나 씩 집어넣는다.

두 network의 output의 distance를 loss로 우리의 목적인 downstream task에서 활용가능한 feature extractor를 가진 online network의 학습을 진행한다. 이때, target network는 loss에 의한 학습이 아닌 online network의 feature extractor와 projector의 weights의 moving exponential average로 갱신되어 학습된다.

여기서 저자들은 두 network가 같은 방향으로 학습되지 않기 때문에 collapsed representation을 학습하지 않고 asymmetric한 network 구조가 collapsed representation을 막는데에 중요한 역활을 한다고 말한다.

### 4. Exploring Simple Siamese Representation Learning
#### Introduction

representation learning에는 다양한 방법에 의한 꾸준한 발전이 있었고 대부분의 경우 siamese구조의 network가 활용되었다. siamese구조의 network는 all outputs collapsing to a constant에 대한 위험이 있고 이를 해결하기 위해 SimCLR, SwAV, BYOL 등의 방법에서 다양한 architecture를 이용한 해결법이 제시되었고 각각 contrastive learning의 negative sample에 의한 제약, momentum learning에 의한 학습 속도 저하 등의 문제가 있었고 위의 방법론들에 아이디어를 얻어 그들의 필요한 부분만을 남긴 simple한 model을 설계하였다.

#### Contribution
a simple siamese network model for representation learning, which uses no negative pairs and momentum learning
#### Key ideas
training on positive pairs only, stop gradient, weights share,asymmetric architecture with predictor
#### Details
![image](https://user-images.githubusercontent.com/67745456/151149553-ac1a666f-b05b-44d0-bf0e-e79c8807ccd8.png)

BYOL구조에서 momentum learning을 쓰지 않고 바로 weights share한 구조다.

저자들은 BYOL에서 momentum learning은 collapsing을 막는데 직접적인 역활을 하지 않는다고 말했고 asymmetric구조가 중요하다고 말했다.

![image](https://user-images.githubusercontent.com/67745456/151149581-a9def428-2e67-4c64-b370-a6e0790c314e.png)


## III. image transformer
### 1. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
#### Introduction
self attention based architecture인 transformer가 nlp분야에서 큰 성공을 거두었지만 여전히 computer vision분야에서는 convolutional architecture가 지배적으로 사용되고 있다. transformer로 covolutional architecture을 대체하거나 조합하기 위한 다양한 시도가 있었지만 아직까진 고전적인 ResNet base의 방법이 state of the art를 차지하고 있다.

transformer based model은 CNN 특유의 inductive biases가 부족하기 때문에 적은 양의 data에서 학습이 잘 이루어지지 않았지만 많은 양의 data에서 오히려 고전적인 방법을 뛰어넘는 성능을 보이는 것을 확인했다.
#### Contribution
CNN에 의존하지 않는 transformer module만을 이용하여 classification tasks를 높은 성능으로 수행하는 model
#### Key ideas
transformer, sequences of image patches, 2D patch to 1D vector
#### Details
![image](https://user-images.githubusercontent.com/67745456/151150710-22beb005-4b8b-4788-88be-cb9e0655e06b.png)

하나의 image를 word 역활을 할 여러 개의 정해진 크기 PxP의 patchs로 나누고 각 patch를 linear projection을 통해 1D vector로 만들고 position embedding을 더한 결과들의 모임 sequence를 transformer module을 통해 self attention 시킨다. transformer의 output에 MLP를 통해 classification을 수행한다.

### 2. COTR: Correspondence Transformer for Matching Across Images
#### Introduction
finding correspondence problem을 수행하기 위한 방법은 크게 (i) image pair의 spase keypoints를 찾아 match를 수행하는 방법과 (ii) image pair의 전체적인 dense correspondence를 얻는 방법으로 나눌 수 있다.
(i)의 방법은 large displacements 상황의 멀리 떨어진 points의 correspondence를 얻을 수 있지만 local feature에 의존하기 때문에 texture-less area에서 잘 작동하지 않을 수 있고 필요한 point의 match를 얻을 수 없을 수도 있다. (ii)의 방법은 context를 활용하여 texture-less area같은 구분이 힘든 region의 correspondece도 얻을 수 있고 arbitrary location의 correspondence를 얻을 수 있지만 low feature resolution에서만 사용할 수 있어 정보 손실이 있고 주로 small dispacements 상황에서만 사용된다.


#### Contribution
data로부터 global, local information을 활용하여 dense, sparse correspondence를 둘 다 얻을 수 있는 model

원하는 만큼 임의의 query points에 대해 corresponding points를 찾아낼 수 있고 large motion에 대해서도 correspondence를 찾아낼 수 있는 model

recursive method를 사용하여 accurate correspondences를 얻을 수 있는 model

#### Key ideas
functional method, transformer, recursive method
#### Details
![image](https://user-images.githubusercontent.com/67745456/151150754-d43de8bd-a108-4354-85f0-957a29a5e05b.png)

image pair로부터 얻은 feature map을 concatenation하고 positional encodings를 추가하고 transformer에 넣는다. 여기에 query point를 앞에서 사용한 positional encoder에 통과시켜 얻은 positional encoding을 decoder에 query로 넣어 원하는 query point에 대해 attention을 진행한다. decoder에 최종적으로 나온 결과를 fcn에 input으로 넣어 query point에 대한 corresponding point를 얻는다.

![image](https://user-images.githubusercontent.com/67745456/151150767-af3ed24d-a370-4ca9-8fcf-b3fc77094f84.png)

찾은 corresponding point와 query point에 대해 zoomed in crops를 반복하여 정확도를 높인다.

### 3. LoFTR: Detector-Free Local Feature Matching with Transformers
#### Introduction
local feature matching에서 주로 사용되는 feature detector based methods는 poor texture나 repetitive pattern이 많은 image에 대해 충분한 interest points를 감지하지 못한다. 최근에는 dense matches 중 high confidence match를 CNN으로 계산하여 선택하는 방법도 연구되었지만 이 역시 convolution layer의 근본적인 문제인 limited receptive field 때문에 indistinctive regions를 잘 구분하지 못한다.
#### Contribution
a detector-free approach to local feature matching
#### Key ideas
small pyramid feature map, transformer, coarse-to-fine method, select high confidence point in dense correlation map, division indistinctive regions by positional encoding, linear attention, soft mutual nearest neighbor matching
#### Details
![image](https://user-images.githubusercontent.com/67745456/151150791-701e39b6-ccfc-4e63-acdb-3a4612531cd5.png)

image pair로부터 coarse level feature map과 fine level feature map을 얻는데 특이하게 residual 방식으로 연결한 encoder-decoder 모양 network에서 fine feature map을 얻는다. 먼저 coarse level feature map을 1D vector로 피고 positional encoding을 더해서 transformer에 넣는다. 여기서 나온 결과를 differentiable matching layer에 넣어 matching map을 얻고 각 point에 대해 soft mutual nearest neighbor matching을 수행하여 high confidence matching map을 얻고 high confidence match에 대해 fine level feature map에서 해당하는 point 주변만 cropping하여 위와 같은 방식으로 attention된 feature map을 얻고 source의 중심 vector에 대해 target의 correlation을 수행하여 최종적으로 correspondence point를 얻는다.

![image](https://user-images.githubusercontent.com/67745456/151150800-825a2f37-d7ce-4388-81f5-ee238906c576.png)

일반적으로 사용되는 dot product attention 대신 linear attention이 사용된다.




## IV. 3D reconstruction

### 1. Occupancy Networks: Learning 3D Reconstruction in Function Space - writing
#### Introduction
기존의 3D reconstruction을 위한 3D representation을 위해 주로 사용된 method는 크게 voxel, point, mesh representation method로 나눌 수 있다.
이 중 voxel representation는 memory footprint가 resolution 증가에 따라 cubically 증가하기 때문에 상대적으로 적은 수의 voxels로 representation이 제한된다. point clouds나 meshs를 활용하는 method들이 대안으로 제시되었지만 point clouds는 underlying mesh의 connectivity structure가 없으므로 model에서 3D geometry 추출을 위한 추가 전처리 작업이 필요하고 mesh representation은 대상 domain의 template mesh가 필요하여 arbitrary topologies를 할 수 없고 매우 특정한 영역으로 제한되어 나타내어진다. 또한 두 방법 모두 standard feed-forward network를 사용하여 안정적으로 나타낼 수 있는 points, vertices 수가 제한된다.

#### Contribution
representation for 3D geometry based on learning a continuous 3D mapping

representation that can use a various input types

approach that can generate high-qualiry meshes

#### Key ideas
classifier, conditional encoder, low to high resolution
#### Details
![teaser](https://user-images.githubusercontent.com/67745456/151742237-63910d96-e9e6-42eb-9b32-ba027b6694a1.png)

3D geometry representation을 표현하려는 object의 내부, 외부를 구별하는 classifier를 사용하여 만든다.

![image](https://user-images.githubusercontent.com/67745456/151747624-2bb07f94-144f-4994-953b-ea1cd32f9b0e.png)

먼저, 사용되는 model은 conditional encoder depending on the type of input와 fcn ResNet block으로 이루어진 classifier model이다. model이 3D representation을 생성할 대상의 input data와 occupancy probability를 예측한 3D locations를 입력받아 각 location의 occupancy probability를 출력한다.

![image](https://user-images.githubusercontent.com/67745456/151751390-d22497e4-96fe-45a4-813b-31eb9eaa1119.png)

input data를 mini batch로 나누고 batch내의 각각의 input data에 대해 random sampling한 K개의 points에 대해 예측한 occupancy prob와 실제 occupancy label간의 loss를 사용하여 학습한다.

![image](https://user-images.githubusercontent.com/67745456/151789311-aa13bc17-bdb0-4428-85a8-8dba126c1e11.png)

generative model을 학습시키는데도 사용할 수 있다. probabilistic latent variable z의 분포를 예측하는 encoder의 결과와 prior distribution을 KL-divergence로 가까워지도록 encoder를 학습한다.

![image](https://user-images.githubusercontent.com/67745456/151794506-a0e9f82f-6e86-48a2-9cd3-2b70a4d74d05.png)

3D reconstruction을 위해 Marching Cubes algorithm으로 approximate isosurface를 얻는다. 즉, 만들고자 하는 3D structure의 경계를 얻는다.

![image](https://user-images.githubusercontent.com/67745456/151793820-2e85d1da-fc78-40ed-a7ce-aa5890a7ba40.png)

initial resolution에서 모든 point에 대해 model에서 occupied or unoccupied 여부를 평가한다. 그리고 각 voxel에 대해 occupied, unoccupied corners를 모두 가진 voxel을 active로 mark하고 active voxels를 8개의 subvoxels로 나눈다. 정해진 resolution이 될 때까지 위의 과정을 반복하여 정해진 resolution grid의 point를 얻고 이를 이용하여 simplify mesh를 얻는다. 마지막으로 gradient를 사용하여 mesh를 보정하고 최종 3D structure를 얻는다.

![image](https://user-images.githubusercontent.com/67745456/151795975-a1b8bb2e-284c-4ed6-866a-1b182142f87a.png)

마지막 mesh 보정을 위해 mesh의 각 면에서 points를 random sampling하고 각 point의 occupancy 평가 값이 경계 값인 τ에 가까워지도록 occupancy 평가 값의 gradients와 mesh 방향 값이 가까워지도록 mesh를 조정한다.


### 2. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis - writing
#### Introduction
#### Contribution
#### Key ideas
#### Details

![image](https://user-images.githubusercontent.com/67745456/151969066-622cd0f9-ffd3-4907-bb28-1d9f4c510ea7.png)

만들려는 view로 부터 각 pixel로 ray를 통과시키고 가상의 3D space에서 ray에 포함된 locations가 가지는 color들을 자신과 앞 locations의 density들을 이용한 weight으로 전부 weighted sum하여 해당 pixel의 color값을 정한다. 이를 위해 특정 location에 대해 location값과 ray direction을 입력받아 color와 density를 출력하는 mlp model을 사용한다. 이때 model은 location과 ray direction에 대한 continuous function이므로 continuous 3D structure를 정의하는 역할을 하며 논문의 목적인 임의의 view에서 image의 각 pixel값을 예측하는데도 사용된다. 따라서 다양한 view의 image를 만들어내는 동시에 3D structure를 정의하는 model을 학습시킨다. 기존의 deep learning model들과 다르게 1개의 대상에 대해 1개의 model이 사용된다. 즉, 각각의 대상을 3D로 만들고 redering하기 위해 각각의 학습된 weigt을 가진 model이 필요하다.

![image](https://user-images.githubusercontent.com/67745456/151969143-628971c9-1838-4926-9dfa-ce53fd5febf9.png)

위에서 말했듯이 view에서 각 pixel의 color값을 구하기 위해 ray에 포함된 3D space의 location의 model로부터 나온 color값의 weighted sum을 이용한다. 여기서 weight은 해당 location보다 앞(view에 가까운) location의 density 적분 값에 반비례하고 자신의 density 값에 비례한다.

![image](https://user-images.githubusercontent.com/67745456/151969205-8da7dd38-6de9-4653-a766-45fd86c3b906.png)

![image](https://user-images.githubusercontent.com/67745456/151969254-a1b7dccf-0ff8-4eec-aad9-aeb515e80fc9.png)

실제로 계산할 때는 continuous 값을 처리할 수 없으므로 각 ray를 등분하여 각 단에서 uniform random sampling한 location을 사용한다.

![image](https://user-images.githubusercontent.com/67745456/151969336-f75454d1-8cdb-4d77-8fde-ce9cc1b6abf1.png)

논문에서는 model의 input으로 들어가는 3D location과 3D direction의 low dimention에 의해 high frequency output을 없기 때문에 periodic function을 이용한 positional encoding으로 input의 차원수를 늘려준다.

![image](https://user-images.githubusercontent.com/67745456/151985991-fd6ccd29-869b-425a-b15d-503684ab8b05.png)

location의 color와 density를 예측하는 model은 단순한 mlp model이며 각각 L=10으로 60D로 만든 location과 L=4로 24D로 만든 direction을 input으로 받고 direction은 color 예측에만 사용된다.

![image](https://user-images.githubusercontent.com/67745456/151969526-291d429a-ee64-4236-9631-b70dd535cb20.png)

![image](https://user-images.githubusercontent.com/67745456/151986846-c5f142b6-9eb4-445a-8ce8-598d9f2d8be0.png)

3D space상에서 실제 집중해야하는 부분을 더 비중있게 계산하기 위해 coarse network와 fine network를 사용하여 hierarchical하게 계산한다. 먼저 ray에서 coarse locations에 대해 계산을 수행하고 높은 weight 값을 가진 부분을 다시 fine location으로 나누어 계산한다. fine network와 coarse network 모두에서 ground truth에 대한 loss를 계산한다.















