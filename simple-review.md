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
3. Exploring Simple Siamese Representation Learning
4. Bootstrap your own latent: A new approach to self-supervised Learning
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




