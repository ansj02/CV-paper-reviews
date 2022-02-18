Perceptual Loss for Robust Unsupervised Homography Estimation

![image](https://user-images.githubusercontent.com/67745456/154397323-7ffc6652-6bd7-484d-9f33-b03cdefa1233.png)

![image](https://user-images.githubusercontent.com/67745456/154397405-007a8a86-a78f-4ad6-81b0-d21dbe47e830.png)

![image](https://user-images.githubusercontent.com/67745456/154397429-a2333f9f-67dc-4c60-af76-f97ae0277886.png)

![image](https://user-images.githubusercontent.com/67745456/154397463-9aca135c-1eb0-4146-a20d-e9d934e3966e.png)

![image](https://user-images.githubusercontent.com/67745456/154397486-b31691a9-9a4a-4a94-bab4-3c76277f1e4e.png)




DFM: A Performance Baseline for Deep Feature Matching

![image](https://user-images.githubusercontent.com/67745456/154414769-220d664a-cd96-4a10-b72b-6e3f1a41efb0.png)

![image](https://user-images.githubusercontent.com/67745456/154414832-0ba0ce3e-7acb-47b3-8b5c-c0ae7caa36f2.png)




SuperGlue: Learning Feature Matching with Graph Neural Networks

현실을 대상으로 하다보니 부드러운 등의 특징들이 있고 좀 자주 나타나는 poses가 있다

![image](https://user-images.githubusercontent.com/67745456/154482100-4425869c-f32b-4db0-bdf5-691f48541490.png)

attention을 transformer가 아니라 gnn으로

![image](https://user-images.githubusercontent.com/67745456/154482209-d0a1cc8b-d587-4552-a6a5-72b5124ccdfe.png)

attention 잘 됐고 position 정보 활용




GLU-Net: Global-Local Universal Network for Dense Flow and Correspondences

![image](https://user-images.githubusercontent.com/67745456/154482816-416f1ece-851a-4243-afa2-99e9eb21972f.png)

feature extractor가 fixed pre-trained backborn이니까 결국 학습되는건 flow estimation decoder란 warp module 밖에 없음, 

warp module  gangealing으로 pre training시키고 전체 fine tuning해보면?





Learning Accurate Dense Correspondences and When to Trust Them

PDC-net

flow estimation을 바로 하지 않고 확률로 표현 P(Y|X) confidence 표현 가능

probabilistic deep learning 그냥 바로 estimation해서 wxhx2로 나타내버리면 잘못된 match들 없앨 때 잃는 정보가 많다.

자주 쓰는 probabilistic model인 Laplacian model은 성능은 좋지만 inlier, outlier 구별 잘 못함->mixture model

over-smoothed and overly confident predictions, unable to identify ambiguous and unreliable matching을 피하기 위해 assesses the uncertainty at a specific location (i, j), without relying on neighborhood information

confidence map 형태로 rich information about matching ability of location을 

![image](https://user-images.githubusercontent.com/67745456/154492424-8a47fe7f-6ec6-4edd-a4de-843ca0a100d2.png)




GOCor: Bringing Globally Optimized Correspondence Volumes into Your Neural Network

![image](https://user-images.githubusercontent.com/67745456/154596253-0ecb3408-5074-4129-ad9d-d6020750f46f.png)

![image](https://user-images.githubusercontent.com/67745456/154596289-81d1e7ee-4f07-484e-a34b-72ceb4c5ef21.png)




pre trained feature extractor (VGG)
warping (homography)
augmentation
Hierarchical
matual consistency
contrastive learning   -> gangealing  어차피 pre training 할거니까 cost는 별로 상관없음, byol method 같이 쓸 수 있나? representation metric 학습과 trasformer 학습 차이 c에 모이게 하는 것도 metric으로 볼 수 있으니까
epipolar geometry



gangealing에서 pre train시킨 여러 c type STN들로 fixed STN 만들고
GLU net  upsampling 마다 맞는 c 골라서 warp 시켜도 괜찮을 듯

아니면 dense map에서 homography translation 하고 coarse to fine하게 resolution 늘리면서 flow map 찾고 pixel wise translation, flow map 따라 pixel 옮기면 남는 부분은 interpolation?

한 번만 homography

그 다음 low resolution부터 high resolution까지 flow

먼저 덩어리를 맞추고 내부의 세부 요소 맞춘다.


low resolution에서 얻은 homography matrix로 high resolution에 적용 한 번하고 다시 resolution 낮춰서 coarse to fine   flow estimation




correlation 안하고 feature map 하나에서 key point 찾으려면 그냥 feature값 다 합친 절댓값 큰 거 찾으면 될 듯
correlation을 row resolution에서 밖에 못쓰는 이유가 모든 pixel wise score를 전부 계산해서니까
각 feature map에서 찾은 high feature score point에서만 correlation 연산하면 high feature map에서도 쓸 수 있지 않을까?

기존 pyramid feature map은 resolution만 다른게 아니라 다른 level의 feature들을 가짐, coarse feature map을 upscaling해서 같이 쓰면 어떨까

2D image 3D model 만들면 다른 pose matched point 쉽게 찾을 수 있다.


global, local method를 직렬로 배치하다보니 결국 한가지 방법의 의존성을 가져온다. 병렬로 처리?




residual 방식으로 연결할 때 앞 단계에서 사용한 warp 가져오는 correlation source 좌표에도 적용해야됨 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


global correlation map이 처음 한 번 크게 보정할 때만 쓰이니까 혼자 크게 움직인 object가 있으면 감지가 어려움 real scene은 3D이므로 object들이 다 정지해 있다고 가정해도 view에 따라 object들의 displacement가 다 다름

처음 homography로 매칭시키니까 처음 매칭 대상을 평면으로 가정해야하는 문제










![image](https://user-images.githubusercontent.com/67745456/154647743-b7cad196-c4ef-4b49-9892-e7db30a826ff.png)

![image](https://user-images.githubusercontent.com/67745456/154647837-f5ecd8ca-302c-4f3a-be92-39b6140bc2e3.png)

![image](https://user-images.githubusercontent.com/67745456/154648086-a1800d68-eee6-4155-bc85-93c0591e79b8.png)















