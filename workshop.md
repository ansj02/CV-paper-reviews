Perceptual Loss for Robust Unsupervised Homography Estimation

![image](https://user-images.githubusercontent.com/67745456/154397323-7ffc6652-6bd7-484d-9f33-b03cdefa1233.png)

![image](https://user-images.githubusercontent.com/67745456/154397405-007a8a86-a78f-4ad6-81b0-d21dbe47e830.png)

![image](https://user-images.githubusercontent.com/67745456/154397429-a2333f9f-67dc-4c60-af76-f97ae0277886.png)

![image](https://user-images.githubusercontent.com/67745456/154397463-9aca135c-1eb0-4146-a20d-e9d934e3966e.png)

![image](https://user-images.githubusercontent.com/67745456/154397486-b31691a9-9a4a-4a94-bab4-3c76277f1e4e.png)



pre trained feature extractor (VGG)
warping (homography)
augmentation
Hierarchical
contrastive learning   -> gangealing  어차피 pre training 할거니까 cost는 별로 상관없음, byol method 같이 쓸 수 있나? representation metric 학습과 trasformer 학습 차이 c에 모이게 하는 것도 metric으로 볼 수 있으니까

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
