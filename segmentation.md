## List
### I. Segmentation
1. MaskFormer: Per-Pixel Classification is Not All You for Semantic Segmentation
2. Mask2Former: Masked-attention Mask Transformer for Universal Image Segmentation
3. Panoptic SegFormer: Delving Deeper into Panoptic Segmentation with Transformers
4. Semantic Segmentation by Early Region Proxy
5. Rethinking Semantic Segmentation: A Prototype View
### II. Language-Guided Segmentation
1. Segmentation in Style: Unsupervised Semantic Image Segmentation with Stylegan and CLIP
2. Language-Driven Semantic Segmentation
3. DenseCLIP: Language-Guided Dense Prediction with Context-Aware Prompting
4. Panoptic Narrative Grounding
5. Decoupling Zero-Shot Semantic Segmentation

### III. Hierarchical Semantic Segmentation
1. Deep Hierarchical Semantic Segmentation
2. Deep grouping model for unified perceptual parsing

## I. Segmentation
### 1. MaskFormer: Per-Pixel Classification is Not All You for Semantic Segmentation
#### Introduction
#### Method

![image](https://user-images.githubusercontent.com/67745456/164975535-d60b2352-e3ab-48f3-a7c8-57063b7d4df3.png)

#### Conclusion

### 2. Mask2Former: Masked-attention Mask Transformer for Universal Image Segmentation
#### Introduction
#### Method

![image](https://user-images.githubusercontent.com/67745456/164975575-923a232f-4cc4-4ffe-a450-014ec954e8dc.png)

#### Conclusion


## III. Hierarchical Semantic Segmentation
### 1. Deep Hierarchical Semantic Segmentation
#### 대충 정리

기존의 구조들에 대해 class 구조만 tree 형태로 사용하고 그에 맞는 costraint와 loss만 바꿔서 사용할 수 있음
(기존의 hierarchical model들은 hierarchical class assignment를 위해 기존 model의 구조를 크게 바꾸거나 object 세부 분할에만 초점을 두고 있다고 한다(like person part task))
loss는 hierchical classification loss(일종의 contrastive loss (negative는 멀고 positive는 가깝게) + tree constraint)와 hierarchy properties를 이용한 representation loss를 이용하여 학습하고
비교는 기존의 성능 잘나오는 hierarchical-agnostic models (maskformer ...) 들과 비교 (다른 hierchical model들은 구조를 크게 바꿔서 비교가 불가, person part task에서는 비교했을 수도 있는데 확인 필요)
기존에 Hierarchy 구조가 포함된 Mapillary Vistas 2.0 같은 데이터셋과 비교적 class 수가 20개 정도로 적은 데이터셋에 대해 임의로 상위 클래스를 만들고 지정하여 비교
hierarcical-agnostic model에 대해서도 하위 classes score를 합쳐서 상위 class score로 하여 평가를 진행

top down 방식이 확실히 하위 레벨에서 성능이 잘나올것 같은데 높은 레벨에서 성능이 HSSN방식과 비교했을때 못 나롱 수도 있을 것 같음 (constrain이 super가 sub을 돕는데만 쓰이니까)








