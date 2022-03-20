# NeRF-Supervision: Learning Dense Object Descriptors from Neural Radiance Fields

## Problems

scale, illumination, and pose에 invariant한 Dense descriptors를 얻는 것은 computer vision의 오랜 문제였다.

이를 위한 self supervised data를 얻기 위해 대표적으로 i)an object에 대해 synethetically rendering하여 multiple views를 얻어 ground truth를 가진 데이터를 제작하거나 ii)하나의 image에 random affine transformations 같은 augmenting을 가해 데이터 쌍을 제작하는 방법이 있지만 real data와 synthetic data의 gap에 의한 성능 저하나 정해진 transformation 방식과 다른 방식의 데이터에 대해서는 잘 작동하지 못하는 등 각각의 혹은 공통의 한계를 가진다.

또한, depth camera를 통한 a set of posed RGB-D images를 이용하여 dense correspondences를 얻는 방법도 thin, reflective objects에 대해 좋은 성능을 보이지 못하였다.

## Contributions

self-supervised pipeline for learning object-centric dense descriptors

a novel distribution-of-depths formulation, enabled by the estimated density field

