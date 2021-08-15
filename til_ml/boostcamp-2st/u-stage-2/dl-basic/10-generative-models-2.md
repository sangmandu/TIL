---
description: '210813'
---

# \(10강\) Generative Models 2

## Latent Variable Models

많은 사람들이 Variational encoder가 generative model이기 때문에 Autoencoder 역시 gen model이라고 생각한다. 사실은 그렇지 않다.

* 그래서 Variational autoencoder가 autoencoder를 gen model이 되도록 한것이 무엇인지 알아보려고 한다



## Variational Auto Encoder

#### Variational inference

목적은 Posterior distribution을 찾는데 있다. Posterior distribution은 observation이 주어졌을 때 random variable의 확률분포이다.

![](../../../../.gitbook/assets/image%20%28909%29.png)

* 여기서 z는 latent variable이다.
* p\(x \| z\)을 우리는 likelihood라고 한다.

![](../../../../.gitbook/assets/image%20%28900%29.png)

일반적으로 Posterior distribution을 계산하기가 힘들다. 불가능할때도 많다. 그래서 이 분포와 가깝게 근사하겠다는것이 목적이고, 그 분포가 바로 Variational distribution 이다.

여기서는,  Kullback–Leibler divergence 라는것을 활용해서 Variational distribution과 Posterior distribution과의 차이를 줄이려고 한다.

![](../../../../.gitbook/assets/image%20%28918%29.png)

문제가 무엇이냐면, 애초에 P-dstb도 모르는데, V-dstb를 구하려고 한다는 것. 무엇인지도 모르는 것과 가까운것을 구하려는 것. 이 문제를 해결하는 것이 V-dstb에 있는 ELBO Trick이다.

![](../../../../.gitbook/assets/image%20%28929%29.png)

목적은 P-dstb와 V-dstb의 KL divergence를 줄이는 것이다. 근데 이것이 불가능. 그래서 ELBO, Evidence Low Bound 라는것을 계산하고 증가시킴으로써 목표항을 낮추려고한다.

![](../../../../.gitbook/assets/image%20%28919%29.png)

ELBO는 위와 같이 두 개의 텀으로 나뉘게 된다.

Reconstruction Term

* encoder를 통해서 x라는 입력을 latent space로 보냈다가 다시 decoder로 돌아오는 이 reconstruction loss를 줄이는 부분이다.

Prior Fitting Term

* latent space에 있는 입력들의 분포가 사전에 정해준 사전 분포와 비슷하도록 하는 부분이다.

이것을 잘 설명해서 구현까지 한 것인 Variational Auto-encoder가 gen model이 될 수 있다. 엄밀한 의미에서 explicit 모델은 아니고 impulse 모델.



이해가 안된것이 뻔하니 [자료조사](https://hugrypiggykim.com/2018/09/07/variational-autoencoder%EC%99%80-elboevidence-lower-bound/)를 통해 내가 이해한 것을 정리해보면,

* Auto encoder는 Encoder와 Decoder의 두 파트로 구성되며, 입력에 대한 특징을 추출해서 Latent variables에 담고, 이 변수로부터 다시 입력을 복호화 하는 알고리즘이다.
  * 이 때의 Latent variables가 있는 hidden space\(=layer\)를 latent space라고 한다.
  * 원래는 인코더를 통해 압축, 디코더를 통해 복원하는 역할을 수행했다.
* 여기서 이 Latent space로 데이터를 생성해 낼 수 없을까? 라는 점에서 VAE\(=Variational auto-encoder\)가 주목을 받는다. 문제는, 인코더에 입력되는 데이터 X가 너무 많고 고차원이라는 것. 그래서, 이 X의 분포를 내가 알고, 표현할 수 있다면 새로운 데이터 \_X 도 추정할 수 있지 않을까 라는 곳에서 시작한다.
* 그래서 우리는 이 X의 분포를 예상하기 위해 임의의 분포를 가정하고 이 분포를 X의 분포와 최대한 비슷하게 하려고 한다. 여기서 KL Divergence 개념이 나온다.
* KL Divergence는 두 확률 분포가 얼마나 다른지를 표현하는 방법이다. 두 분포가 같으면 0의 값을 갖고 두 분포가 다를 수록 값이 커진다. 그 외의 특징은 다음과 같다
  * 항상 0이상의 값을 가진다
  * 계산 순서에 따라 값이 다르다 : $$ D_{KL}(p||q) \ne D_{KL}(q||p) $$
  * 비교하는 두 분포가 가우시안 분포를 따르면 간략하게 표현이 가능하다
* 다시 원래 이야기로 돌아가면, 우리가 구하고자는 데이터 X의 분포식을 전개해보니 ELBO항과 KL항으로 나누어진다. 근데 여기서 KL항은 반드시 두 분포를 알아야 구할 수 있는 값이므로 ELBO항에 관심을 두게 된다. 그리고, 데이터 X의 분포식은 고정된 값이므로 ELBO값을 최소화 시키면 반대로, KL값이 최대화 될 것이라고 예상한다.

  * 마치 10 = 4 + 6 인데, 10이 고정값이라서 4를 1로 바꾸면 6이 9가 되는 원리



다시, 본문으로 돌아가서 AE가 왜 Gen 모델이 될수가 없었나 생각해보면, VAE는 latent space를 가지고 새로운 이미지를 생성할 수 있는데비해 AE는 단지 입력이 latent space를 거쳐서 출력으로 나올뿐이기 때문이다.

#### Key limitation

* VAE는 explicit한 모델이 아니기 때문에 likelihood를 구하기 어렵다. 그래서 interactable 모델이라고 불린다
* KL항은 반드시 미분이 가능해야 한다. 그렇지 않으면 풀수가 없음. 그래서 미분이 가능하게 하기위해 VAE는 주로 가우시안을 사용한다.
* 그래서 대부분 isotropic Gaussian을 사용한다.
  * 이는 모든 아웃풋이 독립적인 가우시안을 의미한다.
  *  그래서 가우시안을 사용하면 다음과 같이 이쁜 꼴이 나오게 된다.

![](../../../../.gitbook/assets/image%20%28916%29.png)



## Adversarial Auto Encoer

VAE의 가장 큰 단점은 인코딩을 활용할 때 KL을 사용한다는 것이다. 결국 가우시안 분포가 아닌 경우는 풀 수 없다는 것.

이 때는 GAN을 활용해서 latent distribution 사이의 분포를 맞춰주는해 AAE를 사용한다.



## Generative Adversarial Network

![](../../../../.gitbook/assets/image%20%28926%29.png)

GAN의 장점은 어떠한 Fixed Discriminator의 의해 학습이 진행되는 것이 아니라 Discriminator와 Generator가 서로 영향을 주며 학습이 진행되는 것

* GAN은 explicit 모델이다 ㅎ\_\_ㅎ

#### VAE vs GAN

![](../../../../.gitbook/assets/image%20%28899%29.png)

* 한쪽은 높이고 싶어하고 한쪽은 낮추고 싶어하는 minimax game과 같다

discriminator에 입장에서는 다음과 같다

![](../../../../.gitbook/assets/image%20%28927%29.png)

이 값을 항상 최대화 시키는 D는 다음과 같다. \(generator가 fix 되었다고 가정\)

![](../../../../.gitbook/assets/image%20%28905%29.png)

반대로, generator 입장에서는 다음과 같다

![](../../../../.gitbook/assets/image%20%28915%29.png)

그리고 이 값을 항상 최소화 시키는 G는 다음과 같다 \(discriminator가 fix 되었다는 가정\)

![](../../../../.gitbook/assets/image%20%28913%29.png)



### DCGAN

![](../../../../.gitbook/assets/image%20%28920%29.png)

이미지 도메인으로 활용한 GAN이 Dense Convolution GAN



### INFO-GAN

![](../../../../.gitbook/assets/image%20%28901%29.png)

z라는 것을 통해서 매번 이미지를 생성하는 것이 아니라 c라는 보조의 클래스를 이용하여 generate를 할 때 GAN이 특정 모드에 집중할 수 있게 해준다

* 특정 모드라고 함은, 식을 통해 Conditional Vector에 집중할 수 있게한다.



### Text2Image

![](../../../../.gitbook/assets/image%20%28924%29.png)

DALL-E 처럼 문장을 입력하면 이미지를 출력

* 이 모델이 먼저임



### Puzzle-GAN

![](../../../../.gitbook/assets/image%20%28921%29.png)

이미지의 부분을 입력하면 원본을 완성시켜주는 모델



### CycleGAN

![](../../../../.gitbook/assets/image%20%28928%29.png)

이미지 사이의 도메인을 바꿀 수 있는 모델

* EX\) 말을 얼룩말로 만듬
  * 보통은 이런걸 가능하게 하려면 동일 배경의 말 사진 하나와 얼룩말 사진이 하나 필요한데, 그런것 없이 말사진 잔뜩 그리고 얼룩말 사진 잔뜩 학습해서 할 수 있다



### Star-GAN

![](../../../../.gitbook/assets/image%20%28906%29.png)

단순히 이미지의 도메인을 바꾸는 느낌이 아니라, 이미지의 도메인을 세부적으로 선택할 수 있게하는 모델



### Progressive-GAN

![](../../../../.gitbook/assets/image%20%28910%29.png)

4x4부터 1024x1024까지 단계적으로 이미지 크기를 키우면서 저차원 이미지부터 고차원 이미지를 학습시킨다.















