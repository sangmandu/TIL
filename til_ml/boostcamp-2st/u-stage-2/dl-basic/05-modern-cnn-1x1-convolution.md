---
description: '210811'
---

# \(05강\) Modern CNN - 1x1 convolution의 중요성

강의 제목은 Modern CNN이지만, 비교적 과거의 CNN을 비교했을 때의 그렇다는 거지 정말 Modern 하지는 않다. 좀 더 최신 CNN은 이후 Vision에서 다시 다룬다.

## ILSVRC

**I**mageNet **L**arge-**S**cale **V**isual **R**ecognition **C**hallenge

* Classification/Detection/Localization/Segmentation
* 1000개의 다른 범주
* 100만장이 넘는 이미지

![](../../../../.gitbook/assets/image%20%28858%29.png)

해마다 발전하는 딥러닝의 인지 능력

## AlexNet

![](../../../../.gitbook/assets/image%20%28854%29.png)

당시에 GPU가 부족해서 두 개로 분할해서 학습했다.

시작부터 커널의 크기가 11x11 인데, 이는 좋은 크기 선택은 아니다. 물론 한 커널에서 이미지를 넓게 볼 수 있지만 그만큼 파라미터의 수가 매우 많아지게 된다.

5개의 Conv Layer와 3개의 Dense Layer로 이루어져 있는 8단 Layer이다

* 최근에 나오는 모델들은 쉽게 200~300단을 넘어가는 것을 보면 이 정도는 Light한 모델이다

알렉스넷이 왜 성공했을까?

* ReLU
  * 비선형 함수이면서 네트워크를 망칠 수 있는 요소가 적다
* 2개의 GPU 사용
* Local response normalization
  * 어떤 입력공간에서 response가 많이 나오면 죽여버리는 것.
  * 지금은 많이 활용되지 않는다
* Overlapping pooling
* Data augmentation
* Dropout

지금은 당연한 것들이 그 당시에는 최초였다. 지금의 근간이 되준 딥러닝 모델로 볼 수 있다.

ReLU가 왜 좋을까?

* 선형 모델의 특징을 유지한다
* GD를 최적화하기 쉽다
* 일반화에 좋다
* Gradient Vanishing 문제를 극복한다

## VGGNet

알렉스넷은 11, 5, 3의 크기의 필터를 사용한데 비해 3 크기의 필터만 사용했다. 왜 3x3 Conv. 만 사용했을까?

필터의 크기가 커졌을 때 장점은 이미지와 Conv 했을 때 얻는 정보인 Receptive Field의 크기가 커진다는 것이다. 

3x3을 두번 하는 것과 5x5를 한번 하는 것은 동일한 결과를 낸다. \(Receptive Field가 똑같음\) 그러나 파라미터의 수는 3x3는 294,912개, 5x5는 409,600개이다.

* 왜? 3x3=9 이고 레이어를 2층 쌓으면 18인데 비해,
* 5x5=25기 때문에 18 &lt; 25 라서 파라미터 수가 더 많다.

## GoogLeNet

3x3을 여러번 쓰는것이 큰 필터를 쓰는것보다 낫다는 것을 알게되었다. 그래서 이후로는 이런 방식을 채택했다.

![](../../../../.gitbook/assets/image%20%28864%29.png)

구글넷은 동일한 구조가 반복되는 구조이다. 그래서 이런 것을 Network In Network, NIN 구조라고 한다.

![](../../../../.gitbook/assets/image%20%28849%29.png)

Inception blocks를 활용했는데 이는 입력이 들어오면 모두 펼치고 다시 종합하는 구조이다. 장점은 여러가지 Receptive Field를 종합한다는 점도 있지만 1x1 Conv이 중간중간에 끼게 되면서 네트워크의 전체적인 파라미터 수가 감소하게 된다.

* 1x1 필터는 채널의 수를 감소시키는 효과가 있다.

![](../../../../.gitbook/assets/image%20%28850%29.png)

Inception Block을 사용했을 때 파라미터 수가 1/3로 줄어드는 모습

## ResNet

Kaiming He가 만든 모델.

일반적으로 파라미터가 많게되면 다음과 같은 문제가 있다.

* 오버피팅 : train error는 감소하지만 test error는 증가하는 것
  * 여기서 다룬 문제는 이것이 아님
* 네트워크가 커질수록 비교적 작은 네트워크보다 학습이 어렵다.

![](../../../../.gitbook/assets/image%20%28853%29.png)

ResNet은 두번째 문제를 해결하기 위해 Identity map이라는 것을 추가했다. 모델의 출력값에 입력값을 더하는 것

* 이로 인해 네트워크를 깊게 쌓을 수 있게 되었다.

![](../../../../.gitbook/assets/image%20%28846%29.png)

기본적으로 Simple 방법을 사용한다. 근데 이 때 입력과 출력을 더해주려면 차원이 같아야 하는데 차원이 다를 수도 있으니 1x1 Conv를 이용해 차원을 동일하게 한뒤 더하는 것이 Projected

* 일반적으로 Simple을 많이 사용한다

현재는, Activation 전에 BN을 적용하는데 이 순서에 대한 논란이 있다. 순서를 바꾸면 성능이 더 잘나온다는 의견이 있는것. 원래 논문에서는 순서가 BN을 먼저했다.

## DenseNet

Identity map에 대해서 입력과 출력을 더하는 것이 아니라 concatenate\(잇다\)하는 방법. 이렇게 되면 점점 채널이 커지게 되고 이에 따라 필터의 채널도 커지게 된다. 결국 파라미터수의 증가

![](../../../../.gitbook/assets/image%20%28852%29.png)

그래서 Densenet에서는 중간중간마다 채널의 수를 1x1 Conv로 줄이게 된다. Densenet은 Dense Block Transition Block으로 이루어져있는데, Dense Block에서는 계속적으로 채널을 늘리고 Transition Block에서는 BN, 1x1 Conv, 2x2 AvgPooling을 거치면서 채널수를 줄이게 된다.

## 정리

VGG : 3x3 block만 사용

GoogLeNet : 1x1 Conv 사용

ResNet : skip-connection\(=identity map\)

DenseNet : concatenation











