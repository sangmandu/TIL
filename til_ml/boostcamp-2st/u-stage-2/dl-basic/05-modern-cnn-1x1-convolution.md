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

![](../../../../.gitbook/assets/image%20%28840%29.png)

해마다 발전하는 딥러닝의 인지 능력

## AlexNet

![](../../../../.gitbook/assets/image%20%28839%29.png)

당시에 GPU가 부족해서 두 개로 분할해서 학습했다.

시작부터 커널의 크기가 11x11 인데, 이는 좋은 크기 선택은 아니다. 물론 한 커널에서 이미지를 넓게 볼 수 있지만 그만큼 파라미터의 수가 매우 많아지게 된다.

5개의 Conv Layer와 3개의 Dense Layer로 이루어져 있는 8단 Layer이다

* 최근에 나오는 모델들은 쉽게 200~300단을 넘어가는 것을 보면 이 정도는 Light한 모델이다

알렉스넷이 왜 성공했을까?

* ReLU
  * 비선형 함수이면서 네트워크를 망칠 수 있는 요소가 적다











