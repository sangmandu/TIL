---
description: '210826'
---

# DAY 4 : VIT \| F1\_Loss \| LrScheduler

## VIT

![](../../../.gitbook/assets/image%20%281025%29.png)

나름 이미지넷에서 Rank 1, 2의 스펙을 가지고 있는 VIT를 써보기로 했다. 이전에 쓴 efficientnet 같은 경우는 직접 다운받아서 불러왔는데, `timm` 이라는 라이브러리를 사용하면 원하는 모델을 쉽게 import 할 수 있었다. 단순히 `torchvision.models` 와는 종류도 더 다양하고 최신이며 개별 모델의 종류도 굉장히 다양하다.

오늘 하루는 거의 VIT의 성능을 끌어올리기 위해 썼다. 결론은,

> 성능이 쥐뿔도 나오지 않는다

오늘 고생한 것들을 사진과 코드로 좀 남겨둘 걸 그랬다. 거기까지는 생각 못하고, 삽질을 하다보니 지쳐서 우발적으로 VIT와 관련된 파일을 다 지워버렸다. ~~속은 시원했다~~

이 부분에서 멘토님과 이야기해봤는데, 멘토님도 VIT로 성능을 내본적이 없다고 했다. VIT를 학습하기 위한 데이터셋이 보통 모델과 비교해서 훨씬 많아야 되는데, 우리가 가진 만개정도의 데이터셋으로는 충분한 학습을 거치기 힘들것이라고 했다. ~~EfficientNet 만세~~



## F1\_LOSS

지금 사용하는 Loss 함수는 `nn.CrossEntropyLoss` 인데,  그 외에도 다양한 Loss 함수가 있었다. 그래서 이것들을 실험해봤다.

![](https://lh4.googleusercontent.com/MoM3Nj2A5wlwljwnlDYCyh4c7x7FSQZu5q1Rmijb6IVDZ7WP2rVxnGqBFk3xMenpLtn_GRSLpTiMvjPd6astZK8Re1iFcEQpkZ1A1regJM-bAZFjlRx4j1mz_kduArV2A10SWd57=s0)

실제로 실험상으로는 `Symmetric` 함수가 제일 높은 valid f1 점수를 냈다. 다만 `focal` 함수와도 뒤치닥 거릴 정도로 비슷했다. 반면 제일 기대했던 `f1` 함수는 성능이 제일 안나왔으며 `cross entropy` 와 `label smoothing` 은 성능이 거의 동일했다.

처음에는 symmetric을 주로 쓰다가, focal도 번갈아가면서 써보았다. 뭐가 좋은지는 확실히 알 수가 없지만 focal이 성능이 더 잘나오는 느낌을 받았다.



## LrScheduler

![](https://lh6.googleusercontent.com/wEO-74vcia_HAsgRWjGyzlrli77jz7r_CxjI4GBENqS5_sejMIQvOoY25D_yaNDtDgiRFmdHUHzAL5674UbzVxfqozZntpudjyznT76rzqMEQWm4JXaXhlabIWDUXvCutMCBFF3H=s0)

스케쥴러의 성능은 다음과 같다. 물론 세부적인 하이퍼 파라미터 차이도 있을 것이다. 내가 CyclicLR을 사용할 때 쓴 파라미터는 다음과 같다.

* STEP\_UP : 데이터로더의 길이
* STEP\_DOWN : 데이터로더의 길이 \* 2
  * 실제로는 len\(train\_dataset\)//CFG\['train\_bs'\] 로 구했다.
* base\_lr : 1e-6
* max\_lr : 1e-3
* cycle\_mometum = False
* mode = triangular2

triangular2의 학습률 변화 그래프는 다음과 같다.

![](../../../.gitbook/assets/image%20%281066%29.png)

이 그래프를 원했던 이유가 있다.

* 학습률은 점점 작아져야 한다. 그래야 최고의 성능을 내는 지점을 찾을 수 있다.
  * `mode = triangular1` 나 `cosine` 을 사용하지 않은 이유

또, 학습률을 1e-3과 1e-6으로 했을 때 가장 잘, 가장 빨리 찾았다.

* 1e-2나 1e-7로 했을 때는 더뎠다.

