---
description: '210913'
---

# \(08강\) Transformer \(2\)

## 2. Transformer\(cont'd\)

cont'd 는 continued의 약자이다. 이전과 이어진다는 의미

### Transformer : Multi-Head Attention

![](../../../.gitbook/assets/image%20%281148%29.png)

![](../../../.gitbook/assets/image%20%281154%29.png)

single attention 방식을 병렬적으로 하는 방법이다. 방법은 똑같으며 내부 파라미터만 다르기 때문에 여러 가지 output이 나온다. 방법론적으로는 앙상블의 느낌으로도 볼 수 있다. 최종 output은 concat하게된다.

왜 하는걸까? 단순히 모델을 여러가지 방법으로 조작하면서 일반화 성능을 높이기 위해서? 도 맞는 말이지만 이를 좀 더 자세하게 이야기 할 수 있다. 각 시퀀스마다 병렬적으로 서로 다른 정보를 얻어서 해당 시퀀스에 대해 풍부한 정보를 가지고 output을 내기위함으로 볼 수 있다.

![](../../../.gitbook/assets/image%20%281151%29.png)

![](../../../.gitbook/assets/image%20%281158%29.png)

만약 8번의 attention을 실행했다면 각각의 결과를 concat하므로 가로로 매우 길어진 최종 output을 얻게된다.

여기에, 선형 layer를 적용해서 어떤 W와의 곱을 통해 최종적으로 Z 벡터를 얻게된다.

Multi head Attention에서의 계산량을 알아보자.

![](../../../.gitbook/assets/image%20%281145%29.png)

#### Complexity per Layer

Self-Attention은 RNN 보다 훨씬 많은 연산량과 메모리가 발생한다.

* d는 하이퍼 파라미터라서 조절할 수 있는데 비해 n은 주어진 데이터에 따라 결정되는 부분이라서 데이터가 크면 클수록 많은 연산량을 필요로 한다.

#### Sequential Operations

Self-Attention은 병렬적으로 이를 처리하면 한번에 처리할 수 있지만, RNN은 이전 step이 끝나야 다음 step을 진행할 수 있으므로 병렬화할 수 없다. 그래서 RNN은 Forward & Backward Propagation은 sequence의 길이만큼 시간이 소요된다. 

* 실제로 입력은 한번에 주어지므로 한꺼번에 처리되는 듯이 보이지만 위와 같은 이유때문에 절대 병렬화가 이루어질 수 없다.

정리하면, RNN은 연산량이 작지만 속도는 느리고, Self-Attention은 연산량이 큰대신 속도는 빠르다.

#### Maximum Path Length

Long Term Dependency와 관련이 있는 부분이다.

RNN에서는 마지막 step에서 첫번째 단어의 정보를 얻기위해 n개의 레이어를 지나와야 하지만, T.F 에서는 time step 과 관련없이 attention을 이용해 직접적으로 정보를 가져올 수 있다.



### Transformer : Block-Based Model

![](../../../.gitbook/assets/image%20%281156%29.png)

* 아래에서 부터 세 갈래로 나누어지는데 모두 K, Q, V 를 의미한다. 이들은 개별적인 head attention에서 각각의 Wk, Wq, Wv를 얻게되며 이를 모두 concat해서 output을 반환한다.

여기서 처음보는 부분이 있다. 바로 Add & Norm

* Residual 연산인 Add가 수행되고 Layer Normalization이 수행된다.
* 이후, Feed Forward를 통과하고 또 수행이 된다.

#### Add

* 깊은 레이어에서 Gradient Vanishing 문제를 해결하고 학습을 안정화하여 더 높은 성능을 내게하는 기술이다.
* 만약 "I study math" 라는 문장에서 "I" 에 해당하는 임베딩 벡터가 \[1, -4\] 이고 head attention을 통과한 인코딩 벡터가 \[2, 3\] 이라고 하자. 이 때 add를 적용하면 두 벡터를 더해서 \[3, -1\] 을 얻게되고 이를 "I"의 최종 인코딩 벡터로 결정한다.

![](../../../.gitbook/assets/image%20%281147%29.png)

몇 가지 Normalization이 존재하는데 이중에서 Batch Norm과 Layer Norm 알아보자.

#### Batch Normalization

![](../../../.gitbook/assets/image%20%281137%29.png)

* 각 배치의 값의 평균과 표준편차를 구하고 이를 이용해 각 배치를 평균이 0이고 표준편차가 1인 정규분포를 따르도록 정규화해준다.
* 이후 Affine Transformation을 적용해서 원하는 평균과 분산으로 맞춰준다.

#### Layer Normalization

![](../../../.gitbook/assets/image%20%281136%29.png)

* Batch Norm. 은 한 batch에 대해서\(=가로로\) 정규화했다면 Layer Norm.은 한 Feature에 대해서\(=세로로\) 정규화한다.



### Transformer : Positional Encoding

만약에 우리가 지금까지 본 모델에서 "I love you" 와 "love I you"를 입력했을 때의 결과는 항상 똑같을 것이다. 왜냐하면 Transformer는 time step을 고려하지 않고 입력에 대해 한번에 처리하기 때문에 순서를 고려하지 않고 처리하기 때문이다.

순서를 고려해주는 방법이 필요하다. 다음과 같은 예를 들어보자.

"I Study math" 에서 "I"의 인코딩 벡터가 \[3, -2, 4\] 라고 하자. 그러면 I는 첫번째 순서에 나왔으므로 벡터의 첫번째 값에 상수 1000을 더해서 \[1003, -2, 4\] 로 만들어주는 방법이 Positional Encoding의 아이디어이다.

* 순서에 따라 벡터가 다른 값을 가지게 된다.
* 여기서는 간단하게 1000을 더해줬지만 실제로는 간단하게 이루어지는 부분은 아니다.

위치에 따라 구별할 수 있는 벡터를 sin과 cos함수로 이루어진 주기함수를 사용해서 결정한다.

![](../../../.gitbook/assets/image%20%281144%29.png)

![](../../../.gitbook/assets/image%20%281139%29.png)

dimension 개수만큼 서로 다른 그래프가 존재하며 각 sequence의 인덱스를 x값이라고 할 수 있다.

![](../../../.gitbook/assets/image%20%281133%29.png)

위 그래프에서는 가로축은 임베딩 차원, 세로축은 인덱스\(=위치\)이다. 그래서 해당 인덱스에 해당하는 임베딩 차원만큼의 벡터를 positional encoding 벡터로 사용해서 기존 벡터에 더해주게 된다.



### Transformer : Warm-up Learning Rate Scheduler

우리는 loss가 가장 작은 지점을 목표로 학습을 할 것이고 이 때의 파라미터들은 임의로 초기화하게 되는데 아무래도 Goal과는 대부분 멀리 존재할 가능성이 크다. 또한, 이 때는 Loss 함수 특성상 멀리있을 수록 Gradient가 매우 클 가능성이 높다.

![](../../../.gitbook/assets/image%20%281150%29.png)

* "gradient 매우 큼" 이라고 작성된 것임

그래서, 초반에 너무 큰 gradient를 가지고 있기 때문에 너무 큰 보폭으로 걷지 않게 조절하기위해 작은 학습률에서 시작해서 학습률을 키워나간다. 그리고 목표지점에 가까워질 때 학습률이 너무커서 수렴하지 못하는 문제가 발생하지 않도록 하기 위해 다시 학습률을 감소시키는 방향으로 학습을 하게 된다.

![](../../../.gitbook/assets/image%20%281132%29.png)

* 그래프의 범주에서 앞 숫자는 batch size 뒷 숫자는 epoch 수를 의미한다.
* batch size가 작을수록 학습률의 상승 곡선의 기울기를 크게 가지며, epoch수가 적을수록 최고점이 낮아지고 도달속도도 오래걸리게된다.



### Transformer : Encoder Self-Attention Visualization

Attention 벡터를 분석해 시각화해보자.

![](../../../.gitbook/assets/image%20%281157%29.png)

* 주어진 문장에서 making 이라는 단어는, 자기 자신도 참조 하지만 more와 difficult라는 단어를 가장 많이 참조하는 것으로 알 수 있다. 더욱 어렵게 만들었다라는 목적 보어의 단어들을 참조한다. 또, 2009와 since라는 시기적인 의미의 단어도 조금 참조한다.

다른 단어를 보자.

![](../../../.gitbook/assets/image%20%281140%29.png)

* its는 어떤 단어를 가리키는 지에 대해 알 수 있고, 이러한 its에 대해 application이라는 단어가 어느정도 관련이 되어있음을 알 수 있다.





