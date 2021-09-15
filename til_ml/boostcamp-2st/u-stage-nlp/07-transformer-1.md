---
description: '210913'
---

# \(07강\) Transformer \(1\)

## Transformer

기존에 Add-on 으로만 사용되는 Attention을 전반적으로 사용하고 RNN과 CNN 모듈을 사용하지 않는 모델이다.

![](../../../.gitbook/assets/image%20%281141%29.png)

* 이전 정보들을 hidden state에 담아 넘기는 모습이다. hidden state와 각 임베딩 벡터와의 관계는 오른쪽과 같다.
* 그러나, 어쩔 수 없이 각 time step을 거치면서 정보가 손실될 수 밖에 없는 구조이다.

양방향 RNN에 대한 구조는 다음과 같다.

![](../../../.gitbook/assets/image%20%281162%29.png)

* 예를 들어 `GO` 를 기준으로 본다면, go의 왼쪽 단어들에 대한 정보가 담겨있는 Forward RNN의 hf 와 go의 오른쪽 단어들에 대한 정보가 담겨있는 Backwrad RNN의 hb를 concat해서 기존 hidden state의 2배 크기로 만들 수 있는데, 이것을 go의 인코딩 벡터로 생각할 수 있다.

Transformer의 구조는 다음과 같다.

![](../../../.gitbook/assets/image%20%281137%29.png)

* 입력과 출력의 크기는 유지되며 입력 벡터의 정보를 잘 반영할 수 있도록 한다. 구체적으로 알아보자
* 초기에는 주어진 임베딩 벡터들을 가지고만 연산을 했다.
  * 만약, 첫번째 time step에 대한 벡터를 반환한다고 하면, I와 I, I와 go, I와 home을 내적해서 반환했다. \(I가 기준이 되고 자신을 포함한 나머지 임베딩 벡터와 내적을 한 것\)
  * 자신과 자신을 내적하면 비교적 다른 벡터와의 내적보다 값이 크게 나오기 때문에 반환된 값들이 대체로 본인과 내적한 값이 큰 분포의 양상을 보였다.
* 그래서, 기준이 되는 임베딩 벡터의 Wq 를 곱해서 얻은 q1 이라는 쿼리벡터를 얻게되고 모든 임베딩 벡터는 이 쿼리벡터와 매칭된다. \(매칭이 될 뿐 내적되는 것은 아니다! 바로 이어서 설명!\)
  * Wq도 그냥 정하는 것이 아니라, 매번 학습을 통해 쿼리벡터에 대한 최적의 가중치를 결정하게 된다.
* 이 때 임베딩 벡터들과 쿼리 벡터를 바로 내적하는 것이 아니다. 각 임베딩 벡터들과 Wk를 곱해서 얻은 k 라는 키벡터를 얻어야 한다. 그래서 이 키벡터와 쿼리벡터를 내적하게 된다.
  * 모든 임베딩 벡터들은 쿼리벡터와 키벡터를 하나씩 가지고있다고 보면 된다.
  * 어떤 기준이 되는 임베딩 벡터가 있으면, 그 때 사용되는 쿼리벡터는 1개, 키벡터는 n개 이다. \(n은 임베딩 벡터의 개수\)

정리하자면, 임베딩 벡터는 기존에는 유사도를 구하는 재료벡터로 쓰였지만, 지금은 유사도를 구하기 위해 필요한 쿼리벡터와 키벡터의 재료벡터로 쓰이게 되는 것이다. 또, 유사도와 곱해지는 가중치 벡터를 구하기 위한 재료벡터로도 쓰이며 이 것이 밸류벡터이다.

* 밸류벡터는 임베딩 벡터\(=재료 벡터\)와 Wv와 곱해져서 얻어지며 키벡터와 쿼리벡터를 내적해서 나온 값에 softmax를 취하고 얻어진 값에 가중평균으로 곱해져서 최종 인코딩 벡터를 얻게된다.

이렇게 구성을 하면, 비록 자신의 쿼리벡터와 자신의 키벡터의 내적을 거쳐 구한 값이라고 할지라도 다른 벡터들보다 값이 작을 수도 있다는 특징이 있다.

또, 타임스텝에 상관없이 각각의 고유 정보만을 가지고 인코딩 벡터에 기여할 수 있다는 장점이 있다.



정리해볼게요!!

* 결과물\(=인코딩 벡터\)은 밸류벡터의 가중합으로 계산된다!
* 이 때의 가중치는 각각의 임베딩 벡터의 쿼리벡터와 키벡터의 내적값으로 계산된다!
* 내적을 해야하므로 쿼리벡터와 키벡터는 차원이 동일해야한다!
* 밸류벡터의 차원은 꼭 동일하지 않아도 된다.

이를 식으로 나타내면 다음과 같다.

![](../../../.gitbook/assets/image%20%281164%29.png)

* A : Attention 모듈에서는
* q, K, V : 쿼리 벡터 한개, 키 벡터 전체, 밸류 벡터 전체가 필요하며,
  * 쿼리 벡터가 한개라 소문자로 쓴 디테일!!

![](../../../.gitbook/assets/image%20%281158%29.png)

* 쿼리벡터 하나와 키 벡터 모두를 내적하여 이에 대한 softmax값을 구하고,

![](../../../.gitbook/assets/image%20%281145%29.png)

* 이를 밸류벡터와 가중합해서 최종 결과물을 얻는다!

그리고 각각의 결과물을 구하는 과정을 행렬을 이용하여 전체 과정으로 이해할 수 있다.

![](../../../.gitbook/assets/image%20%281131%29.png)

* GPU를 사용해서 행렬 연산을 빠르게 할 수 있기 때문에 Transformer는 기존 RNN 모델보다 학습을 더 잘할 수 있게된다.
* 실제 Transformer를 구현했을 때는 Q, K, V의 shape가 모두 동일했다.



트랜스포머의 과정을 그림으로 나타내면 다음과 같이 나타낼 수 있다.

![](../../../.gitbook/assets/image%20%281146%29.png)

* 근데 저기서 $$ \sqrt d_k $$라는 값으로 나누어주는 부분이 있는데 이건 뭘까?



다음의 예시가 있다고 하자.

![](../../../.gitbook/assets/image%20%281135%29.png)

그리고, a와 b, x와 y는 각각 독립이면서 평균이 0이고 분산이 1인 분포의 확률변수라고 가정하자.

이 때, ax는 마찬가지로 평균이 0이고, 분산이 1이되며 by도 마찬가지로 평균이 0이되고, 분산이 1이된다.

그리고 ax+by는 평균이 0이고, 분산이 2가 된다.

![https://ko.khanacademy.org/math/statistics-probability/random-variables-stats-library/combine-random-variables/a/combining-random-variables-article](../../../.gitbook/assets/image%20%281161%29.png)





그래서 의도치않게 차원을 크게하고 Scaling이 따로 없다면, 소프트맥스가 특정 값에 몰리게 되고 이로 인해 Gradient Vanishing 문제가 발생할 수 있다.












