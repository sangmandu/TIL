---
description: '210908'
---

# \(05강\) Sequence to Sequence with Attention

## 1. Seq2Seq with attention Encoder-decoder architecture Attention mechanism

### Seq2Seq Model

앞서 배운 RNN의 구조 중 Many to Many에 해당하는 모델이다. 보통 입력은 word 단위의 문장이고 출력도 동일하다.

![](../../../.gitbook/assets/image%20%281109%29.png)

이 때, 입력 문장을 받는 모델을 인코더라고 하고 하나하나 답을 내놓는 부분을 디코더라고 한다. 인코더와 디코더는 서로 다른 RNN 모델이다. 그래서 파라미터를 공유하거나 하지 않는다. \(인코더와 디코더 각각은 내부적으로 공유한다\)

또한, 내부 구조를 자세히 보면 LSTM을 채용한 것을 알 수 있다. 인코더의 마지막 단어까지 읽은 후 생성되는 마지막 스텝의 Hidden state는 디코더의 h0로서의 역할을 한다. 이 hidden state는 입력에 대한 정보를 잘 가지고 있다고 볼 수 있고 이를 바탕으로 디코더에서 사용한다고 볼 수 있다.

&lt;Start&gt; 토큰 또는 &lt;SoS&gt; \(Start of Sentence\) 토큰이 입력되면서 디코더가 작동되기 시작하며 &lt;End&gt; 토큰 또는 &lt;EoS&gt; \(End of Sentence\) 토큰이 나올 때 까지 디코더 RNN을 구동한다. 

Hidden state의 크기는 처음에 고정하기 때문에 아무리 짧은 문장이라도 hidden dimension만큼의 정보를 저장해야 하고, 아무리 긴 문장이라도 hidden dimnesion 만큼으로 정보를 압축해야 한다.

또, LSTM이 Long Term Dependency를 해결했다고 하더라도 훨씬 이전에 나타난 정보는 변질되거나 소실된다. 그래서 문장이 길다보면 첫번째 단어에 대한 정보가 적기 때문에 디코더의 시작부터 품질이 나빠지는 문제가 발생한다. 이에 대한 테크닉으로 "I go home" 으로 입력하는 것이 아닌 "home go I"로 입력해서 문장의 초반 정보를 잘 유지할 수 있도록 한다.

디코더는 인코더에서 마지막으로 나온 hIdden state를 h0으로 사용하지만 이것만을 사용하지 않는다. 인코더의 각 time step에서 나온 hidden state를 모두 제공받고 이 중 선별적으로 사용해서 예측에 도움을 주는 형태로 활용한다. 이것이 attention 모듈의 기본적인 아이디어이다.



### Seq2Seq Model with Attention

![](../../../.gitbook/assets/image%20%281108%29.png)

hidden state가 4개의 차원으로 구성되었고 프랑스어를 영어로 변환하는 과정을 예시로 든 이미지이다. 다음과 같은 순서로 구성된다.

* 인코더에서 입력별로 hidden state가 생성되며 최종 hidden state가 디코더에 제공된다.
* 디코더는 h0와 &lt;sos&gt; 토큰을 가지고 첫번째 h state를 생성한다.
* 첫번째 h state는 인코더의 각각의 h state와 내적을 하게 된다.
  * 내적을 한다는 것은 유사도를 비교하겠다는 의미.
* 이후, 각 유사도를 sofrmax한 값을 가중치로 얻게된다.
* 이 때 attention output 벡터는 가중평균된 벡터이며 context 벡터라고도 부른다.

![](../../../.gitbook/assets/image%20%281110%29.png)

* 이후 디코더는 디코더의 h state와 attention output 을 concat 하며 예측값을 반환하게된다.

![](../../../.gitbook/assets/image%20%281107%29.png)

* 마찬가지로, 디코더의 두번째 step에서도 동일한 메커니즘이 적용된다.
* &lt;eos&gt; 토큰이 나올때까지 작동된다.

정리하면 RNN의 디코더는 1\) 다음 단어를 예측하고 2\) 인코더로부터 필요로 하는 정보를 취사선택하도록, 학습이 진행된다. 역전파에 관점에서도, Attention 벡터가 다시 선택될 수 있도록 인코더의 hidden state가 갱신된다. 인코더의 h state가 갱신되므로 당연히 디코더의 h state도 갱신된다.

학습을 할 때에는 디코더의 각 타임스텝의 예측값이 무엇이든 간에 Ground Truth 값을 넣어주게 되지만 추론을 할 때에는 이전 타임스텝의 예측값을 다음 타임스텝의 입력값으로 넣어주게 된다.

* 이렇게 학습 중에 입력을 Ground Truth로 넣어주는 방법을 `Teacher Forcing` 이라고 한다.
* 물론, 학습은 잘 되지만 실제로 우리가 적용해야 하는 문제는 `Teacher Forcing` 과는 괴리가 있다. 그래서 이를 섞어서 사용하는 방법이 나왔는데, 학습 초반에는 빠른 학습을 위해서 이를 적용했다가, 학습이 어느 정도 되고나서는 적용하지 않도록 하는 방법도 존재한다.

### Different Attention Mechanisms

이전에는 유사도를 구하기 위해 내적을 사용했는데, 내적 이외에도 다양한 방법으로 attention을 구성하는 방법을 알아보도록 한다.











