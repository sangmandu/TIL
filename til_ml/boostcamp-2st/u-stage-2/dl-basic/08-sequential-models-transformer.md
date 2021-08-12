---
description: '210812'
---

# \(08강\) Sequential Models - Transformer

시퀀스는 다음과 같은 시퀀스들 때문에 모델을 구현하기 어렵다.

![](../../../../.gitbook/assets/image%20%28847%29.png)



## Transformer

트랜스포머는 RNN과 달리 재귀적인 부분이 없고 `attention`이라는 것을 사용했다

이 트랜스포머는 인코딩의 작업을 거치기 때문에 단순히 NMT 문제에만 적용되지 않는다. 이미지 분류나 탐지 또는 DALL:E 등에 적용될 수 있다

* NMT는 Nerual machine translation의 약어이다. 
* DALL:E 는 단어를 제시하면 단어에 해당하는 이미지를 주는 것



불어를 영어로 바꾸는 등의 과정을 Sequence To Sequnce 라고 한다.

![](../../../../.gitbook/assets/image%20%28851%29.png)

입력 시퀀스와 출력 시퀀스의 길이, 도메인이 다를 수 있다. 그리고 하나의 모델로 이루어져있다. RNN 같은 경우는 3개의 단어가 들어가면 3번을 재귀적으로 도는데이 반해 Transformer는 100개든 1000개든 1번에 Encoding 할 수 있다.

* 물론 Generation 할 때는 한 단어씩 만지게 된다. Enocding 할 때를 말하는 것

우리가 이해해야 할 것은 다음과 같다

* N개의 단어가 어떻게 인코더에서 한번에 처리가 되는지
* 인코더와 디코더가 어떤 정보를 주고 받는지
* 디코더가 어떻게 제너레이션 할 수 있는지

이 세번째는 시간상의 이유로 수업에서 덜 다룰 것이다.

![](../../../../.gitbook/assets/image%20%28848%29.png)

새로운 부분은 `Self-attention`

주어진 3개의 벡터가 있다고 하자.

![](../../../../.gitbook/assets/image%20%28865%29.png)

이 때, 인코더는 벡터들을 다른 벡터들로 대응시켜준다.

![](../../../../.gitbook/assets/image%20%28843%29.png)

이 때 중요한 점은 단순히 x1 에서 z1으로 대응되는 것이 아니라 나머지 x2와 x3벡터를 고려해서 z1으로 대응된다는 것. 즉, i번째 x를 i번째 z로 바꿀 때에는 나머지 i-1 개의 x벡터를 고려하게 된다. 그래서 dependencies 가 있다고 한다.

반대로, 이후의 feed-forward는 그냥 네트워크를 통과하는 과정이기 때문에 dependencies가 없다.

![](../../../../.gitbook/assets/image%20%28868%29.png)

이번에는 두 시퀀스 "Thinking" 과 "Machine" 이 있다고 해보자. 이 때 `Self-Attention` 을 설명하고자 하는데, 다음과 같은 문장이 있다고 하자

> The animal didn't cross the street because **it** was too tired.

여기서 `It` 은 단순히 단어의 의미로 해석하면 안되고 단어가 문장속에서 다른 단어들과 어떠한 Interaction이 있는지 파악해야 한다.

![](../../../../.gitbook/assets/image%20%28862%29.png)

이 때, Transformer는 다른 단어들과의 관계성을 학습하게 되고 결론적으로 Animal과 높은 관계성을 가지고 있다고 학습하게 된다.

![](../../../../.gitbook/assets/image%20%28846%29.png)

Self-Attention 구조는 3가지 벡터를 만들어 내게 된다. 3가지 벡터가 있다는 것은 3개의 NN이 있다는 뜻과 같다.

* 3개의 벡터는 Queries, Keys, Values 이다.

그래서 x1벡터가 입력되는 이를 통해 q1, k1, v1 벡터를 만들게 되고 이 벡터들을 통해 x1 벡터를 다른 벡터로 바꿔주게 된다.

![](../../../../.gitbook/assets/image%20%28845%29.png)

Thinking과 Machine이라는 단어가 입력되었다면 각각의 단어에 대해 세가지 벡터를 만들게 된다. 이후에, Score라는 벡터를 만들게 되는데 내가 인코딩하고자 하는 벡터의 쿼리벡터와 자신을 포함한 나머지 단어벡터들의 키벡터를 내적한다.

* 이를 통해 해당 단어가 나머지 단어들과 얼마나 유사한지를 파악하게 된다
* 내적을 한 것은 해당 단어와 나머지 단어들 사이에서 얼마나 Interaction을 해야하는지 알아서 학습하게 하기 위함이다.
* 이것이 바로 `Attention` 에 해당한다.  내가 어떤 단어를 인코딩하고싶은데 어떤 나머지 단어들과 intraction이 많이 일어나야 하는지 파악해야함

![](../../../../.gitbook/assets/image%20%28850%29.png)

이후 이 점수를 8로 나눠주는데 이 8이라는 숫자는 Keys 벡터의 Dimension에 따라 결정된다. 여기서는 키벡터는 총 64개가 있고 이것을 square한 8로 나눠주게된다.

* 이는 score value가 너무 커지지 않도록 어떤 레인지 안에 두게 하는 효과가있다.

![](../../../../.gitbook/assets/image%20%28863%29.png)

이후 계산된 점수는 Softmax를 거치게되고 이후 Values 벡터와 곱해지게 된다.

* 결국 Value 벡터의 가중치를 구하는 과정은 Queries 벡터와 Keys 벡터의 내적을 통해 얻은 값을 구한다. 이 값을 Key의 Dimension의 square로 나눠주고 Softmax해서 나온 attention을 Values 벡터와 곱하게 된다.

여기서 중요한 점

* 쿼리벡터와 키 벡터는 내적을 해야하기 때문에 차원이 항상 같아야 한다.
* 하지만 밸류 벡터는 Weight sum만 하면 되기 때문에 차원이 달라도 된다.
* 단어 벡터 인코딩 된 벡터의 차원은 밸류 벡터의 차원과 동일해야 한다
  * Mutl-Attention에서는 또 달라진다.















