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

![](../../../../.gitbook/assets/image%20%28852%29.png)

입력 시퀀스와 출력 시퀀스의 길이, 도메인이 다를 수 있다. 그리고 하나의 모델로 이루어져있다. RNN 같은 경우는 3개의 단어가 들어가면 3번을 재귀적으로 도는데이 반해 Transformer는 100개든 1000개든 1번에 Encoding 할 수 있다.

* 물론 Generation 할 때는 한 단어씩 만지게 된다. Enocding 할 때를 말하는 것

우리가 이해해야 할 것은 다음과 같다

* N개의 단어가 어떻게 인코더에서 한번에 처리가 되는지
* 인코더와 디코더가 어떤 정보를 주고 받는지
* 디코더가 어떻게 제너레이션 할 수 있는지

이 세번째는 시간상의 이유로 수업에서 덜 다룰 것이다.

![](../../../../.gitbook/assets/image%20%28849%29.png)

새로운 부분은 `Self-attention`

주어진 3개의 벡터가 있다고 하자.

![](../../../../.gitbook/assets/image%20%28867%29.png)

이 때, 인코더는 벡터들을 다른 벡터들로 대응시켜준다.

![](../../../../.gitbook/assets/image%20%28843%29.png)

이 때 중요한 점은 단순히 x1 에서 z1으로 대응되는 것이 아니라 나머지 x2와 x3벡터를 고려해서 z1으로 대응된다는 것. 즉, i번째 x를 i번째 z로 바꿀 때에는 나머지 i-1 개의 x벡터를 고려하게 된다. 그래서 dependencies 가 있다고 한다.

반대로, 이후의 feed-forward는 그냥 네트워크를 통과하는 과정이기 때문에 dependencies가 없다.

![](../../../../.gitbook/assets/image%20%28870%29.png)

이번에는 두 시퀀스 "Thinking" 과 "Machine" 이 있다고 해보자. 이 때 `Self-Attention` 을 설명하고자 하는데, 다음과 같은 문장이 있다고 하자

> The animal didn't cross the street because **it** was too tired.

여기서 `It` 은 단순히 단어의 의미로 해석하면 안되고 단어가 문장속에서 다른 단어들과 어떠한 Interaction이 있는지 파악해야 한다.

![](../../../../.gitbook/assets/image%20%28864%29.png)

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

![](../../../../.gitbook/assets/image%20%28851%29.png)

이후 이 점수를 8로 나눠주는데 이 8이라는 숫자는 Keys 벡터의 Dimension에 따라 결정된다. 여기서는 키벡터는 총 64개가 있고 이것을 square한 8로 나눠주게된다.

* 이는 score value가 너무 커지지 않도록 어떤 레인지 안에 두게 하는 효과가있다.

![](../../../../.gitbook/assets/image%20%28865%29.png)

이후 계산된 점수는 Softmax를 거치게되고 이후 Values 벡터와 곱해지게 된다.

* 결국 Value 벡터의 가중치를 구하는 과정은 Queries 벡터와 Keys 벡터의 내적을 통해 얻은 값을 구한다. 이 값을 Key의 Dimension의 square로 나눠주고 Softmax해서 나온 attention을 Values 벡터와 곱하게 된다.

여기서 중요한 점

* 쿼리벡터와 키 벡터는 내적을 해야하기 때문에 차원이 항상 같아야 한다.
* 하지만 밸류 벡터는 Weight sum만 하면 되기 때문에 차원이 달라도 된다.
* 단어 벡터 인코딩 된 벡터의 차원은 밸류 벡터의 차원과 동일해야 한다
  * Mutl-Attention에서는 또 달라진다.

![](../../../../.gitbook/assets/image%20%28853%29.png)

행렬 연산으로 생각해보자. 단어가 두개이고 임베딩 차원은 4차원이어서 \(2, 4\)로 표현한다

* 원래 단어의 개수만큼 임베딩 벡터의 차원의 크기가 결정되는데 여기서는 그냥 4차원으로 표현한 것 같다

그리고 W를 곱해서 Q, K, V를 얻는다.

![](../../../../.gitbook/assets/image%20%28848%29.png)

그리고 쿼리와 키를 곱해서 소프트맥스를 거친뒤 밸류값을 구해서 Sum을 구한다.

코드로 구현하면 한두줄로 구현이 가능하다.

왜 트랜스포머가 잘될까?

기존의 CNN 같은 모델은 입력이 고정되면 출력도 고정된다. 그러나 Transformer 같은 경우는 입력이 고정되어도 주변 단어들에 따라 인코딩 값이 바뀔 수 있기 때문에 Flexible 한 모델이라고 볼 수 있다.

* 그래서 다양한 출력을 낼 수 있다.

Competition도 존재한다.

* n개의 단어가 있으면 n\*n개의 attention map이 있어야 한다.
  * RNN은 천개의 시퀀스가 있으면 천번 돌리면 된다.\(시간적 여유가 있다면 돌리는데 문제가 없다는 뜻\) 그러나 트랜스포머는 N개의 단어를 한번에 처리해야 하기 때문에 메모리를 많이 잡아 먹는다
  * 그대신 이런것을 극복하면 더 Flexble 하고 더 성능이 좋은 모델이 된다

![](../../../../.gitbook/assets/image%20%28893%29.png)

한 개의 단어에 대해서 여러개의 벡터를 만들 수 있으면 Multi-headed attention, MHA 라고 한다.

만약에 8개의 벡터를 만든다고 하자. 

![](../../../../.gitbook/assets/image%20%28888%29.png)

그런데 여기서, 입력 크기와 출력 크기가 동일해야 하는데, 출력이 8개이므로 출력 크기가 8배가 된다.

![](../../../../.gitbook/assets/image%20%28882%29.png)

그래서 출력 크기를 맞춰주기 위해 추가로 행렬을 곱해주게 된다.

* 예를 들어 총 \(10, 80\) 행렬이 완성된다면 \(80, 10\) 행렬을 곱해서 \(10, 10\) 행렬로 만든다.

그러나, 실제로 이렇게 구현되지는 않다. 자세한 건 코드실습에서.



우리가 N개의 단어를 sequential 하게 넣었지만 실제로 Transformer에는 순서정보가 반영되지 않는다. 그래서 positional encoding이 필요하게 된다. 만들어지는 방법은 사전에 pre-defined 된 방법을 가지고 만든다고 한다\(그래서 그 방법이 뭘까...ㅎ\)

![](../../../../.gitbook/assets/image%20%28883%29.png)

왼쪽은 예전에 쓰이던 positional encoding 방법이고 최근에는 오른쪽처럼 사용한다고 한다.

![](../../../../.gitbook/assets/image%20%28890%29.png)

어쨋든, attention이 되면 Z 벡터를 생성하게 되고 이 때 LayerNorm 과정을 거친다. 이게 무엇인지 설명하지 않으므로 내가 찾아봐야겠지...

[여기](https://yonghyuc.wordpress.com/2020/03/04/batch-norm-vs-layer-norm/)를 참고한 결과 LN은 다음과 같다 \(BN과의 차이를 통해 설명\)

> 먼저 BN은 “각 feature의 평균과 분산”을 구해서 batch에 있는 “각 feature 를 정규화” 한다.
>
> 반면 LN은 “각 input의 feature들에 대한 평균과 분산”을 구해서 batch에 있는 “각 input을 정규화” 한다.

![](../../../../.gitbook/assets/image%20%28887%29.png)

그림만 봐도 이해가 잘된다!



![](../../../../.gitbook/assets/image%20%28884%29.png)

인코더는 결국 디코더에게 키와 밸류를 보내게 된다.

* 출력하고자 하는 단어들에 대해 attention map을 만드려면 인풋에 해당하는 단어들의 키벡터와 밸류벡터가 필요하기 때문이다.

![](../../../../.gitbook/assets/image%20%28891%29.png)

인풋은 한번에 입력받지만 출력은 한 단어씩 디코더에 넣어서 출력하게 된다.



![](../../../../.gitbook/assets/image%20%28895%29.png)

디코더에서는 인코더와 달리 순차적으로 결과를 만들어내야 해서 self-attention을 변형하게된다. 바로 masking을 해주는 것. 인코더는 입력순서가 이미 정해져있기 때문에 decoder입장에서는 i번째 단어가 무엇인지 예측하기가 쉬워지기 때문에 이러한 마스킹을 해준다.

![](../../../../.gitbook/assets/image%20%28896%29.png)

`Encoder-Decoder Attention` 은 디코더가 쿼리 벡터를 제외하고는 키 벡터와 밸류 벡터는 인코더에서 생성된 것을 사용하겠다라는 인코더와 디코더의 상호연결성을 의미한다



추가적으로 동료와 이런 이야기를 나누었다.

> 근데 positional encoding이 과연 필수일까?

[Language Modeling with Deep Transformers](https://arxiv.org/pdf/1905.04226v2.pdf) 논문에서는 이렇게 말한다

![](../../../../.gitbook/assets/image%20%28885%29.png)

보통 언어모델에서는 positional encoding이 거의 필수이다. 사용하지 않아도 작동은 그럭저럭 한다. 근데 깊은 autoregressive model\(트랜스포머 모델의 한 종류\)에서는 데이터셋이 많고 모델이 깊다보니까 순서정보가 없는데도 데이터 자체에서 가중치에게 순서에 대한 정보를 제공해준다. 그래서 마지막에는 오히려 순서정보를 제공하지 않았더니 성능이 늘어났다.



## Vision Transformer

self-attention을 단어들의 sequence에만 사용하는 것이 아니라 이미지에도 사용하게 되었다.

![](../../../../.gitbook/assets/image%20%28886%29.png)

인코더만 사용하고, 인코더에서 나오는 벡터를 바로 분류모델에 사용하게된다.

차이점이라고 하면 언어는 문장들을 sequence하게 넣어준 것에 비해 이미지는 몇 개의 부분조각으로 나눈뒤 Linear Layer를 통과해가지고 하나의 입력인 것 처럼해서 넣는다.

* 물론 positional encoding이 들어간다.



### DALL-E

문장을 주면 이미지를 제공하는 것. 트랜스포머에 있는 디코더만 활용을 했다.

![](../../../../.gitbook/assets/image%20%28892%29.png)



















