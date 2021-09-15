---
description: 21.09.15
---

# \(09강\) Self-supervised Pre-training Models

### Recent Trends

트랜스포머 및 self-attention block은 범용적인 sequence encoder and decoder로써 최근 자연어처리에 많은 분야에서 좋은 성능을 내고있다. 심지어, 다른 분야에서도 활발하게 사용된다.

트랜스포머는 이러한 self-attention block을 6개만 쌓았는데, 최근의 발전동향은 \(모델 구조 자체의 변화는 없이\) 이를 점점 더 많이 쌓게되었다. 이를, 대규모 학습 데이터를 통해서 학습할 때 self-supervised learning framework를 통해 학습하고 transfer learning의 형태로 fine tuning해서 좋은 성능을 내고있다.

트랜스포머는 추천 시스템, 신약 개발, 영상 처리분야까지도 확장하고 있지만, 자연어 처리라는 분야에서는 &lt;sos&gt; 라는 토큰부터 **하나씩** 단어를 생성한다는 점에서 벗어나지 못한다는 한계가 있다.



## GPT-1

테슬라의 창업자 일론 머스크가 세운 비영리 연구기관인 Open AI에서 나온 모델이다. 최근에 GPT-2와 3까지 이어져 놀라운 성능을 보여주고 있다.

다양한 Special Token을 제안해서, 단순한 언어 모델 뿐만 아니라 다양한 언어 모델을 동시에 커버하는 통합된 모델을 제안했다는 것이 특징이다.

기본적으로 GPT-1의 모델구조와 학습방식에 대해 알아보자.

![](../../../.gitbook/assets/image%20%281149%29.png)

* 트랜스포머와 모양은 달라도, Text에 Position Embedding을 더한 값이 입력으로 들어가며, self-attention을 쌓은 층이 12개이다.
* 결과는 Text Prediction과 Text Classifier로 반환된다.
* Text prediction
  * 첫 단어부터 다음 단어를 순차적으로 예측한다.
* Text Classifer
  * 시퀀스에 대한 감정 분류등의 결과를 반환한다.

단순한 Task 뿐만 아니라, 문장 레벨 또는 다수의 문장이 존재하는 경우에도 모델이 손쉽게 변형 없이 활용될 수 있도록 학습의 framework를 제시했다.

![](../../../.gitbook/assets/image%20%281151%29.png)

* 우리가 알고 있는 토큰 이외에도 Delim이나 Extract라는 토큰을 사용하면서 여러가지 Task를 진행할 수 있다.

만약 모델을 통해서, 주제 분류를 해본다고 하자. \(ex 해당 doc이 정치, 경제, 사회, 스포츠 분야 중 어떤 분야인지\) 이 때는 이전에 사용하던 Text Prediction이나 Task Classifier는 떼버리고 그 전까지의 output인 word별 embedding 벡터들을 사용해서 추가적인 Task에 대한 레이어를 추가해서 학습한다.

* 이 때 마지막에 추가되는 layer는 random initialization이 되었기 때문에 추가적으로 학습을 하지만, 그 이전까지의 layer들은 학습이 이미 되어있는 상태이다. 그래서, 이전 layer들에게는 학습률을 매우 작게주면서 큰 변화가 일어나지 않도록 한다. 그래서 이전에 학습한 내용을 잘 담고있으면서 원하는 Task에는 잘 활용할 수 있도록 한다. 이는 pre-training과 fine-tuning을 동시에 적용하는 과정이다.
* 또, document의 class를 분류하는 문제에서는 데이터가 적을 수 밖에 없다보니, 이러한 데이터를 늘려도 좋지만

  또, 텍스트의 class를 구분하려면 labeling이 되어있어야 하는데, 이러한 데이터셋은 상대적으로 그 양이 작다. 그래서, 이전에 self-supervised 방식의 pre-trained 모델을 불러와서 fine tuning 하게 된다.

이렇게 pre trained 된 GPT-1 을 다양한 task에 fine tuning했을 때의 성능은 다음과 같다.

![](../../../.gitbook/assets/image%20%281168%29.png)

* 거의 대부분의 task에서 성능이 훨씬 좋은 모습을 보인다.



## BERT

버트 모델은 현재까지도 널리 쓰이는 Pre trained 모델이다. GPT와 마찬가지로 Language 모델로써 문장의 일부 단어를 맞추는 Task에 대해 Pretrained를 수행한 모델이다.

Self-supervised learning 방식으로 학습하는 Transformer 이전에 language 모델 중에서는 LSTM 기반의 인코더로 Pre-train 하는 접근 기법도 존재했는데, 이것이 바로 ELMo이다. 이러한 LSTM 기반의 인코더를 Transformer 기반의 인코더로 바꾸면서 여러 Task에 대해 좋은 성능을 가지는 모델들이 나왔는데, 그 중 하나가 BERT이다.

기존에 GPT는 전후 문맥을 파악하지 못하고 앞쪽 문맥만 보고 뒷 단어를 예측해야 한다는 한계점이 존재했다.

* 실제 사람의 대화에서든, 텍스트에서든 뒤쪽에서 문맥을 파악하는 일은 자주있는 일이다.





