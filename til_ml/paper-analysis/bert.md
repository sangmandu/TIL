---
description: 210926~
---

# BERT

## BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding



## Abstract

트랜스포머의 양방향 인코더 표현을 의미하는, 새로운 언어 모델 버트에 대해 소개하겠다. 최근 언어 모델과 달리 버트는 깊은 양방향 언어 표현 모델을 학습했다. 이는 모든 레이어의 이전 그리고 이후 context 정보를 공동으로 사용한 라벨링 되지 않은 텍스트를 기반으로 한다. 결과적으로 사전학습된 버트모델은 질의응답이나 언어 추론등의 광범위한 task들에 대해서 실질적인 특정 task 기반 모델 구조를 위한 변형없이 단 하나의 추가적인 output layer만을 사용해서 fine tuning 했더니 신기록을 세울 수 있었다.

버트는, 개념적으로는 간단하지만 실험적으로\(=경험적으로\) 매우 강력하다. 11개의 NLP task에 있어서 신기록을 갱신했고 이 때의 GLUE 점수는 최고 점수에서 7.7% 차이나는 80.5%이고 MultiNLI 정확도는 최고 점수에서 4.6% 차이나는 86.7%를 달성했다. 또 SQuAD v1.1 질의응답에서는 1.5점 차이나는 93.2의 F1 Score를 달성했고 SQuAD v2.0에 대해서는 5.1점 차이나는 83.1의 F1 Score를 달성했다.



## 1 Introduction

여러 NLP task를 개선하는데 효율적인 언어 모델 사전학습 방법이 여태까지 소개가 되었다 \(Dai and Le, 2015; Peters et al., 2018a; Radford et al., 2018; Howard and Ruder, 2018\). 또, 자연어 추론과 같은 문장 레벨에 대해서는 이러한 논문\(Bowman et al., 2015; Williams et al., 2018\)에서 소개되었고 전체적으로 문장을 분석하면서 얻어지는 문장 간의 관계를 예측하는 것이 목표인 의역은 이 논문\(Dolan and Brockett, 2005\)에서 소개되었다. 또, fine-grained 방식으로 토큰 레벨에서 수행하는 개체명 인식이나 질의응답과 같은 task는 이 논문\(\(Tjong Kim Sang and De Meulder, 2003; Rajpurkar et al., 2016\)에서 소개되었다.

* NER, Named Entity Recognition, 개체명 인식은 이름을 의미하는 단어가 어떤 유형인지를 인식하는 task이다. 예를 들어 '상민이는 2021년에도 잘생겼다'에서 상민:사람, 2021년:시간 으로 분류한다.
* fine-grained란 세부적으로 분류하는 작업을 의미하며, 분석해야 하는 task들이 비슷한 특징을 가졌을 때 사용한다. 대비되는 의미로는 coarse-grained가 있다. 

down stream task에 대해서 언어 표현 모델을 사전학습하는 두 가지 전략이 있다. 바로 feature-based 와 fine-tuning 이다. feature-based 접근법은 엘모에서 사용되었으며 특정 task에 기반한 구조를 사용한다. 이 구조는 모델을 pre-train 할 때 추가적인 특징들을 포함하게된다. fine-tuning 접근법은 트랜스포머에서 사용되었으며 기본적인 특정 파라미터만을 사용하고, 이후에 down stream task에 대해서 간단하게 fine tuning 하는 과정에서 모든 파라미터를 학습하게 된다. 두가지 접근법은 일반적인 언어 표현을 학습하기 위해 단방향 언어 모델을 사용하는 pre training이 이루어지는 과정에서는 동일한 목적을 지닌다.

* down stream task란 구체적으로 해결하고자 하는 문제들을 의미한다. 자세히 이야기해보자. 최근 자연어 처리분야에서는 pre-trained 된 모델을 fine-tuning 하는 방식을 사용해서 구체적인 문제를 해결하는데 이 때 이러한 방식으로 해결하고자 하는 문제들을 down stream task라고 지칭하는 것

우리는 이러한 테크닉들이 사전 학습으로 표현할 수 있는 능력을 제한한다고 생각한다. 특히 fine-tuning에서는 더더욱 제한된다. 주된 한계점은 표준 언어 모델들이 단방향적이라는 것이고 이는 사전 학습될 때 모델의 예측에 제한을 주게된다. 예를 들어 OpenAI의 GPT 모델같은 경우 개발자들은 왼쪽에서 오른쪽의 방향으로 설계된 구조를 사용했고 이는 트랜스포머처럼 모든 토큰들이  왼쪽에서 오른쪽으로만 접근가능하게 되었다. 문장 레벨 task에서 이러한 제한이 남아있으면 최선의 해답을 내놓을 수 없고 질의 응답과 같은 토큰 레벨의 task에 적용되는 fine tuning에서 안좋은 영향을 줄 수 있다. 이것이 양방향적으로 context 정보를 포함해야 하는 중요한 이유이다.

이 논문에서 우리는 BERT라는 fine tuning에 기반한 방법을 제시한다. BERT는 양방향 인코더 표현 from 트랜스포머 의 약어이다. 버트는 이러한 단방향의 masked 모델이 가지는 한계점을 \(Taylor, 1953\)의 Cloze task에서 영감을 받아 사용한 MLM, masked language model 방법을 사용해서 완화시킨다. MLM은 주어진 입력에 대해 무작위로 토큰을 정해서 마스킹한다. 이는 주변 문맥을 통해 마스킹된 단어를 예측하려는 목적성을 가지고있다. 기존 왼쪽에서 오른쪽의 방향만을 가지고 pre train하는 것과는 달리 MLM의 목적은 양쪽에서 얻는 문맥적 특징을 섞는 것을 가능하게 하게한다. 그리고 이를 위해 깊은 양방향 트랜스포머를 사용한다. 게다가 MLM은 "다음 문장 예측" task를 수행할 수 있도록 전후 문장쌍의 특징\(=representations\)을 학습했다. 이 논문의 협업자는 다음과 같다.

* 여기서 fuse는 blend의 의미로 사용되었다고 생각했다.

언어 특징에 대해 양방향으로 학습하는 것이 중요함을 증명했다. 단방향을 사용하는 Radford et al. \(2018\)와 달리 버트는 MLM을 사용해서 깊은 양방향 표현이 가능하다. 이는 왼쪽에서 오른쪽 또는 오른쪽에서 왼쪽의 방향을 가진 LM, Language Model들이 독립적으로 얕게 연결된 구조를 사용하는 Peters et al. \(2018a\)와는 대조적인 결론이다.

특정 task를  해결하기 위해서 이 task를 위한 구조를 어느정도 만져야 하는데,\(=heavily-engineered\) 버트는 사전학습된 특징이 이런 필요성을 줄여줄 수 있는 것을 보여준다. 문장 레벨 또는 토큰 레벨에 대한 많은 task에서 최고의 성능을 내고있는 모델들을 fintuning 한 버트는 여러 task를 위해 특정 구조를 형성하는 것보다 더 좋은 성능을 낸다.

버트는 11개의 NLP task에 신기록을 세웠다. 이 코드와 사전학습된 모델은 https://github.com/ google-research/bert.에서 볼 수 있다.



## 2 Related Work

매우 예전부터 일반적인 언어 모델을 사전학습하기 시작했는데 이렇게 흔히 사용되는 접근법에 대해서 간단하게 다뤄보자

### 2.1 Unsupervised Feature-based Approaches

여러  task에서 작동하도록 단어들의 특징을 학습하는 것은 최근 수십년동안 활발하게 연구된 분야이다. 신경망을 사용하지 않고도 \(Brown et al., 1992; Ando and Zhang, 2005; Blitzer et al., 2006\)에서. 그리고 신경망을 사용해 \(Mikolov et al., 2013; Pennington et al., 2014\)에서 연구되었다. 사전 학습된 워드 임베딩은 통합된\(=거의 임베딩을 모든 언어모델이 사용한다는 뜻\) NLP 시스템의 현대적인 부분 중 하나이며 임베딩을 학습하는데에 두드러진 발전을 가져왔다. 임베딩 벡터를 사전학습하기 위해 좌우방향의 언어모델들이 사용되었고 \(Mnih and Hinton, 2009\) 올바르지 않은 단어로부터 올바른 단어를 결정하는 목적을 두고도 좌우방향의 언어모델이 사용되었다. \(Mikolov et al., 2013\).

이러한 접근들은 문장 임베딩\(Kiros et al., 2015; Logeswaran and Lee, 2018\)이나 문단 임베딩\(Le and Mikolov, 2014\)과 같은 점점 세밀한 task들에 일반화되어왔다. 문장의 특징을 학습하기위해서 이전의 연구들은 다음 문장으로 올 후보들을 선정하거나\(Jernite et al., 2017; Logeswaran and Lee, 2018\), 좌우 방향의 언어모델로 이전 문장의 특징을 통해 다음 문장을 생성했으며\(Kiros et al., 2015\), 오토 인코더에서 이러한 목적을 실현\(=derived, 파생하다\)하려고 했다. 

ELMO와 ELMO의 조상모델들은 서로 다른차원으로 전통적인 워드 임베딩을 생성하고자 연구했다. 이들은 문맥-감각적 특징을 좌우방향 또는 우좌방향의 모델을 통해 얻으려고했다. 각 토큰의 문맥적인 특징은 좌우 또는 우좌 방향의 특징의 연결로 결정했다. 이러한 문맥적인 워드 임베딩을 기존의 특정 task에만 적용되는 구조들과 통합했을 때 ELMO는 특정 주요 NLP task 에서 최고성적을 얻었다. 질의 응답, 감정 분석, 개체명 인식등이 이에 속한다. Melamud et al. \(2016\) 논문은 문맥적인 특징을 학습할 때 양쪽의 문맥으로 부터 한 단어를 예상하는 LSTMs을 통해 학습하는 방법을 제안했다. ELMO도 이와 비슷했지만 feature-based하지만 깊은 양방향은 아니었다. Fedus et al. \(2018\)는 cloze task가 텍스트 생성 모델의 일반화 성능을 개선할 수 있다고 했다.

* 이번 문단에서는 논문 언급을 하지 않았다.
* cloze task는 1953년 Taylor 논문에서 언급된 것으로 하나 또는 여러개의 단어가 한 문장에서 제거되고 학생이 이 제거된 단어를 예측하는 문제였다.



### 2.2 Unsupervised Fine-tuning Approaches

feature-based 방법을 적용한 첫번째 연구는 워드 임베딩만을 사전학습했다. \(Collobert and Weston, 2008\).

매우 최근에는 문장 또는 문서를 인코딩할 때 문맥적인 특징을 가진 토큰들을 생성하는데 이것들을 라벨링되지 않은 텍스트에서 학습하고 downstream task에서 지도학습으로 fine tuned 한다 \(Dai and Le, 2015; Howard and Ruder, 2018; Radford et al., 2018\). 이런 접근법의 장점은 pre train할 때 적은 파라미터 수로도 가능하다는 것. 이런 장점때문에 GPT는 적은 파라미터를 가지고도 GLUE 데이터셋의 많은 task에서 최고 성적을 낼 수 있었다. 좌우 방향 언어모델이나 오토 인코더는 다음과 같은 모델들을 pre training 하기위해 사용되었다 \(Howard and Ruder, 2018; Radford et al., 2018; Dai and Le, 2015\).

* objective를 어떻게 해석하면 좋을까가 고민이다. 사실 직역하면 목적 정도이겠지만 어떠한 -로 할 수 있는 것 또는 -의 기능 정도로 자연스럽게 생각하려고 해도 어려운 부분 ㅠ\_ㅠ



### 2.3 Transfer Learning from Supervised Data

자연어 추론이나 기계 번역과 같이 지도학습 task의 방대한 데이터셋으로부터 효율적인 번역 task를 위한 연구도 있었다. 컴퓨터 비전 연구는 사전학습된 대형 모델의 전이 학습의 중요성을 증명해왔다. 이러한 증명은 이미지넷을 pre train하고 이를 fine tuning하는 효율적인 방법에서 증명되었다.



## 3 BERT 

버트와 버트의 자세한 구현에 대해 소개하겠다. 큰 구조는 두 가지 과정으로 이루어져있다. pre-training과 fine-tuning. 사전 학습시에는 여러가지 tasks에서 작동할 수 있도록 모델은 unlabeled 데이터를 학습한다. fint tuning시에는 버트는 제일먼저 사전학습된 파라미터들로 초기화하고 모든 파라미터들은 downstream task에 맞추어 미세조정하게된다. 각각의 task들은 똑같은 파라미터들도 사전학습되어 초기화되었을지라도 각각의 task들에 대해서 개별적으로 처리된다. 예를 들어 Figure 1에서는 질의응답을 task의 한 예로 들었는데 여기서 버트의 작동 예시를 보여준다.

버트 특유의 특징은 서로 다른 task들에 대해 하나의 모델 구조를 사용한다는 것이다. 물론 pre-trained 구조와 fine-tuning을 거친 구조와는 최소의 차이는 있다.

![](../../.gitbook/assets/image%20%281215%29.png)

Figure 1 : 버트의 전체적인 사전학습과 미세조정 과정이다. output layer를 제외하고는 사전학습과 미세조정에서 동일한 구조를 사용한다. 서로 다른 task에 똑같은 사전학습 모델의 파라미터로 초기화한다. fine-tuning 시에는 모든 파라미터가 fine tuned 된다. CLS 토큰은 특별한 의미를 지니는데, 모든 input 문장 앞에 추가된다. 그리고 SEP 토큰은 예를 들면 질문과 답변을 구분해주는 것처럼 특별한 구분자로 사용된다.

#### Model Architecture

버트의 모델 구조는 다중 레이어의 양방향 트랜스포머 인코더를 사용한다. 이 인코더는 Vaswani et al. \(2017\) 에서 구현된 모델을 기반으로 했고 이는 tensor2tensor library에 공개되어있다. 트랜스포머의 사용이 대세가 되었고 우리가 사용한 트랜스포머도 원래의 것과 거의 동일하기 때문에 이러한 모델의 전반적인 배경과 구조를 대부분 생략할 것이며 독자들에게는 잘 정리된 Vaswani et al. \(2017\)를 읽기를 권한다. 

이번 논문에서 사용할 용어를 설명하려고 한다. 레이어의 수는 L로, 히든 사이즈는 H로, self attention head의 수는 A로 나타낸다. 우리는 두가지 모델 사이즈에 대해 다룰 것인데, 하나는 BERT-BASE \(L=12, H=768, A=12, Param=100M\) 이고 하나는 BERT-LARGE \(L=24, H=1024, A=16, Param=340M\) 이다.

BERT-BASE는 GPT와 비교하고자 하는 목적으로 동일한 크기의 모델로 생성했다. 그렇지만 버트는 양방향 self attention을 사용하고 GPT는 왼쪽에서만 접근이 가능한 제한적인 self attention을 사용하는 차이가 있다.

#### Input/Output Representations

버트가 다양한 task를 해결하도록 input으로 하나의 문장 또는 한 쌍의 문장을 입력받는다. 연구 내내 문장이라는 개념이 등장하는데 이는 단순히 실제 언어적인 문장을 의미한다기 보다는 연속적인 텍스트의 임의의 부분으로 이해하면 된다\(=연속된 시퀀스라는 형태적인 부분으로 이해하라는 뜻 같음\) 이러한 한 개의 또는 한 쌍의 시퀀스에서 얻은 token을 버트에 입력하게 된다. 

우리는 3만개의 토큰을 가진 WordPiece 임베딩을 사용했다. 이 때 각 문장의 첫번째 토큰은 CLS라는 특별한 토큰이 위치한다. 마지막 히든 스테이트에서 이 토큰은 분류 태스크를 위한 문장 집계 특징으로 사용된다. 한 쌍의 문장은 한개의 문장으로 묶여 있는데 이를 구별하는 방법은 두가지이다. 첫번째는 두 문장 사이에 SEP 토큰을 추가하는 것. 두번째는 토큰에다가 A 문장의 토큰인지 B 문장의 토큰인지에 대한 정보를 추가하는 것이다. Figure 1 에서 E는 input embedding, CLS 토큰의 final hidden vector 를 C로 나타냈으며 i번째 input token의 final hidden vector는 Ti 로 나타냈다.

* WordPiece는 underbar를 이용해서 word를 subword로 만들어 tokenize하는 분류기이다.

주어진 토큰과 segment, position embeddings를 합산해서 입력 representation을 구성할 수 있다. 이 구성에 대한 시각적인 자료는 Figure 2에서 볼 수 있다.



### 3.1 Pre-training BERT

ELMO와 GPT-1과 달리 우리는 좌우 또는 우좌방향의 모델로 버트를 학습시키지 않았다. 대신에 두 개의 비지도학습 task를 통해 학습했다.

#### Task \#1: Masked LM

직관적으로 깊은 양방향 모델은 단향방 모델이나 이러한 모델들을 얕게 연결한 것보다 더 성능이 좋다는 것은 합리적이다. 불행하게도 조건부 표준 언어 모델은 단방향으로만 학습이되었다. 반면에 양방향은 각각의 단어들이 자기자신을 간접적으로만 참조할 수 있게했고 모델은 타겟 단어를 다층 구조의 context를 이용하여 좀 더 구체적으로 예측할 수 있게된다. 

깊은 양방향 표현을 학습하기 위해서 우리는 간단하게 몇몇 입력 토큰들을 무작위로 마스킹하고 이 마스킹된 토큰을 예측한다. 이러한 과정을 masked LM, MLM이라고한다. 이 개념은 \(Taylor, 1953\)에 언급된 Cloze task를 참고했다. 여기서 mask token에 해당하는 final hidden 벡터는 output sofrmax에 입력된다. 실험결과 15%의 토큰 마스킹 비율을를 적용하는 것이 가장 좋았다. denosing auto-encoder와는 달리 전체적으로 input을 재구성하는 것보다는 masking된 단어들을 예측했다.

이와 같이 양방향 모델을 구성했지만 \[MASK\] 토큰이 fine tuning 시에는 존재하지 않기 때문에 pre training과 fine tuning 사이에 불합이 발생한다. 이러한 차이를 줄이기 위해 masked word를 늘 \[MASK\] 토큰으로 대체하지는 않는다. 학습 데이터에서 15%의 비율로 무작위로 예측에 사용될 토큰으로 지정된다. 이 때 i번째 토큰이 정해지면 이 토큰중 80%는 \[MASK\] 토큰으로, 10%는  random token으로, 10%는 변경하지 않는다. 그 이후 cross entropy loss를 가지고 원래 토큰을 예측하기 위해 i번째 토큰의 마지막 히든 벡터 T가 사용된다. 우리는 이 과정의 변화를 C.2 에서 비교할 것이다.



#### Task \#2: Next Sentece Prediction \(NSP\)

질의 응답이나 자연어 추론과 같은 중요한 task들은 두 개의 문장 사이의 관계를 파악하는 것에 기반을 둔다. 이러한 문장은 언어 모델링에 의해 직접적으로 얻어지지 않는다. 문장 관계를 파악하기 위한 모델을 학습하기 위해 우리는 단일 언어\(아마 여러 나라의 언어가 섞이지 않은 이라는 뜻인 듯\)로 이루어진 말뭉치에서 대충\(=trivially\) 만들어낸 문장을 구분하는 다음 문장 예측 데이터를 학습했다. 특히 A와 B문장이 선택될 때 50%의 확률로 B는 정말로 A의 뒷문장이거나 또는 아무렇게나 생성된 문장이다. Figure 1에서 볼 수 있듯이 C\(=CLS 토큰\)는 다음 문장을 예측하는 NSP에 사용된다. 이렇게 간단한 구조에도 불과하고 QA와 NLI에서 엄청난 효율을 보였다. 이는 5.1에서 확인할 수 있다. NSP task는 Jernite et al.\(2017\)과, Logeswaran and Lee \(2018\)에서 사용된 특징 학습과 \(=representation-learning objectives\) 매우 관련이 있다. 그러나 이전의 연구에서 버트는 end-task 모델 파라미터를 초기화하기위해 많은 임베딩 중 문장 임베딩만 down-stream task의 파라미터로 사용되었다.

![](../../.gitbook/assets/image%20%281217%29.png)

#### Pre-training data

사전학습 과정의 대부분은 기존의 언어 모델 사전학습 절차를 따른다. 800M 크기의 BooksCorpus와 2500M 크기의 English Wikipedia의 말뭉치를 사전학습했다. 위키피디아에서는 텍스트 구절만 뽑아왔고 그 외의 리스트나 표, 헤더는 무시했다. 문서단위의 말뭉치를 사용하는 것은 Billion Word Benchmark에 있는 문장 단위의 말뭉치를 뽑는 것보다 시퀀스가 더 연속적\(더 길기\)이기 때문에 더 중요하다.



### 3.2 Fine-tuning BERT 

트랜스포머의 self attention 메커니즘은 버트가  input과 output을 적절히 바꾸게 하면서 여러 down-stream task를 다룰 수 있도록 했기 때문에 fine tuniing은 어렵지 않았다\(=straightforward\). 이 task들이 single text의 task인지 text paris의 task인지는 상관없다. text pairs로 해결해야 하는 task들에서는 일반적으로 Parikh et al. \(2016\)나 Seo et al. \(2017\)처럼 양방향 cross attention을 적용하기 직전에 text pair를 독립적으로 인코딩한다는 것이다. 그래서 버트는 두 단계\(text pair를 독립적으로 인코딩 하는 것과 양방항 cross attention을 적용하는 것\)를 통합해서 self attention 메커니즘을 사용하려고 했다. 두 문장간에 얻어지는 양방향 cross attention으로 연결된 두 문장을 효율적으로 인코딩하려고 했다.

각 task 마다 우리는 간단하게 특정 input과 output을 버트로 입력해주기만 하면 되었고 알아서 처음부터 끝까지 모든 파라미터가 fine tuning 되었다. 사전 학습할 때 입력되는 문장 A와 B는 다음 중 하나의 특징을 가질 수 있다.\(=유사하다와는 의미를 의역\) 1\) 문단에서의 두 문장 쌍 2\) 함의에서 가설과 전제 3\) 질의응답에서 질문 쌍 4\) 텍스트 분류나 문장 태깅에서의 degenerate text-0 pair 출력에서 token의 특징은 문장 태깅이나 질의응답 같은 token level의 task를 처리하는 output layer로 입력된다. 그리고 CLS 토큰은 함의나 감정 분석같은 분류를 위한 output layer로 입력된다.

* 4번 같은 경우는 기존 text-text에서 single text 체제로 변환하면서 text-공집합 꼴이 되었고 이러한 모양을 퇴화했다\(=degenerate\)는 의미로 언급한 것 같다.

사전학습과 비교하면 fine tuning은 비교적 비용이 든다. 이 논문에있는 모든 결과는 TPU로는 많으면 1시간, GPU로는 몇시간이 걸려서 동일한 사전 학습 모델을 재구성할 수 있다. 4장에서는 구체적인 task들에 대한 세부사항을 설명한다. 자세한 내용은 A.5 를 참조하자.



## 4 Experiment

11가지 NLP task에 대한 버트의 fine tuning 결과들을 소개한다.

### 4.1 GLUE

The General Language Understanding Evaluation, GLUE benchmark는 다양한 자연어 인지 task 모음집이다. GLUE dataset의 세부사항은 부록 B.1에 있다.

GLUE를 fine tune 하기 위해서 3장에서 말한것처럼 input sequence\(single이든 pair든\) 를 사용할 것이고 마지막 첫번째 input token CLS에 해당하는 hidden vector C를 집합 표현체로 를 사용합니다. fine tuning을 할 때 분류 레이어에 사용되는 K \* H 크기의 W 파라미터가 등장한다. 이 때 K는 라벨의 수를 의미한다. 우리는 C와 W의 곱을 log-softmax를 해서 loss를 계산한다.

* 집합 표현체라는 의미는 문장에 대한 전체적인 정보를 담고있기 때문에 이런 비유\(?\)를 사용했다. 

모든 GLUE tasks에 대해서 전체 데이터를 가지고 3epoch의 fine tune을 거쳤고 이 때의 배치는 32이다. 각각의 task에서 최적의 학습률을 선택했다. \(5e-5, 4e-5, 3e-5, 2e-5 중에서 사용했다.\) 그리고 BERT-LARGE 모델을 실험하다보니 때때로 적은 데이터셋으로 fine tune 하는건 안좋을 수 있다는 것을 알았고 그래서 무작위로 여러번 fine tune하고 이 중에 제일 성능이 좋은 모델을 골랐다. 여러번 fine tuning 할 때는 사전 학습된 check point는 동일하게 사용했지만 데이터를 섞거나 분류기 파라미터는 다르게 사용했다.

Table 1에 결과가 있다. BERT 베이스나 라지는 모든 task에서 기존 sota 모델들보다 충분히 여유있게 4.5%와 7.0%라는 각각의 평균 정확도를 향상시켰다. 버트 베이스는 모델 구조적인 관점에서 attention mask를 제외하고는 GPT와 거의 동일하다. 가장 크기가 크고 넓은 GLUE task인 MNLI에 대해서도 4.6%의 정확도를 향상시켰다. 공식적으로 GLUE 리더보드에 버트 LARGE는 80.5점을 받았다. 그에 비해 GPT는 우리가 마지막으로 확인한 바 72.8점을 기록했다.

버트 LARGE는 두드러지게 버트 BASE보다 모든 태스크에서 아주 작은 학습 데이터만으로도 훨씬 좋은 성능을 냈다. 모델의 사이즈에 대한 이야기는 5.2에서 많이 해보자.

![](../../.gitbook/assets/image%20%281243%29.png)

#### TABLE 1

서버에 기록된 GLUE 테스트 결과이다. 각각의 태스크 밑에 있는 수는 학습 데이터의 수이다. 평균 점수는 공식적인 GLUE 점수랑은 좀 다른데 우리가 WNLI set에 대한 점수는 제외했기 때문이다. 버트와 GPT는 하나의 모델로 하나의 task를 처리할 수 있다\(?\) F1점수는 QQP와 MRPC에서, Spearman Correlation은 STS-B에서, 정확도는 다른 태스크에서의 표준 척도로 정해진 점수이다. 버트를 하나의 요소로 사용하는 모델들의 성능은 제외했다.

* WNLI set 점수를 제외한 8번 각주를 보면 train, valid, test set에 분포가 너무나도 달라서 성능을 측정하기가 애매한 부분이 있어서 poor score를 얻게되는 현상이 있다고 나는 이해했음. 그치만 19년에 이미 90점 이상의 점수를 달성하긴 했음
* Spearman Correlation은 서열상관분석이라 하는, 두 변수간의 상관관계를 분석하는 기법이다. 나도 잘 몰라!!!



> 이 부분은 논문의 해석과는 크게 관련이 없을 수 있으나 GLUE의 Task들을 설명하는 좋은 링크가 있어 추가한다. [참고링크](https://thejb.ai/bert/)



### 4.2 SQuAD v1.1

스탠포드 대학에서 만든 질의응답 데이터셋, SQuAD 1.1 버전은 10만개의 크라우드소싱 질의응답쌍이다. 위키피디아에서 지문\(=passage\)과 질문과 답이 주어지면 지문속에서 답에 해당하는 범위를 예상하는 것이 task이다.

Figure 1의 질의응답 태스크에서는 질문과 지문이 하나의 시퀀스로 이루어져있으며 질문은 A 임베딩으로, 지문은 B 임베딩으로 이루어져있다. 여기서는 파인튜닝시에 사용되는 문장의 시작과 끝을 의미하는 S와 E벡터에 대해 이야기 할것이다. 어떤 단어가 정답에 해당하는 부분의 시작 단어일 가능성은 T와 S를 내적하고 softmax를 거친 값으로 계산된다. \(여기서 T는 i번째 token의 마지막 hideen vector에서 얻어지는 output 값이다\)

![](../../.gitbook/assets/image%20%281252%29.png)

정답에 해당하는 부분의 마지막 단어를 구할 때도 유사한 공식이 사용된다. 정답으로 예상되는 후보들의 점수는 S·Ti + E·Tj 로 구해진다. 그리고 가장 큰 점수가 예측으로 사용된다.  이 때 j &gt;= i 여야한다. 학습할 때의 목적함수는 올바른 start와 end 자리의 로그 우도의 합이다. 32의 배치사이즈, 5e-5의 학습률로 3 epochs의 fine tune을 거쳤다.

표 2에서는 기존에 높은 성적을 지니던 모델들이 있는 리더보드 성적을 보여주지만, SQuAD 리더보드에 올라있는 높은 성적들을 지닌 모델들에 대해서 우리가 이용할 수 있을만한 최근 설명이  없었고 해당 모델들은 아무 데이터나 가지고 학습을 할 수 있었다. 그래서 우리는 SQuAD로 fine tuning하기 전에 TriviaQA로 먼저 finetuing을 했고 그러면서 적당히 data augmentation을 사용했다.

우리 모델은 기존 최고 성적의 모델보다 앙상블에서는 F1 점수가 1.5점이 높았고 단일 모델에서는 1.3점이 높았다. 사실 우리 버트모델은 현존하는 최고의 모델을 앙상블한것보다 뛰어나다. 그리고 사실 TriviaQA 가지고 fine tuning하지 않아도 F1 점수는 0.1에서 0.4점밖에 차이나지 않기 때문에 꽤 큰 격차로 높은 성능을 보인다고 말할 수 있다.

* TriviaQA는 워싱턴 학교에서 만든 QA 데이터셋이다.

![](../../.gitbook/assets/image%20%281230%29.png)

#### Table 2

SQuAD 1.1 버전의 결과이다. 버트의 앙상블 버전은 서로 다른 fine tuning 파라미터\(=seeds\)와 체크포인트를 사용하는 7개의 모델로 앙상블했다.

![](../../.gitbook/assets/image%20%281237%29.png)

#### Table 3

SQuAD 2.0 결과이다. 버트가 장착되어 있는 모델들의 비교는 하지 않았다.



### 4.3 SQuAD v2.0

SQuAD 2.0은 1.1버전에서 더 나아가 정답이 없을 수도 있는 가능성을 추가하면서 좀 더 현실적인 문제를 해결하도록 확장했다.

스쿼드 1.1를 사용하던 버트 모델을 확장하기 위해서 간단한 방법을 썼는데, 정답이 없을 경우에는 start token과 end token의 위치가 모두 CLS 토큰 위로 있도록 하는 방법이다. 확률을 표현할 수 있는 범위가 CLS 토큰까지 start token과 end token이 있을 수 있도록 확장되었이다. 예측할 때는 일단 no-answer 점수인지를 확인한다. 이는 no answer 점수인 S\_null = S·C + E·C과  절대 no answer\(=best non-null\) 이 아닌 점수 $$ s_{\hat{i},j}$$과 비교한다. 이 때 이 non null 점수는 j&gt;=i 면서 S·Ti + E·Tj 가 최대가 되는 점수이다. 그래서 이 answer가 있는지 없는지 확인할 때는 sˆi,j &gt; snull + τ인지를 확인한다. 이 때 타우\(=τ\)는 F1 점수가 가장 높도록 하는, 실험적인 방법으로 결정된다. 스쿼드 2 버전을 쓸 때는 TriviaQA data를 사용하지 않았다. 48의 배치사이즈, 5e-5의 학습률 2 epochs로 fine tune 했다.

이전에 높은 성적을 거둔 리더보드와 논문들과 비교한 결과는 표3에 있다. 이전 모델들과 F1 스코어를 5.1점 벌렸다.

![](../../.gitbook/assets/image%20%281228%29.png)

#### Table 4

SWAG 학습 및 평가 정확도이다. SWAG 논문에는 100개의 샘플에 대한 인간의 예측력도 측정했다.



### 4.4 SWAG

The Situations With Adversarial Generations, SWAG 데이터셋은 11.3만개의 상식 추론에 기반을 둔 데이터를 평가하는 문장쌍으로 이루어져 있다. 문장이 주어지면 4개의 보기중에 그럴 듯한\(=plausible\) 답을 고르는게 TASK이다.

SWAG 데이터셋으로 fine tuning 할 때 4개의 input sequence를 입력해줘야 한다. 이 sequence는 주어진 문장과 주어진 문장 뒤로 이어질 수 있는 선택지의 연결로 구성된다. \(선택지가 4개이므로 총 4개의 input sequence가 나옴\) \(이후 이 4개의 input이 BERT로 입력되어 output을 얻고 이 output의\) CLS 토큰을 가지고 각각의 시퀀스의 점수를 구할 수 있다. 이 점수는 task-specific한 파라미터 V와의 내적해서 구해지며 각각의 점수들은 softmax layer를 거치게 된다.

* 여기서 task-specific하다는 것은 nlp task가 여러개이고 각각의 task마다 CLS토큰과 곱해지는 벡터가 다르다는 것을 의미
* 여기 문장 구조가 독해하기가 어려워서 한번 다루고 넘어가겠음

> 원문
>
> The only task-specific parameters / introduced /  is a vector / whose dot product with the \[CLS\] token representation C / denotes a score / for each choice / which is normalized with a softmax layer

* the only task-specific : SWAG에서 말한 task는 4지선다형 task이고 이 하나밖에 없기 때문에 이러한 표현 사용
* introduced : 지금 계속 언급하고 있는
* is a vector ~ : a vector is 구문이 도치된 문장, 이는 vector를 수식하는 whose절이 길기 때문에 도치한 것임. 또한, a vector과 parameters는 단복수가 맞지 않은 것처럼 보일 수는 있지만 잘 생각해보면 벡터 자체가 이미 복수 집합체임.
* whose dot product ~ : task-specific한 파라미터는 CLS 토큰과 내적을 함 그리고 이는 점수를 의미함
* for each choice : 4지선다에 대해 각각의 sequence를 choice로 표현. 
* which : choice를 꾸미는 것 같긴 하고 또, 의미적으로도 크게 어색하지는 않지만 score를 꾸미는 구로 보임. 각각의 sequence보다는 각각의 sequence의 점수가 softmax layer를 통해 정규화되기 때문. 

모델은 3 epochs, 2e-5 lr, 16 bs로 fine tune 했다. 결과는 Table 4에 있다. 버트 라지모델은 ESIM+ELMO 모델보다 27.1%, GPT 모델보다 8.3% 성능을 압도했다.



## 5 Ablation Studies

이번 장에서는 버트의 관계적 중요성을 잘 이해하기 위해 버트의 여러 분할버전에 대해 ablation 실험을 한다. 추가적인 연구는 부록 C를 참고해라!

* ablation 이란 모델이나 알고리즘을 구성하는 다양한 구성요소\(component\) 중 어떠한 “feature”를 제거할 때, 성능\(performance\)에 어떠한 영향을 미치는지 파악하는 방법을 말한다.
* 실제로 사전적 의미는 일정부분을 제거한다는 뜻이다.



![](../../.gitbook/assets/image%20%281245%29.png)

#### Table 5

pretraining task의 Ablation은 BERT-BASE 구조에서 실험했다. "No NSP"는 next sentence prediction task가 없이 학습된다. "LTR & No NSP"는 "No NSP" 항목에다가 bidirectional이 아닌 GPT처럼 left to right LM으로 attention 방식이 바뀐 항목이다. "+BiLSTM"은 "LTR & No NSP"의 모델에 output 구조에 무작위로 초기화된 BiLSTM을 추가한 항목이다.



### 5.1 Effect of Pre-training Tasks

여기서는 BERT에서 deep bidirectionality의 중요성을 두 개의 training 목적 함수를 사용하면서 설명한다. 이 때 동일한 데이터, 동일한 fine tuning, 동일한 파라미터를 사용한다.

#### No NSP

NSP task가 없고 masked LM만 사용하는 bidirectional model이다.

#### LTR & No NSP

MLM 방식이 아닌, Left-to-Right, LTR 방식의 Language Model, LM을 이용해 학습하는 left-context-only model이다. 이 left-only는 \(pre-trained 뿐만 아니라\) fine tuning에서도 한계점으로 작용한다. MLM 방식을 포기하면서 pre-train과 fine-tune에서downstream task들의 성능이 하락되는 문제\(=mismatch, LTR방식의 LM이 downstream과 잘 맞지 않는다는 것을 mismatch로 표현했다\)가 발생한다. 게다가, NSP task가 없이 사전 학습되었기 때문에 이는 GPT와 직접적으로 비교가 가능하지만 BERT가 좀 더 큰 데이터셋과, 큰 임베딩 차원 그리고 버트만의 fine tuning 방식을 사용했다는 차이점이 있다.

우리는 NSP가 가져다 주는 영향에 대해 실험했다. 표 5에서 NSP를 제거하면 QNLI나 MNLI 그리고 SQuAD 1.1에서 성능이 두드러지게 하락하는 것을 볼 수 있다. 또, 양방향 attention 모델의 영향을 확인하기 위해 "No NSP"와 "LTR & No NSP"를 비교해봤더니 LTR모델이 MLM보다 모든 태스크에서 더 낮은 성능을 냈다. 특히 MRPC와 SQuAD에서 대폭 하향됐다.

* 왜 대폭 하향했을까? MRPC는 온라인 뉴스에서 자동으로 추출된 두 문장간의 유사도를 확인하는 Task이고 SQuAD는 질문/답변 쌍이다. 특히 두개의 문장쌍을 이용한 Task에서 LTR 방식을 사용했더니 성능이 하락한 것. LTR보다 MLM이 문장 레벨의 attention을 잘 표현하고 비교할 수 있음을 보여준다.

SQuAD 데이터셋에서 LTR 모델의 token 예측이 형편 없음을 분명히 보여준다. 이는 token-level에서의 hidden states는 오른쪽 sequence의 문맥정보가 없기 때문이다. LTR 모델을 개선할 수 있지 않을까라는 선의\(이미, BERT에서 bidirectionality의 중요성을 말해줬기 때문에 constraint한 LTR은 이제 degenerate한 모델이 되었지만 한번 회생할 수 있는 기회를 준다라는 의미로 받아들여진다\)가 생겨서 모델의 가장 위쪽에 무작위로 초기화된 BiLSTM을 추가했다. 이것은 SQuAD 데이터셋에 대해서는 두드러진 개선을 보였다. 하지만 아직도 양방향 모델의 성적보다는 꽤 많이 못미쳤다. BiLSTM은 \(오히려\) GLUE task의 성적을 더 해쳤다.

물론, ELMO가 그랬던 것처럼 LTR과 RTL 모델을 각각 학습하고 두 모델의 representation을 concatenation해서 사용할 수도 있다. 그러나 \(a\) 이는 하나의 양방향 모델보다 두 배 더 비싼 방법이다. \(b\) 이러한 방법은 QA task에 관해서는 비직관적이다. 왜냐하면 RTL 모델은 질문에 대한 답변을 조절할 수 없기 때문이다. \(c\) 깊은 양방향 모델은 모든 레이어에서 동시에 왼쪽과 오른쪽의 문맥을 사용하기 때문에 이보다는 분명히 성능이 낮을 수 밖에 없다.



### 5.2 Effect of Model Size

이번 장에서는 모델의 크기가 fine tuning시 task에 정확도에 미치는 영향에 대해 알아보겠다. 우리는 서로 다른 레이어 수, hidden unit, attention heads를 조절해가며 여러 BERT 모델을 학습시켰다. 이 때 언급하지 않은 다른 파라미터는 이전에 사용한것과 동일하다.

GLUE tasks로 진행된 결과는 표 6에서 볼 수 있다. 여기서는 무작위로 초기화 된 5개의 Dev Set의 평균 정확도를 나타낸다. 4개의 데이터셋에 대하여 모델이 클수록 확실한\(=strict, 엄격한\) 정확도 개선을 이루어지는 것을 알 수 있다. MRPC 데이터는 겨우 3600개의 라벨링 된 데이터만이 존재하고 pre-training task와는 대체로 차이가 있는데도 말이다. 놀랄지도 모르는데, 우리는 기존에 이루어진 연구들에서 소개된\(=매우 밀접한 관련이 있는\) 모델들의 꼭대기에서 \(한번 더 \) 두드러진 성능 개선을 이루어냈다는 것이다. 예를 들어 Vaswani et al. \(2017\)에 소개된 초대형 트랜스포머\(L=6, H=1024, A=16\)는 인코더에서만 1억개의 파라미터를 가지고 있고 Al-Rfou et al., 2018에 소개된 \(또 다른\) 초대형 트랜스포머 모델\(L=64, H=512, A=2\)는 2.35억개의 파라미터를 가지고 있다. 대조적으로 버트 베이스 모델은 1.1억개의 파라미터를, 버트 라지 모델은 3.4억개의 파라미터를 가진다.

* dev set은 검증 데이터 valid set을 의미한다.
* pre-training task와 MRPC가 차이가 있다는 부분은 무슨 의미인지 잘 모르겠다.
* 버트 모델과 기존 트랜스포머 모델을 비교한 이유가 무엇인지 잘 모르겠다.

모델 사이즈를 키우면 기계번역이나 언어 모델링같은 매우 큰 task들에 대해 계속 성능이 늘어난다는 것은 이전부터 알고있었다. 이러한 사실은 표 6에서 held-out training data를 가지고 평가된 LM perplexity 점수로도 증명된다. 그러나 여기서 우리가 최초로 증명한 것은 \(단순히 큰 task들 뿐만 아니라\) pre training만 충분히 잘 되었다면 매우 작은 task에 대해서도 엄청난 성능 개선이 확실하게 이루어진다는 것이다. \(그래서 위에서 3600개의 데이터밖에 없는 MRPC를 언급한 것\) Peters et al. \(2018b\) 에서는 pre-trained된 bi-LM의 사이즈를 2개의 레이어에서 4개로 늘렸을 때 downstream task에 미치는 영향에 대한 서로 다른\(=mixed\) 결과를 내놓았다. 또, Melamud et al. \(2016\) 에서는 \(차원을 키우는 과정에서\) hidden 차원을 200에서 600으로 늘렸을 때는 도움이 됐지만 1000으로 늘렸을 때는 더이상의 개선이 없었다.

![](../../.gitbook/assets/image%20%281238%29.png)

#### Table 6

버트의 모델 사이즈에 대한 ablation 실험이다. \#L은 레이어의 수 \#H는 히든 차원, \#A는 attention head의 수이다. LM\(ppl\)은 held-out방식의 training data의 MLM perplexity 이다.

* ppl은 perplexity의 준말로 모델 내에서 자신의 성능을 수치화 한 내부평가이다. 외부평가보다 조금 부정확할 수는 있지만 테스트 데이터에 대해서 빠르게 식으로 계산되어서 더 간단한 평가방법이다.
  * 이 뜻은 직역하면 혼잡한, 헷갈리는 이라는 뜻이며 실제로 얼마나 헷갈리냐에 대한 척도이다.
  * ppl은 단어의 수로 정규화 된 테스트 데이터에 대한 확률의 역수인데, 쉽게 말하면 특정 시점에서 평균적으로 몇 개의 선택지를 가지고 고민을 하는지에 대한 모델의 경우의 수를 의미한다고 보면 된다.
  * 이 수치가 낮을 수록 언어 모델의 성능이 좋다. 이 값이 낮으면 높은 정확도를 보이게 된다.
* held-out은 교차검증에 한 방법으로 쓰이는 hold-out 교차검증의 의미로 쓰인것으로 추측된다. hold-out 교차검증은 데이터셋은 훈련셋과 테스트셋 또는 훈련셋과 테스트셋과 검증셋으로 나누어 학습하는 방법을 의미한다.







