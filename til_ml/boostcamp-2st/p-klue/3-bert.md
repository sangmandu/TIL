---
description: '210928'
---

# \(3강\) BERT 언어모델 소개

## 1. BERT 언어모델

### 1.1 BERT 모델 소개

버트 모델이 등장하기 전에는 다음의 모델들이 있었다.

![](../../../.gitbook/assets/image%20%281228%29.png)

* 컨텍스트 벡터의 정보를 넘겨주는, 인코더와 디코더가 분리된 구조
* 컨텍스트 벡터만으로 얻는 정보의 한계가 있어 개선된, 어텐션을 사용한 인코더 디코더 구조
* 인코더와 디코더가 하나로 합쳐져 전반적으로 어텐션을 사용하는 구조



이미지에도 인코더와 디코더의 구조를 가지고 있는 `오토인코더` 가 존재한다.

![](../../../.gitbook/assets/image%20%281231%29.png)

* 인코더의 목표는 어떠한 DATA를 압축된 형태로 표현하는 것
* 디코더의 목표는 압축된 DATA를 가지고 원본 DATA로 복원하는 것

이것을 버트에 대입해보자.

버트는 self attention을 이용하는 transformer 모델을 사용한다. 입력된 정보를 다시 입력된 정보로 representation 하는 것을 목표로 한다. 근데 이 때 MASK를 사용하게 된다. 이 때 MASK가 있으므로 입력된 정보가 다시 입력된 정보로 나오기가 어려워진다.

![](../../../.gitbook/assets/image%20%281223%29.png)

버트는 MASK된 자연어를 원본 자연어로 복원하는 작업을, GPT는 특정한 시퀀스를 잘라버리고 그 NEXT 시퀀스를 복원하는 작업이 이루어지게된다.

버트 모델의 구조는 다음과 같다.

![](../../../.gitbook/assets/image%20%281236%29.png)

* 인풋은 문장 2개를 입력받는다.
* 12개의 레이어가 ALL TO ALL로 연결되어있다.
* CLS는 두 개의 문장이 진짜 연결되어있는지 아닌지에 대한 부분을 학습하기 위한 CLASS LABEL로 사용된다.

버트의 CLS 토큰은 문장1과 문장2의 벡터들이 녹아들어있다고 가정하고있다. 그래서 CLS 토큰을 얻고 이를 Classification layer를 부착해 pre training을 진행하게 된다.

![](../../../.gitbook/assets/image%20%281239%29.png)

Tokenizing이 끝나면 masking을 하게된다.

![](../../../.gitbook/assets/image%20%281237%29.png)

* cls와 sep 토큰을 제외한 토큰에서 15%를 고른다.
* 이 중 80%는 masking, 10%는 vocab에 있는 또 다른 단어로 replacing, 10%는 unchanging 한다



![](../../../.gitbook/assets/image%20%281224%29.png)

GLUE 데이터셋을 사용하며, 여기서 최고기록을 내는 모델이 Sota 라고 할 수 있다.

이러한 12가지의 Task를 4가지 모델로 다 표현할 수 있다.

![](../../../.gitbook/assets/image%20%281238%29.png)

* 단일 문장 분류
  * 버트 모델의 한 개의 문장이 입력됐을 때 이 문장이 어떤 class에 속하는지에 대해 분류
* 두 문장 관계 분류
  * NSP
  * s1이 s2의 가설이 되는가
  * s1과 s2의 유사도
* 문장 토큰 분류
  * 보통 CLS토큰이 입력된 sentence에 대한 정보가 녹아들어 있다고 가정되는데 이 처럼 각 토큰들에 대해 분류기를 부착한다.
  * 보통 개체명인식에 많이 사용
* 기계 독해 정답 분류
  * 2가지 정보가 주어진다.
    * 하나는 질문 \(ex 이순신의 고향은\)
    * 하나는 그 정답이 포함된 문서
  * 문서에서 정답이라고 생각되는 부분의 start token과 end token을 반환한다.



### 1.2 BERT 모델의 응용

#### 감성 분석

입력된 문장의 긍부정을 파악한다.

![](../../../.gitbook/assets/image%20%281227%29.png)



#### 관계 추출

주어진 문장에서 sbj와 obj가 정해졌을 때 둘은 무슨 관계인가?

![](../../../.gitbook/assets/image%20%281226%29.png)





#### 의미 비교

두 문장이 의미적으로 같은가?

![](../../../.gitbook/assets/image%20%281233%29.png)

* task에 살짝 문제가 있는데, s1과 s2가 너무 상관없는 문장으로 매칭되었다. 실제로 적용하기에는 어려운 부분이 있다.
* 그래서, 98.3점의 점수는 높지만 데이터 설계부터 잘못된 task이다.



#### 개체명 분석

![](../../../.gitbook/assets/image%20%281222%29.png)



#### 기계 독해

![](../../../.gitbook/assets/image%20%281225%29.png)



### 1.3 한국어 BERT 모델

#### ETRI KoBERT의 tokenizing

바로 WordPiece 단위로 tokenizing 한것이 아니라 형태소 단위로 분리를 먼저 한뒤 tokenizing했다. 한국어에 특화되게 토크나이징 했다는 점에서 많은 성능향상을 가져왔다.

![](../../../.gitbook/assets/image%20%281241%29.png)

* CV : 자모
* Syllable : 음절
* Morpheme : 형태소
* Subword : 어절
* Morpheme-aware Subword : 형태소 분석 후 Wordpiece
* Word : 단어



#### Advanced BERT model

버트 내에는 entity를 명시할 수 있는 구조가 존재하지 않는다. 그래서 전처리로 각 entity앞뒤로 ENT 태그를 붙여주었다. 그랬더니 성능이 향상되었다.

![](../../../.gitbook/assets/image%20%281230%29.png)

![](../../../.gitbook/assets/image%20%281240%29.png)

![](../../../.gitbook/assets/image%20%281242%29.png)

이렇게 entity 태그를 달아주면 성능이 향상되는데, 영어권 분야에서도 이렇게 태그를 달아준 모델이 sota를 찍고있다.



