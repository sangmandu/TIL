---
description: '210909'
---

# \[필수 과제 3\] Subword-level Language Model

과제2와 매우 비슷하므로 코드 분석은 패스하고 한 가지 주제만 다루고자 한다.

tokenization의 방법은 세 가지 정도로 말할 수 있다. 예를 들어 다음과 같은 문장이 있다고 하자

> I love my pet very much

이 때, 토크나이징 방법에  따라 다음의 토큰이 생성된다.

* word : I, love, my, pet, very, much
* subword : I, lo, \#\#ve, my, pet, ve, \#\#ry, muc, \#\#h
  * 이러한  subword 방식은 한 단어에서만 이루어지며 두 단어의 일부 알파벳이 이어지는 경우는 없다.
* character : I, l, o, v, e, m, y, ...



word 기반 tokenization의 파라미터 개수는 다음과 같다.

![](../../../.gitbook/assets/image%20%281129%29.png)

subword 기반 tokenization의 파라미터 개수는 다음과 같다.

![](../../../.gitbook/assets/image%20%281125%29.png)

* 이 때의 tokenizer는 `BertTokenizer.from_pretrained("bert-base-cased")` 를 사용했다.



이상하지 않은가? subword 기반은 생성되는 token이 더 많아졌을텐데, 어째서 파라미터 수가 더 줄어들은 것일까?

* 사실 나만 이상하게 생각했을 수도 있다. 데이터셋이 작다면 당연히 subword의 토큰이 더 많은 것은 사실이다.
* 애초에 elephant 라는 단어 하나만 있다고 해보자.
* word : elephant
* subword : ele ph ant
* character : e l p h a n t

character가 제일 많지 않은가? 그러나, 데이터가 많아질수록 character는 26개의 token으로 모두 표현할 수 있게된다.

이와 마찬가지로 subword도 단순히 당장에는 많아보이지만 dataset의 크기가 늘어날수록 word보다 더 작은 vocab을 구성할 수 있다.

