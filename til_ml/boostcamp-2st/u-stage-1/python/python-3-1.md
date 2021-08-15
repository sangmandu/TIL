---
description: '210803'
---

# \(Python 3-1강\) Python Data Structure

### 파이썬 기본 데이터 구조

* 스택과 큐
* 튜플과 집합
* 사전
* Collection 모듈



### 스택

* 나중에 넣은 데이터를 먼저 반환하도록 설계된 메모리 구조
* Last In First Out \(LIFO\)
* Data의 입력을 Push, 출력을 Pop이라고 함



### 큐

* 먼저 넣은 데이터를 먼저 반환하도록 설계된 메모리 구조
* First In First Out \(FIFO\)
* Stack과 반대되는 개념



### 튜플

* 값의 변경이 불가능한 리스트
* 선언 시 “\[ \]” 가 아닌 “\( \)”를 사용
* 리스트의 연산, 인덱싱, 슬라이싱 등을 동일하게 사용
* 왜 사용하는가?
  * 프로그램을 작동하는 동안 변경되지 않은 데이터의 저장 Ex\) 학번, 이름, 우편번호 등등
  * 함수의 반환 값등 사용자의 실수에 의한 에러를 사전에 방지



### 집합

* 값을 순서없이 저장, 중복 불허 하는 자료형
* set 객체 선언을 이용하여 객체 생성



### 딕셔너리

* 데이터를 저장 할 때는 구분 지을 수 있는 값을 함께 저장 예\) 주민등록 번호, 제품 모델 번호
* 구분을 위한 데이터 고유 값을 Identifier 또는 Key 라고함
* Key 값을 활용하여, 데이터 값\(Value\)를 관리함



### Collections

* List, Tuple, Dict에 대한 Python Built-in 확장 자료 구조\(모듈\)
* 편의성, 실행 효율 등을 사용자에게 제공함
* 아래의 모듈이 존재함

```python
from collections import deque
from collections import Counter
from collections import OrderedDict
from collections import defaultdict
from collections import namedtuple
```

* deque
  * Stack과 Queue 를 지원하는 모듈
  * List에 비해 효율적인=빠른 자료 저장 방식을 지원함 =&gt; 처리 속도 향상
  * rotate, reverse등 Linked List의 특성을 지원함
  * 기존 list 형태의 함수를 모두 지원함
* OrderedDict
  * Dict와 달리, 데이터를 입력한 순서대로 dict를 반환함

    그러나 dict도 python 3.6 부터 입력한 순서를 보장하여 출력함
* defaultdict
  * Dict type의 값에 기본 값을 지정, 신규값 생성시 사용하는 방법
* Counter
  * Sequence type의 data element들의 갯수를 dict 형태로 반환
  * `counter.elements()` 로 count만큼의 key를 list로 반환
  * `counter.subtract(counter)` 로 set의 연산들을 사용 가능
    * `counter - counter` 로도 사용가능



### namedtuple

* Tuple 형태로 Data 구조체를 저장하는 방법
* 저장되는 data의 variable을 사전에 지정해서 저장함



