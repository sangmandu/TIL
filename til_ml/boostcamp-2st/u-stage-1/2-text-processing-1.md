---
description: '210804'
---

# \[필수 과제 2\] Text Processing 1

### 사전 세팅

* 따로 없다.



### Normalize

```python
def normalize(input_string):
    normalized_string = ' '.join(input_string.lower().split())
    return normalized_string
```

* 모든 단어들은 소문자로 되어야함 : `lower()` 사용
* 띄어쓰기는 한칸으로 되어야함 : `' '.join()` 사용
  * join의 앞에 오는 ' ' 는 구분자를 의미한다. 여기서는 구분자를 "한칸" 으로 설정
* 앞뒤 필요없는 띄어쓰기는 제거해야함 : `split()` 사용
  * 물론 띄어쓰기를 한칸으로 하기위한 작업에도 관여를 했다.
  * join의 구분자가 들어갈 위치를 split으로 지정해준 것



### No\_vowels

```python
def no_vowels(input_string):
    no_vowel_string = ''.join([i for i in input_string if i not in 'aeiouAEIOU'])
    return no_vowel_string
```

* 각 스트링을 한 글자씩 검사해서 모음이 아닐 경우만 리스트의 원소로 선택한다
  * 여기서 리스트는 새로운 리스트
  * 이 때 검사는 `not in` 을 사용하여 한다.
* 만들어진 리스트를 join으로 string화 한다.

















