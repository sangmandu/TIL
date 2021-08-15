---
description: '210804'
---

# \[필수 과제 3\] Text Processing 2

### 사전세팅

* 없다



### Digits\_to\_words

```python
def digits_to_words(input_string):
    number_word = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    digit_string = ' '.join([number_word[int(i)] for i in input_string if i.isdigit()])
    return digit_string
```

* 숫자에 해당하는 영어단어를 각 숫자의 인덱스에 위치한 `number_word` 를 선언한다.
  * 이 때 영어단어들은 소문자
* `isdigit()` 로 숫자인지 확인하고 숫자라면 `number_word` 의 원소와 매칭한다.
  * 숫자가 존재하지 않아도 빈 문자열이 반환된다.
* 반환은 띄어쓰기가 한칸이 존재해야 하므로 `' '.join` 으로 구성한다.



### To\_camel\_case

```python
def to_camel_case(underscore_str):
    pre_camel = underscore_str.split('_')
    pre_camel = [i for i in pre_camel if i]
    if len(pre_camel) == 1:
      return ''.join(pre_camel)
    camelcase_str = ''.join(pre_camel[:1]).lower() + ''.join([i[:1].upper()+i[1:].lower() for i in pre_camel[1:]])
    return camelcase_str
```

* `split('_')`
  * camelcase로 변환하기 위해 각 단어를 나눠준다
    * 나눠진 각 단어의 첫글자를 대문자로 설정하면 된다
  * 앞과 뒤에 여러개의 underscore를 무시하게 된다.
    * 이 때 빈 문자열이 반환된다.
    * 따라서, 빈 문자열이 아닌 문자만 얻는다
* 6번 라인 코드
  * \``''.join(pre_camel[:1]).lower()`
    * 가장 첫 단어는 모두 소문자여야 한다
  * `''.join([i[:1].upper()+i[1:].lower() for i in pre_camel[1:]])`
    * 가장 첫 단어를 제외하고는 각 단어의 첫 글자는 대문자여야 하고 나머지 글자는 소문자어야한다.
  * 인덱싱이 아니라 슬라이싱을 사용함으로써 빈 문자열일 때도 통과되게 한다
    * 빈 문자열일 경우 인덱싱을 사용하면 오류가 발생
* 4번 라인 코드
  * 단어 수가 하나일 때는 두 단어의 결합이 아니므로 camel\_case를 적용할 필요가 없다. 이 때는 바로 단어를 출력하도록 한다.



