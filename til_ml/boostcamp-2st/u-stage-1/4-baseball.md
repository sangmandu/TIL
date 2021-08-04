---
description: '210804'
---

# \[필수 과제 4\] Baseball

### 사전 세팅

* 없다



### Is\_digit

```python
def is_digit(user_input_number):
    result = user_input_number.isdigit()
    return result
```

* `isdigit` 를 이용해 자연수인지 판별한다.
  * `isdigit` 는 문자 뿐만 아니라 음수, 실수에 대해서도 `False` 를 반환한다.



### Is\_between\_100\_and\_999

```python
def is_between_100_and_999(user_input_number):
    result = 100 <= int(user_input_number) < 1000
    return result
```

* 주어진 수의 범위를 조사한다
  * python은 부등호의 동시 비교가 가능하다



### Is\_duplicated\_number

```python
def is_duplicated_number(three_digit):
    result = len(set(three_digit)) != 3
    return result
```

* set\(\) 을 이용해서 각 자리수가 중복되는지 검사한다.



### Is\_validated\_number

```python
def is_validated_number(user_input_number):
    result = is_digit(user_input_number) and is_between_100_and_999(user_input_number) and not is_duplicated_number(user_input_number)
    return result
```

* 지금까지 구성한 세 함수에 대해 만족하는지 `and` 를 사용해서 조사한다.
  * `is_duplicated_number` 는 중복되면 아니므로 `not`을 조사한다.













