---
description: '210803'
---

# \(Python 3-2강\) Pythonic code

### Pythonic Code

* 파이썬 스타일의 코딩 기법
* 파이썬 특유의 문법을 활용하여 효율적으로 코드를 표현함
* 그러나 더 이상 파이썬 특유는 아님, 많은 언어들이 서로의 장점을 채용
* 고급 코드를 작성 할 수록 더 많이 필요해짐

```python
>>> colors = ['red', 'blue', 'green', 'yellow']
>>> result = ''
>>> for s in colors:
result += s

>>> colors = ['red', 'blue', 'green', 'yellow']
>>> result = ''.join(colors)

>>> result
'redbluegreenyellow'
```

* 전자\(1-4\)보다 후자\(6-7\)가 더 파이토닉한 방법



### Split

* string type의 값을 “기준값”으로 나눠서 List 형태로 변환



### Join

* String으로 구성된 list를 합쳐 하나의 string으로 반환



### List Comprehension

* 기존 List 사용하여 간단히 다른 List를 만드는 기법
* 포괄적인 List, 포함되는 리스트라는 의미로 사용됨
* 파이썬에서 가장 많이 사용되는 기법 중 하나
* 일반적으로 for + append 보다 속도가 빠름



### Enumerate

* list의 element를 추출할 때 번호를 붙여서 추출



### Zip

* 개의 list의 값을 병렬적으로 추출함



### Lambda

* 함수 이름 없이, 함수처럼 쓸 수 있는 익명함수
* 수학의 람다 대수에서 유래함

```python
# General Function
def f(x, y):
    return x + y
print(f(1, 4))

# Lmabda Function
f = lambda x, y: x + y
print(f(1, 4))
```

* Python 3부터는 권장하지는 않으나 여전히 많이 쓰임
  * 왜 권장하지 않는가?
    * 어려운 문법
    * 테스트의 어려움
    * 문서화 docstring 지원 미비
    * 코드 해석의 어려움
    * 이름이 존재하지 않는 함수의 출현



### Map

* 두 개 이상의 list에도 적용 가능

```python
ex = [1,2,3,4,5]
print(list(map(lambda x: x+x, ex)))

>>> [2, 4, 6, 8, 10]
```

### 

### Reduce

* map function과 달리 list에 똑같은 함수를 적용해서 통합

```python
from functools import reduce
print(reduce(lambda x, y: x+y, [1, 2, 3, 4, 5]))

>>> 15
```



### Function passing arguments

Keyword arguments

* 함수에 입력되는 parameter의 변수명을 사용, arguments를 넘김

```python
def print_somthing(my_name, your_name):
    print("Hello {0}, My name is {1}".format(your_name, my_name))

print_somthing(your_name="TEAMLAB", my_name="Sungchul")
```

Default arguments

* parameter의 기본 값을 사용, 입력하지 않을 경우 기본값 출력

```python
def print_somthing_2(my_name, your_name="TEAMLAB"):
    print("Hello {0}, My name is {1}".format(your_name, my_name))
    
print_somthing_2("Sungchul", "TEAMLAB")
print_somthing_2("Sungchul")
```

Variable-length

* 개수가 정해지지 않은 변수를 함수의 parameter로 사용하는 법
* Keyword arguments와 함께, argument 추가가 가능
* Asterisk\(\*\) 기호를 사용하여 함수의 parameter를 표시함
* 입력된 값은 tuple type으로 사용할 수 있음
* 가변인자는 오직 한 개만 맨 마지막 parameter 위치에 사용가능
* 가변인자는 일반적으로 \*args를 변수명으로 사용

```python
def asterisk_test(a, b, *args):
    return a+b+sum(args)
    
print(asterisk_test(1, 2, 3, 4, 5))
```

Keyword variable-length

* Parameter 이름을 따로 지정하지 않고 입력하는 방법
* asterisk\(\*\) 두개를 사용하여 함수의 parameter를 표시함
* 입력된 값은 dict type으로 사용할 수 있음
* 가변인자는 오직 한 개만 기존 가변인자 다음에 사용

```python
def kwargs_test_3(one,two, *args, **kwargs):
    print(one+two+sum(args))
    print(kwargs)
    
kwargs_test_3(3,4,5,6,7,8,9, first=3, second=4, third=5
```

