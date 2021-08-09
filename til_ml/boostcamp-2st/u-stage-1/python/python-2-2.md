---
description: '210802'
---

# \(Python 2-2강\) Function and Console I/O

### 함수

어떤 일을 수행하는 코도의 덩어리

* 반복적인 수행을 할 때 1회만 작성하고 반복 호츨 할 수 있다
* 코드를 논리적인 단위로 분리한다
* 캡슐화 : 인터페이스만 알면 타인의 코드를 사용할 수 있다

문법

```python
def 함수이름 (parameter1, ...):
    수행문 #1
    수행문 #2
    return 반환값
```

예시

```python
def cal_add(a, b):
    result = a+b
    return result
```

Parameter vs Argument

* Parameter : 함수의 입력 값 인터페이스
  * def f\(x\): 에서의 x
* Argument : 실제 Parameter에 대입된 값
  * f\(2\) 에서의 2

콘솔창 입출력

* Input\(\) : 콘솔창에서 문자열을 입력 받는 함수
* print\(\) : 콘솔창에서 출력을 담당



### Print formating

기본적인 출력 외에 3가지 출력 형식을 지정 가능하다

```python
print(1,2,3)

print("a" + " " + "b" + " " + "c")

print("%d %d %d" % (1,2,3))

print("{} {} {}".format("a","b","c"))

print(f"value is {value})
```

old-school formatting

* 일반적으로 %-format과 str.format\(\) 함수를 사용함

```python
print('%s %s' % ('one', 'two'))

print('{} {}'.format('one', 'two'))

print('%d %d' % (1, 2))

print('{} {}'.format(1, 2))
```

% - format

```python
print("I eat %d apples." % 3)

print("I eat %s apples." % "five")

number = 3; day="three"
print("I ate %d apples. I was sick for %s days." % (number, day))

print("Product: %s, Price per unit: %f." % ("Apple", 5.243))
```

* %s 문자열 \(String\)
* %c 문자 1개\(character\)
* %d 정수 \(Integer\)
* %f 부동소수 \(floating-point\)
* %o 8진수
* %x 16진수
* %% Literal % \(문자 % 자체\)

str.format

```python
age = 36; name='Sungchul Choi'
print("I’m {0} years old.".format(age))
print("My name is {0} and {1} years old.".format(name,age))
print("Product: {0}, Price per unit: {1:.3f}.".format("Apple", 5.243))
```

naming

```python
print("Product: %(name)10s, Price per unit: %(price)10.5f." %
{"name":"Apple", "price":5.243})

print("Product: {name:>10s}, Price per unit:
{price:10.5f}.".format(name="Apple", price=5.243))
```

f-string

```python
name = "Sungchul"
age = 39
print(f"Hello, {name}. You are {age}.")
print(f'{name:20}')
print(f'{name:>20}')
print(f'{name:*<20}')
print(f'{name:*>20}')
print(f'{name:*^20}')

Hello, Sungchul. You are 39.
Sungchul
Sungchul
Sungchul************
************Sungchul
******Sungchul******
```







