---
description: '210804, 210807'
---

# \[선택 과제 1\] Gradient Descent

## 1. Gradient Descent \(1\)

```python
# https://docs.sympy.org/latest/modules/polys/domainsintro.html
>>> from sympy import Symbol, Poly
>>> x = Symbol('x')
>>> Poly(x**2 + x)
Poly(x**2 + x, x, domain='ZZ')
>>> Poly(x**2 + x/2)
Poly(x**2 + 1/2*x, x, domain='QQ')
```

* `Symbol`은 변수를 수식의 문자로 사용할 수 있게 끔 하는 함수이다.
* `Poly`는 `Symbol` 변수를 가지고 수식을 구성하는 함수이다.
  * 이 때 domain은 다음과 같다
    * ZZ : 다항식의 계수가 정수
    * QQ : 다항식의 계수가 유리수

```python
# https://www.geeksforgeeks.org/python-sympy-subs-method-2/
# import sympy
from sympy import *
  
x, y = symbols('x y')
exp = x**2 + 1
print("Before Substitution : {}".format(exp)) 
    
# Use sympy.subs() method
res_exp = exp.subs(x, y) 
    
print("After Substitution : {}".format(res_exp))

# Output
Before Substitution : x**2 + 1
After Substitution : y**2 + 1
```

* 두 개 이상의 심볼은 공백을 두고 설정할 수 있다.
* `sym.subs` 는 기존의 다항식의 변수를 다른변수로 치환하거나 값들 대입해준다.
  * 첫번째 인자는 기존 심볼, 두번째 인자는 치환할 심볼이다.

```python
# https://www.geeksforgeeks.org/python-sympy-diff-method/
# import sympy
from sympy import * x, y = symbols('x y')
gfg_exp = x + y
exp = sympy.expand(gfg_exp**2)
print("Before Differentiation : {}".format(exp))

# Use sympy.diff() method
dif = diff(exp, x)

print("After Differentiation : {}".format(dif))

# Output
Before Differentiation : x**2 + 2*x*y + y**2
After Differentiation : 2*x + 2*y
```

* `sym.diff` 는 도함수를 출력한다.
  * 첫번째 인자는 다항식을, 두번째 인자는 미분할 심볼이다.



```python
import numpy as np
import sympy as sym
from sympy.abc import x
from sympy.plotting import plot

def func(val):
    fun = sym.poly(x**2 + 2*x + 3)
    return fun.subs(x, val), fun

def func_gradient(fun, val):
    diff = sym.diff(fun(x)[1], x)
    return diff.subs(x, val), diff

def gradient_descent(fun, init_point, lr_rate=1e-2, epsilon=1e-5):
    cnt = 0
    val = init_point
    while True:
        cnt += 1
        diff, _ = func_gradient(fun, val)
        if diff <= epsilon:
            break
        val -= lr_rate * diff
    print("함수: {}\n연산횟수: {}\n최소점: ({}, {})".format(fun(val)[1], cnt, val, fun(val)[0]))
```

* `func`
  * 주어진 다항식에 인자로 받은 상수를 대입한 결과와, 다항식을 반환
* `func_gradient`
  * 인자로 받은 다항식의 도함수를 구한다. 인자로 받은 상수를 도함수에 대입한 결과와 도함수를 반환
* `gradient_descent`
  * 반복적으로 도함수를 구해서 경사하강법을 구현한다.
  * 이 때 시작지점 `val` 이 학습률과 변화율을 곱한 `lr_rate * diff` 만큼씩 변화해가도록 한다
  * 특정 `val` 에서 도함수 값이 `epsilon` 보다 작으면 반복을 끝내도록 한다.



## 2. Gradient Descent \(2\)

```python
def func(val):
    fun = sym.poly(x**2 + 2*x + 3)
    return fun.subs(x, val)

def difference_quotient(f, x, h=1e-9):
    return (f(x+h)-f(x))/h

def gradient_descent(func, init_point, lr_rate=1e-2, epsilon=1e-5):
    cnt = 0
    val = init_point
    ## Todo
    while True:
        cnt += 1
        diff = difference_quotient(func, val)
        if diff <= epsilon:
            break
        val -= diff * lr_rate

    print("연산횟수: {}\n최소점: ({}, {})".format(cnt, val, func(val)))
```

* `func`
  * 1. 과 동일
* `difference_quotient`
  * 실제 미분 공식을 반환한다.
* `gradient_descent`
  * 1. 과 동일



## 3. Linear Regression

### 3.1. Basic function

```python
train_x = (np.random.rand(1000) - 0.5) * 10
train_y = np.zeros_like(train_x)

def func(val):
    fun = sym.poly(7*x + 2)
    return fun.subs(x, val)

for i in range(1000):
    train_y[i] = func(train_x[i])

# initialize
w, b = 0.0, 0.0

lr_rate = 1e-2
n_data = len(train_x)
errors = []

for i in range(100):
    ## Todo
    # 예측값 y
    _y = w * train_x + b

    # gradient
    gradient_w = np.dot(_y - train_y,  train_x) / n_data
    gradient_b = np.sum(_y - train_y) / n_data

    # w, b update with gradient and learning rate
    w -= lr_rate * gradient_w
    b -= lr_rate * gradient_b

    # L2 norm과 np_sum 함수 활용해서 error 정의
    error = np.sum((_y - train_y) ** 2) ** 0.5 / n_data
    # Error graph 출력하기 위한 부분
    errors.append(error)

print("w : {} / b : {} / error : {}".format(w, b, error))
```

![](../../../../.gitbook/assets/image%20%28785%29.png)

* `gradient_w`
  * 예측값과 실제값의 차 그리고 x의 곱을 전체 개수로 나눠준 것이 w의 그레디언트 값이다.
    * 이 때 곱을 `np.dot` 으로 연산해준다.
      * `np.dot(a, b)` 에서 a또는 b가 행벡터 또는 열벡터라면 따로 전치를 하지 않아도 된다.
    * 각 변수의 사이즈는 다음과 같다
      * `x` : \(m, n\)
      * `w` : \(n, 1\)
      * `y-wx` : \(m, 1\)
      * `x_T` : \(n, m\)
      * `np.dot(x_T, y-wx)` : \(n, 1\)
        * `w` 와 크기가 같다. 갱신가능
* `gradient_b` : 위와 동일
* `error`
  * L2 norm을 이용한 에러 표현



### 3.2. More complicated function

```python
train_x = np.array([[1,1,1], [1,1,2], [1,2,2], [2,2,3], [2,3,3], [1,2,3]])
train_y = np.dot(train_x, np.array([1,3,5])) + 7

# random initialize
beta_gd = [9.4, 10.6, -3.7, -1.2]
# for constant element
expand_x = np.array([np.append(x, [1]) for x in train_x])

for t in range(5000):
    beta_gd -= lr_rate * np.dot(np.dot(beta_gd, expand_x.T) - train_y, expand_x) / len(train_x)

print("After gradient descent, beta_gd : {}".format(beta_gd))
```

* `expand_x`
  * `wx+b` 꼴 대신 b를 가중치의 한 원소로 생각해서 `wx` 꼴로 표현한다.
* `np.dot(beta_gd, expand_x.T)`

  * `wx` 꼴
  * `beta_gd` : \(1, 4\)
  * `expand_x` : \(6, 4\)

## 4. Stochastic Gradient Descent

```python
train_x = (np.random.rand(1000) - 0.5) * 10
train_y = np.zeros_like(train_x)

def func(val):
    fun = sym.poly(7*x + 2)
    return fun.subs(x, val)

for i in range(1000):
    train_y[i] = func(train_x[i])

# initialize
w, b = 0.0, 0.0

lr_rate = 1e-2
n_data = 10
errors = []

for i in range(100):
    ## Todo
    idx = np.random.choice(1000, 10, False)
    x_choice = train_x[idx]
    y_choice = train_y[idx]

    _y = w * x_choice + b

    w -= lr_rate * np.dot(_y - y_choice,  x_choice) / n_data
    b -= lr_rate * np.sum(_y - y_choice) / n_data

    error = np.sum((_y - y_choice) ** 2) ** 0.5 / n_data
    # Error graph 출력하기 위한 부분
    errors.append(error)

print("w : {} / b : {} / error : {}".format(w, b, error))
```

* `np.random.choice(a, b, c)`
  * `a` : 무작위로 뽑을 수의 범위
  * `b` : 뽑을 수의 개수
  * `c` : 복원 추출 여부
  * `np.array` 는 그 자체로 인덱스로 사용할 수 있어서 `train_x[idx]` 와 같이 사용이 가능하다

 

