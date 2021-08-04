---
description: '210804'
---

# \[선택 과제 1\] Gradient Descent

## 1. Gradient Descent

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
* 
### Func\_gradient

```python

```

* \`\`
* `sym.diff` 는 도함수를 구해준다.



