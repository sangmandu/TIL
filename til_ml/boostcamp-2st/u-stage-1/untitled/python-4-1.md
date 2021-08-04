---
description: '210803'
---

# \(Python 4-1강\) Python Object Oriented Programming

### 객체지향 프로그래밍 개요

* Object-Oriented Programming, OOP
* 객체: 실생활에서 일종의 물건
* 속성\(Attribute\)과 행동\(Action\)을 가짐
* 파이썬 역시 객체 지향 프로그램 언어



### 클래스 선언

* class 선언, object는 python3에서 자동 상속

```python
class SoccerPlayer(object):
# class : class 예약어
# SoccerPlayer : class 이름
# (object) : 상속받는 객체명
```

* 명명법이 존재
  * 파이썬 함수/변수명에는 snake\_case를 사용
  * Class명은 CamelCase를 사용



### 상속

부모클래스로부터 속성과 메서드를 물려받은 자식클래스를 생성하는 것



### 다형성

* 같은 이름 메소드의 내부 로직을 다르게 작성
* Dynamic Typing 특성으로 인해 파이썬에서는 같은 부모클래스의

  상속에서 주로 발생함

```python
class Animal:
    def __init__(self, name): # Constructor of the class
        self.name = name
    def talk(self): # Abstract method, defined by convention only
        raise NotImplementedError("Subclass must implement abstract method")

class Cat(Animal):
    def talk(self):
        return 'Meow!'
        
class Dog(Animal):
    def talk(self):
        return 'Woof! Woof!'

animals = [Cat('Missy'),
        Cat('Mr. Mistoffelees'),
        Dog('Lassie')]
for animal in animals:

    print(animal.name + ': ' + animal.talk())
```



### 가시성

* 객체의 정보를 볼 수 있는 레벨을 조절하는 것
* 누구나 객체 안에 모든 변수를 볼 필요가 없음

  1\) 객체를 사용하는 사용자가 임의로 정보 수정

  2\) 필요 없는 정보에는 접근 할 필요가 없음

  3\) 만약 제품으로 판매한다면? 소스의 보호



### First-class objects

* 일등함수 또는 일급 객체
* 변수나 데이터 구조에 할당이 가능한 객체
* 파라메터로 전달이 가능 + 리턴 값으로 사용
* 파어썬의 함수는 일급함수이다



### Inner function

* 함수 내에 또 다른 함수가 존재

```python
def print_msg(msg):
    def printer():
        print(msg)
    printer()

print_msg("Hello, Python")
```

* closures
  * inner function을 return값으로 반환

```python
def print_msg(msg):
    def printer():
        print(msg)
    return printer
    
another = print_msg("Hello, Python")
another()
```

