---
description: '210804'
---

# \(Python 5-1강\) File / Exception / Log Handling

### Exception

예외에는 예상이 가능하거나 불가능하다.

예상 가능한 예외

* 발생 여부를 사전에 인지할 수 있는 예외
* 사용자의 잘못된 입력, 파일 호출 시 파일 없음
* 개발자가 반드시 명시적으로 정의 해야함

예상 불가능한 예외

* 인터프리터 과정에서 발생하는 예외, 개발자 실수
* 리스트의 범위를 넘어가는 값 호출, 정수 0으로 나눔
* 수행 불가시 인터프리터가 자동 호출ㅇ



예상이 가능하면 if문으로 조건을 달아주면 된다. 예상이 불가능 하다면? =&gt; Exception Handling



### Exception Handling

try - except 문법

```python
try:
    예외 발생 가능 코드
except <Exception Type>:
    예외 발생시 대응하는 코드
```

try-except-finally 문법

```python
try:
    예외 발생 가능 코드
except <Exception Type>:
    예외 발생시 동작하는 코드
finally:
    예외 발생 여부와 상관없이 실행됨
```



Exception의 종류

* Built-in Exception: 기본적으로 제공하는 예외
  * IndexError : List의 Index 범위 초과
  * NameError : 존재하지 않는 변수 호출
  * ZeroDivisionError : 0으로 숫자를 나눌 때
  * Value Error : 변환할 수 없는 문자/숫자 변환할 때
  * FileNotFoundError : 존재하지 않는 파일을 호출할 때
  * 그 외에도 제공되는 예외가 굉장히 많다



예외 정보 표시하기

```python
for i in range(10):
    try:
        print(10 / i)
    except ZeroDivisionError as e:
        print(e)
        print("Not divided by 0")
```



raise 구문

* 필요에 따라 강제로 Exception을 발생시킨다.

```python
raise <Exception Type>(예외정보)
```

```python
while True:
    value = input("변환할 정수 값을 입력해주세요")
    for digit in value:
        if digit not in "0123456789":
            raise ValueError("숫자값을 입력하지\
            않으셨습니다")
        print("정수값으로 변환된 숫자 -", int(value))
```



assert 구문

* 특정 조건에 만족하지 않을 경우 예외 발생

```python
assert 예외조건
```

```python
def get_binary_nmubmer(decimal_number):
    assert isinstance(decimal_number, int)
    return bin(decimal_number)
    
print(get_binary_nmubmer(10))
```



### File Handling

파일의 종류

* 기본적인 파일 종류로는 text 파일과 binary 파일로 나뉜다.
* 모든 텍스트 파일도 실제로는 바이너리 파일이다.
  * 컴퓨터는 텍스트 파일을 처리하기 위해 바이너리 파일로 변환시킨다.
* text 파일
  * 인간도 이해할 수 있는 형태인 문자열 형식으로 저장된 파일
  * 메모장으로 열면 내용 확인 가능
  * 메모장에 저장된 파일, HTML 파일, 파이썬 코드 파일 등
* binary 파일
  * 컴퓨터만 이해할 수 있는 형태인

    이진\(법\)형식으로 저장된 파일

  * 일반적으로 메모장으로 열면

    내용이 깨져 보임 \(메모장 해설 불가\)

  * 엑셀파일, 워드 파일 등등



Python File I/O

```python
f = open("<파일이름>", "접근 모드")
f.close()

# 접근 모드
r : 읽기모드
w : 쓰기모드
a : 추가모드
```

* w, 쓰기모드의 경우 encoding 방식을 utf8과 cp949 등의 방법으로 정할 수 있다. 협업할 때는 이를 통일하는게 좋다.
* 또, w는 기존 파일을 덮어쓰기 떄문에 주의가 필요하다.
* a, 추가모드는 기존 파일의 마지막부터 이어서 붙이게 된다.



### Directory 다루기

디렉토리 생성

```python
import os
os.mkdir("log")
```

디렉토리 존재 여부 확인

```python
if not os.path.isdir("log"):
    os.mkdir("log")
```

경로 연결하기

```python
folder = os.getcwd()
file = 'abc.txt'
os.path.join(folder, file)
```

* 이 때 문자열로 연결할 수도 있는데, 맥과 윈도우가 디렉토리를 구분하는 기호가 다르기 때문에 `join` 을 권장한다.

pathlib 모듈을 사용해서 path를 객체로 다루기

```text
>>> import pathlib
>>>
>>> cwd = pathlib.Path.cwd()
>>> cwd
WindowsPath('D:/workspace')
>>> cwd.parent
WindowsPath('D:/')
>>> list(cwd.parents)
[WindowsPath('D:/')]
>>> list(cwd.glob("*"))
[WindowsPath('D:/workspace/ai-pnpp'),
WindowsPath('D:/workspace/cs50_auto_grader'),
WindowsPath('D:/workspace/data-academy'),
WindowsPath('D:/workspace/DSME-AI-SmartYard'),
WindowsPath('D:/workspace/introduction_to_python_TEAMLAB_MOOC'), 
```

파일이 존재하는지 확인

```python
if not os.path.exists("log/count_log.txt"):
    f = open("log/count_log.txt", 'w', encoding="utf8")
```



### Pickle

파이썬의 객체를 영속화 하는 built-in 객체. 데이터나 object등 실행중 정보를 저장하고 불러와서 사용할 수 있다. python 전용 binary 파일이라고 이해할 수 있다.

* 영속화는 파이썬에서 사용하는 객체를 파일로 저장해서 쓰는 것을 의미한다.

피클 파일 작성하기

```python
import pickle
f = open("list.pickle", "wb")
test = [1, 2, 3, 4, 5]
pickle.dump(test, f)
f.close()
```

* binary 파일이므로 'w' 대신 'wb'를 붙여줬다.

피클 불러오기

```python
f = open("list.pickle", "rb")
test_pickle = pickle.load(f)
print(test_pickle)
f.close()
```



### Loggin Handling

* 프로그램이 실행되는 동안 일어나는 정보를 기록을 남기기
* 유저의 접근, 프로그램의 Exception, 특정 함수의 사용
* Console 화면에 출력, 파일에 남기기, DB에 남기기 등등
* 기록된 로그를 분석하여 의미있는 결과를 도출 할 수 있음
* 실행시점에서 남겨야 하는 기록, 개발시점에서 남겨야하는 기록

logging 모듈

```python
import logging

logging.debug("틀렸잖아!")
logging.info("확인해")
logging.warning("조심해!")
logging.error("에러났어!!!")
logging.critical ("망했다...")
```

logging level

* debug : 개발시 처리 기록을 남겨야하는 로그 정보를 남김
* info : 처리가 진행되는 동안의 정보를 알림
* warning : 사용자가 잘못 입력한 정보나 처리는 가능하나 원래 개발시 의도치 않는 정보가 들어왔을 때 알림
* error : 잘못된 처리로 인해 에러가 났으나, 프로그램은 동작할 수 있음을 알림.
* critical : 잘못된 처리로 데이터 손실이나 더 이상 프로그램이 동작할 수 없음을 알림

```python
import logging
# Logger 선언
logger = logging.getLogger("main")
# Logger의 output 방법 선언
stream_hander = logging.StreamHandler()
# Logger의 output 등록
logger.addHandler(stream_hander)

logger.setLevel(logging.DEBUG)
logger.debug("틀렸잖아!")
logger.info("확인해")
logger.warning("조심해!")
logger.error("에러났어!!!")
logger.critical("망했다...")

logger.setLevel(logging.CRITICAL)
logger.debug("틀렸잖아!")
logger.info("확인해")
logger.warning("조심해!")
logger.error("에러났어!!!")
logger.critical("망했다...")
```

Logging formmater

* Log의 결과값의 포맷을 지정해 줄 수 있음

```python
formatter = logging.Formatter('%(asctime)s %(levelname)s %(process)d %(message)s')

2018-01-18 22:47:04,385 ERROR 4410 ERROR occurred
2018-01-18 22:47:22,458 ERROR 4439 ERROR occurred
2018-01-18 22:47:22,458 INFO 4439 HERE WE ARE
2018-01-18 22:47:24,680 ERROR 4443 ERROR occurred
2018-01-18 22:47:24,681 INFO 4443 HERE WE ARE
2018-01-18 22:47:24,970 ERROR 4445 ERROR occurred
2018-01-18 22:47:24,970 INFO 4445 HERE WE ARE
```

### 프로그램 설정

무엇을?

* 데이터 파일 위치, 파일 저장 장소, Operation Type 등

어떻게?

* configparser : 파일에
* argparser : 실행 시점에

configparser

* 프로그램의 실행 설정을 file에 저장함
* Section, Key, Value 값의 형태로 설정된 설정 파일을 사용
* 설정파일을 Dict Type으로 호출후 사용

```python
import configparser
config = configparser.ConfigParser()

config.sections()
config.read('example.cfg')
config.sections()

for key in config['SectionOne']:
    print(key)
```

argparser

* Console 창에서 프로그램 실행시 Setting 정보를 저장함
* 거의 모든 Console 기반 Python 프로그램 기본으로 제공
* 특수 모듈도 많이 존재하지만\(TF\), 일반적으로 argparse를 사용
* Command-Line Option 이라고 부름

```python
import argparse

parser = argparse.ArgumentParser(description='Sum two integers.')
# 짧은 이름, 긴 이름, 표시명, Help 설명, Argument Type
parser.add_argument('-a', "--a_value", dest=”A_value", help="A integers", type=int)
parser.add_argument('-b', "--b_value", dest=”B_value", help="B integers", type=int)

args = parser.parse_args()
print(args)
print(args.a)
print(args.b)
print(args.a + args.b)

>>> python arg_sum.py -a 10 -b 10
Namespace(a=10, b=10)
10
10
20
```



