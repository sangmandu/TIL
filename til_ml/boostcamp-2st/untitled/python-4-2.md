---
description: '210804'
---

# \(Python 4-2강\) Module and Project

파이썬은 대부분의 라이브러리가 이미 다른 사용자에 의해 구현되어 있다!

### 모듈

* 어떤 대상의 부분 혹은 조각을 모듈이라고 한다
* 프로그램에서는 작은 프로그램 조각들과 모듈들을 모아서 하나의 큰 프로그램을 개발한다
* 이미 파이썬 내부에도 많은 모듈이 있다
  * Built In Module

패키지

* 모듈을 모아놓은 단위, 하나의 프로그램

파이썬에서 모듈은 py 파일을 의미한다. 그리고 import문을 사용해서 module을 호출한다.

* import를 할 때는 기본적으로 같은 폴더에 있어야 한다.



#### \_\_pycache\_\_ 가 뭐야?

파이썬 인터프리터가 미리 기계어로 번역한 것. 프로그램을 로딩할 때 빠르게 할 수 있도록 한다.



모듈을 호출할 때는 범위를 정할 수 있다. 일반적으로 py 파일을 import 하면 py 파일안에 모든 코드가 다 로딩이 된다.

* 모든 코드를 로딩하므로 시간이 걸릴 수도 있다
* 불러오려는 함수 또는 클래스를 매번 작성하기 힘들다.
* 실행되면 안되는 코드까지 실행될 수 있다
  * 그래서 if \_\_name\_\_ == \_\_main\_\_ 을 쓰곤 한다.

이 때 사용하는 것이 namespace!

* Alias 설정하기 - 모듈명을 별칭으로 쓰기

```python
import fah_converter as fah
```

* 모듈에서 특정 함수 또는 클래스만 호출하기

```python
from fah_converter import covert_c_to_f
```

* 모듈에서 모든 함수 또는 클래스를 호출하기

```python
from fah_converter import *
```



Built-in Module

* random, time, urllib.request 등이 있다.



수많은 파이썬 모듈을 어떻게 활용할까?

* 구글링
* 모듈을 import 하고 구글링 또는 Help
* 공식 문서



### 패키지

하나의 대형 프로젝트를 만드는 코드의 묶음이다. 다양한 모듈들이 모여있는 폴더 단위로 존재한다.

\_\_init\_\_, \_\_main\_\_ 등 키워드 파일명이 사용되며 다양한 오픈소스들이 패키지로 관리된다.

* 3.3 버전 이전에는 \_\_init\_\_ 파일이 있어야 패키지로 간주되었다.
* 지금은 없어도 되지만 많이들 쓰는 편



package namespace

* package 내에서 다른 폴더의 모듈을 부를 때 상대 참조로 호출하는 방법

```python
# 절대참조
from game.graphic.render import render_test()

# . 현재디렉토리기준
from .render import render_test()

# .. 부모 디렉토리 기준
from ..sound.echo import echo_test()
```



### Python Virtual Environment

virtualenv + pip

* 가장 대표적인 가상환경 관리 도구
* 레퍼런스 + 패키지 개수

conda

* 사용 가상환경도구
* miniconda 기본도구
* 설치가 용이해서 윈도우에서 장점을 가진다

콘다 가상환경

```text
# 만들기
conda create -n my project python=3.8

# 호출
conda activate my_project

# 해제
conda deactivate

# 패키지 설치
conda install <패키지명>
```



