---
description: TIL
---

# 25 Fri

## \[프로그래머스 AI 스쿨 1기\] 4주차 DAY 2

### AWS를 활용한 인공지능 모델 배포 II

 아마존 로그인

 계정을 생성하고 카드를 등록하면 12개월 프리티어 사용 가능.

 EC2 생성

 AMI 선택 : 딥러닝 AMI가 설치된 EC2를 생성하여 필요 개발 환경 사전 세팅 



## \[프로그래머스 AI 스쿨 1기\] 4주차 DAY 3

### django I

 Flask와는 또다른 웹 프레임워크 장고. Python 기반 웹 프레임워크이며 Pinterest나 Instagram은 장고로 만들어짐.

####  flask vs django

 방향성이 다르다.

 flask : micro라는 말이 많이 붙는다. 최소한의 베이스에서 조금씩 채워나간다. 작은 프로젝트에 적합.

 django : 모든 것이 내장되어 있다. 큰 프로젝트에 적합.

 가상환경 설치하기

`virtualenv venv`

`.\venv\Scripts\activate.bat`

`pip install django`

`django-admin startproject webproj`

`cd webproj`

`python manage.py runserver`

 `http://127.0.0.1:8000/` 로 이동 후 아래 화면이 뜨면 성공적으로 완

![](../../.gitbook/assets/image%20%2876%29.png)

####  django 구성요소

 manage.py : 장고를 실행하는 파일이며 실제로는 python - manage.py - runserver의 순서를 통해 실행된다.

![](../../.gitbook/assets/image%20%2877%29.png)

\_\_init\_\_.py : python 모듈로써 인식되게 하는 파일

asgi.py, wsgi.py : 장고에서 서버를 운용할 때 다루는 파일

settings.py : 전반적인 장고 프로젝트에 설정 파일을 반영

* secret key
* debug = True : python 프로젝트를 디버깅 모드로 실행 가능
* allowed\_hosts : 어떠한 주소에 대해서 장고프로젝트가 접근 가능
* installed\_apps :장고 프로젝트는 여러 앱으로 이루어져있다. 설치된 앱 목록
* middle\_ware
* root\_urlconf : url관리를 어떤 모듈에서 진행할 것인지
* templates : 실제 보는 화면에 관한 요소들이 담겨있음 
* wegi\_application : python상에서 웹서버와 소통할 때 필요한 어플리케이션을 담당 
* databases : 프로젝트 상에서 저장되는 데이터를 담당할 곳. default는 sqlite3
* auth\_password\_validation : 관리자가 패스워드를 관리하는 곳
* language\_code
* time\_zone
* use\_i18n
* use\_l10n
* use\_tz
* static\_url : css, js, images 등의 정적 파일들을 어느 폴더에 담아둘지 결정

=&gt; setting만 봐도 플라스크보다 많은 기능을 가지고 있음을 알 수 있다. 기능을 적절하게 활용하면 정말 빠르게 웹사이트를 구축할 수 있다.

urls.py : url을 관리 

* urlpatterns : path\('a', b\) =&gt; 'a' 라는 요청이 들어오면 이에 대한 응답은 b에서 담당한다.

#### django Project and App

 프로젝트는 여러가지 앱으로 구성된다. 앱은 특정 뷰나 템플릿의 모음.

ex\) 스포츠 앱, 블로그 앱 등 

#### django App 만들기

 이전과는 다르게 상위 폴더에서 호출하는 것이 아닌 해당 폴더에서 호출해야 한다.

`django-admin startapp homepage`

![](../../.gitbook/assets/image%20%2878%29.png)

`__init__.py`  : python 모듈로써 인식되게 하는 파일

`admin.py`  : admin 페이지에 관한 부

`apps__init__.py`  : 앱에 대한 설정을 관

`models.py`  : 홈페이지 모듈에서 쓰일 데이터베이스의 스키마를 클래스 형태로 정

`tests.py`  : 테스트 케이스 설명

`views.py`  : 뷰 관

django의 MVT Pattern

 디자인 패턴 : 코드의 모듈화를 이용해서 각 코드가 독립적으로 동작해서 유기적으로 원하는 목표를 달성할 수 있게 하는 구조

 장고는 MVT 패턴을 채택했다. Model View Template. MVC\(Controller\)를 바탕으로 장고만의 디자인 패턴 채택. 유저가 리퀘스트를 보내면 장고\(서버\)는 URL\(urls,py\)을 체크하여 어떤 경로로 요청이 왔는지 파악하고 View\(views.py\)에서 요청을 처리한다. 이 때 DB를 관리 및 소통을 Model에서 담당한다. 장고는 DB를 ORM 방식으로 관리한다. Object Relational Mapping. 쿼리를 통해 DB에 CRUD 접근 가능. 웹 페이지나 웹 문서를 보여주는 요청은 Template에서 관리하며 .html 파일 등으로 전달해줄 수 있다. 이 때 template 언어를 사용한다. 기본적으로 html은 로직이나 변수를 사용하는 행위는 할 수 없지만 template언어를 사용하면 html에 로직을 추가하는 행위를 할 수 있다. 

