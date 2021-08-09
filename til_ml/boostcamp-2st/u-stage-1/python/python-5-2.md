---
description: '210804'
---

# \(Python 5-2강\) Python data handling

### Comma Separate Value

* CSV, 필드를 쉼표\(,\)로 구분한 텍스트 파일
* 엑셀 양식의 데이터를 프로그램에 상관없이 쓰기 위한 데이터 형식
* 탭\(TSV\), 빈칸\(SSV\) 등으로 구분해서 만들기도 함
* 통칭하여 character-separated values \(CSV\) 부름
* 엑셀에서는 “다름 이름 저장” 기능으로 사용 가능

csv 파일 읽기

```python
line_counter = 0 #파일의 총 줄수를 세는 변수
data_header = [] #data의 필드값을 저장하는 list
customer_list = [] #cutomer 개별 List를 저장하는 List
with open ("customers.csv") as customer_data: #customer.csv 파일을 customer_data 객체에 저장
    while True:
        data = customer_data.readline() #customer.csv에 한줄씩 data 변수에 저장
        if not data: break #데이터가 없을 때, Loop 종료
        if line_counter==0: #첫번째 데이터는 데이터의 필드
            data_header = data.split(",") #데이터의 필드는 data_header List에 저장, 데이터 저장시 “,”로 분리
        else:
            customer_list.append(data.split(",")) #일반 데이터는 customer_list 객체에 저장, 데이터 저장시 “,”로 분리
        line_counter += 1
    
print("Header :\t", data_header) #데이터 필드 값 출력
for i in range(0,10): #데이터 출력 (샘플 10개만)
    print ("Data",i,":\t\t",customer_list[i])
print (len(customer_list)) #전체 데이터 크기 출력
```

유의 사항

* Text 파일 형태로 데이터 처리시 문장 내에 들어가 있는“,” 등에 대해 전처리 과정이 필요
*  파이썬에서는 간단히CSV파일을 처리하기 위해 csv 객체를제공함
* 예제 데이터: korea\_foot\_traffic\_data.csv \(from [http://www.data.go.kr](http://www.data.go.kr)\)
* 예제 데이터는 국내 주요 상권의 유동인구 현황 정보 -한글로 되어 있어 한글 처리가 필요

CSV 객체 활용

```python
import csv
reader = csv.reader(f,
delimiter=',', quotechar='"',
quoting=csv.QUOTE_ALL)
```

* delimiter : 글자를 나누는 기준, 기본값은 ,
* lineterminator : 줄 바꿈 기준, 기본값은 \r\n
* quotechar : 문자열을 둘러싸는 신호 문자, 기본값은 "
* quoting : 데이터 나누는 기준이 quotechar에 의해 둘러싸인 레벨, 기본값은 QUOTE\_MINIMAL



### Web

* World Wide Web\(WWW\), 줄여서 웹이라고 부름
* 우리가 늘 쓰는 인터넷 공간의 정식 명칭
* 팀 버너스리에 의해 1989년 처음 제안되었으며, 원래는 물리학자들간 정보 교환을 위해 사용됨
* 데이터 송수신을 위한 HTTP 프로토콜 사용, 데이터를 표시하기 위해 HTML 형식을 사용

동작 과정

1. 요청 : 웹주소, Form, Header 등
2. 처리 : Database 처리 등 요청 대응
3. 응답 : HTML, XML 등으로 결과 반환
4. 렌더링 : HTML, XML 표시



### HTML

* Hyper Text Markup Language
* 웹 상의 정보를 구조적으로 표현하기 위한 언어
* 제목, 단락, 링크 등 요소 표시를 위해 Tag를 사용
* 모든 요소들은 꺾쇠 괄호 안에 둘러 쌓여 있음

   Hello, World ＃제목 요소, 값은 Hello, World

* 모든 HTML은 트리 모양의 포함관계를 가짐
* 일반적으로 웹 페이지의 HTML 소스파일은

  컴퓨터가 다운로드 받은 후 웹 브라우저가 해석/표시



### 정규식

정규 표현식, regexp 또는 regex 등으로 불림

* 복잡한 문자열 패턴을 정의하는 문자 표현 공식
* 특정한 규칙을 가진 문자열의 집합을 추출

```text
010-0000-0000 ^\d{3}-\d{4}-\d{4}$
203.252.101.40 ^\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}$
```

연습하기

1\)정규식연습장\([http://www.regexr.com/](http://www.regexr.com/)\) 으로이동

2\)테스트하고싶은문서를Text 란에삽입

3\)정규식을사용해서찾아보기



### XML

* 데이터의 구조와 의미를 설명하는 TAG\(MarkUp\)를 사용하여 표시하는 언어
* TAG와 TAG사이에 값이 표시되고, 구조적인 정보를 표현할 수 있음
* HTML과 문법이 비슷, 대표적인 데이터 저장 방식
* XML도 HTML과 같이 구조적 markup 언어
* 정규표현식으로 Parsing이 가능함
* 그러나 좀 더 손쉬운 도구들이 개발되어 있음
* 가장 많이 쓰이는 parser인 beautifulsoup으로 파싱

예제

```markup
<?xml version="1.0"?>
<고양이>
    <이름>나비</이름>
    <품종>샴</품종>
    <나이>6</나이>
    <중성화>예</중성화>
    <발톱 제거>아니요</발톱 제거>
    <등록 번호>Izz138bod</등록 번호>
    <소유자>이강주</소유자>
</고양이>
```



BeautifulSoup

* HTML, XML등 Markup 언어 Scraping을 위한 대표적인 도구
* [https://www.crummy.com/software/BeautifulSoup/](https://www.crummy.com/software/BeautifulSoup/)
* lxml 과 html5lib 과 같은 Parser를 사용함
* 속도는 상대적으로 느리나 간편히 사용할 수 있음

```python
# 모듈 호출
from bs4 import BeautifulSoup

# 객체 생성
soup = BeautifulSoup(books_xml, "lxml")

# Tag 찾는 함수 find_all 생성
soup.find_all("author")
```



### JSON

* JavaScript Object Notation
* 원래 웹 언어인 Java Script의 데이터 객체 표현 방식
* 간결성으로 기계/인간이 모두 이해하기 편함
* 데이터 용량이 적고, Code로의 전환이 쉬움
* 이로 인해 XML의 대체제로 많이 활용되고 있음
* Python의 Dict Type과 유사, key:value 쌍으로 데이터 표시
* json 모듈을 사용하여 손 쉽게 파싱 및 저장 가능
* 데이터 저장 및 읽기는 dict type과 상호 호환 가능
* 웹에서 제공하는 API는 대부분 정보 교환 시 JSON 활용
* 페이스북, 트위터, Github 등 거의 모든 사이트
* 각 사이트 마다 Developer API의 활용법을 찾아 사용

XML과 JSON 비교

```markup
<?xml version="1.0" encoding="UTF8" ?>
<employees>
    <name>Shyam</name>
    <email>shyamjaiswal@gmail.com</e
    mail> </employees>
    <employees>
    <name>Bob</name>
    <email>bob32@gmail.com</email>
    </employees> <employees>
    <name>Jai</name>
    <email>jai87@gmail.com</email>
</employees>
```

JSON

```javascript
{"employees":[
    {"name":"Shyam",
    "email":"shyamjaiswal@gmail.com"},
    {"name":"Bob",
    "email":"bob32@gmail.com"},
    {"name":"Jai",
    "email":"jai87@gmail.com"} ]
} 
```

JSON 데이터 읽기

```python
import json
with open("json_example.json", "r", encoding="utf8") as f:
    contents = f.read()
    json_data = json.loads(contents)
    print(json_data["employees"])
```

JSON 데이터 쓰기

```python
import json
dict_data = {'Name': 'Zara', 'Age': 7, 'Class': 'First'}
with open("data.json", "w") as f:
    json.dump(dict_data, f)
```

