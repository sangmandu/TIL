---
description: TIL
---

# 30 Wed

## \[프로그래머스 AI 스쿨 1기\] 4주차 DAY 1

### DAY 1 실습

#### flask에서 **CRUD** 구현하기 

```python
from flask import Flask, jsonify, request
from queue import PriorityQueue

app = Flask(__name__)

menus = [
    {"id":1, "name":"Espresso", "price":3800},
    {"id":2, "name":"Americano", "price":4100},
    {"id":3, "name":"CafeLatte", "price":4600},
]

id_list = PriorityQueue()
for available_id in range(4, 10000): id_list.put(available_id)

# @ = python decorator
# => 다음 주소를 입력받았을 때 아래 함수를 실행하라는 뜻
@app.route('/')
def hello_flask():
    return "Hello World!"

# GET /menus | 자료를 가지고 온다.
@app.route('/menus') # GET은 methods 생략 가능
def get_menus():
    return jsonify({"menus" : menus})

# POST /menus | 자료를 자원에 추가한다.
@app.route('/menus', methods=['POST'])
def create_menu():
    # 전달 받은 자료를 menus 자원에 추가
    # request가 JSON이라고 가정
    request_data = request.get_json() # {"name" : ..., "price" : ...}
    using_id = id_list.get()
    new_menu = {
        "id" : using_id,
        "name" : request_data['name'],
        "price" : request_data['price'],
    }
    menus.append(new_menu)
    return jsonify(new_menu)

@app.route('/menus/<int:id>', methods=['PUT'])
def update_menu(id):
    for idx, menu in enumerate(menus):
        if id in menu.values():
            request_data = request.get_json()
            menus[id-1]["name"] = request_data['name']
            menus[id-1]["price"] = request_data['price']
        return jsonify(menus[id-1])
    return "not existed id"

@app.route('/menus/<int:id>', methods=['DELETE'])
def delete_menu(id):
    for idx, menu in enumerate(menus):
        if id in menu.values():
            id_list.put(menus[idx]["id"])
            menus.pop(idx)
            return "delete successfully"
    return "not existed id"

if __name__ == '__main__':
    app.run()
```

 Flask를 이용해서 CRUD를 구현했다. 여기서 언급할점은 id의 생성과 삭제인데, 매번 id가 4로 고정되는 문제가 있었다.  \(보너스 과제 1에 해당\) 이를 해결하기 위해 우선순위 큐를 사용했다. 삭제되는 번호는 이 큐에 추가되며 새로운 번호를 발급받을 때는 작은 수부터 받을 수 있도록.

![](../../.gitbook/assets/image%20%2891%29.png)

 소스코드를 수정할 때마다 매번 flask를 다시 시작했어야 했는데, 다음과 같은 설정을 거치면 자동으로 flask가 리부팅된다.

```text
set FLASK_ENV=development
flask run
```



#### flask에서 **CRUD** 구현하기 with **MYSQL**

\*\*\*\*



