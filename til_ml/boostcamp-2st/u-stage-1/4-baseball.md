---
description: '210804'
---

# \[필수 과제 4\] Baseball

### 사전 세팅

* 없다



### Is\_digit

```python
def is_digit(user_input_number):
    result = user_input_number.isdigit()
    return result
```

* `isdigit` 를 이용해 자연수인지 판별한다.
  * `isdigit` 는 문자 뿐만 아니라 음수, 실수에 대해서도 `False` 를 반환한다.



### Is\_between\_100\_and\_999

```python
def is_between_100_and_999(user_input_number):
    result = 100 <= int(user_input_number) < 1000
    return result
```

* 주어진 수의 범위를 조사한다
  * python은 부등호의 동시 비교가 가능하다



### Is\_duplicated\_number

```python
def is_duplicated_number(three_digit):
    result = len(set(three_digit)) != 3
    return result
```

* set\(\) 을 이용해서 각 자리수가 중복되는지 검사한다.



### Is\_validated\_number

```python
def is_validated_number(user_input_number):
    result = is_digit(user_input_number) and is_between_100_and_999(user_input_number) and not is_duplicated_number(user_input_number)
    return result
```

* 지금까지 구성한 세 함수에 대해 만족하는지 `and` 를 사용해서 조사한다.
  * `is_duplicated_number` 는 중복되면 아니므로 `not`을 조사한다.



### Get\_not\_duplicated\_three\_digit\_number

```python
def get_not_duplicated_three_digit_number():
    while True:
        result = get_random_number()
        if is_validated_number(str(result)):
            break
    return result
```

* `get_random_number()` 함수로 임의의 수를 얻는다. 이 때 이 수가 유효하지 않으면 반복문을 통해서 다시 얻도록 한다.



### Get\_strikes\_or\_ball

```python
def get_strikes_or_ball(user_input_number, random_number):
    strikes = [a == b for a, b, in zip(user_input_number, random_number)].count(True)
    balls = [a in random_number[:idx]+random_number[idx+1:] for idx, a in enumerate(user_input_number)].count(True)
    result = [strikes, balls]
    return result
```

* strikes : 랜덤 넘버와 유저 입력을 `zip` 으로 비교해서 자리와 수를 동시에 검사한다
* balls : 유저 입력을 기준으로 랜덤 넘버의 각 위치를 제외한  수에 대해 `in` 을 사용해서 검사한다.



### Is\_yes / Is\_No

```python
def is_yes(one_more_input):
    result = one_more_input.lower() in ['y', 'yes']
    return result
    
def is_no(one_more_input):
    result = one_more_input.lower() in ['n', 'no']
    return result
```

* 주어진 문자를 대소문자 구분 없이 받아야 하므로 `lower()`
  * 소문자를 기준으로 검사할 것임
* 소문자에 해당하는 'y'와 'yes' 둘 중 하나에 해당하는지 `in` 을 사용해서 검사



### Main

```python
def main():
    print("Play Baseball")
    while True:
      reset = over = 0
      user_input = 999
      random_number = str(get_not_duplicated_three_digit_number())
      print("Random Number is : ", random_number)
      while True:
          user_input = input('Input guess number : ')
          if user_input == "0":
              over = 1
              break
          if not is_validated_number(user_input):
              print("Wrong Input")
              continue
          strike, ball = get_strikes_or_ball(user_input, random_number)
          print(f"{random_number} {user_input} {strike} Strikes, {ball} Balls")
          if strike == 3:
              user_input = input("restart? : ")
              while (is_yes(user_input) or is_no(user_input)) == False:
                  print("Wrong Input")
                  user_input = input("restart? : ")
              if is_yes(user_input):
                  reset = 1
                  break
              if is_no(user_input):
                  over = 1
                  break
      if reset:
          continue
      if over:
          break
    # ==================================
    print("Thank you for using this program")
    print("End of the Game")
```

`user_input` 에 대해 반복적으로 다음 조건을 수행한다

* `user_input == 0`인가?
  * 종료
* `not is_validated_number` : 유효한 수가 아닌가?
  * "Wrong Input" 출력 후 `user_input` 다시 입력받음
* `get_strikes_or_ball` 함수를 통해 strike와 ball 구하기
* `strike == 3` 인가?
  * 맞다 : 재시작에 대한 `user_input` 입력받음
    * `is_yes` 또는 `is_no` 에 해당하는가?
      * 해당한다 : 재시작 또는 종료
      * 해당하지 않는다 : 다시 입력받기
  * 아니다 : 반복문 재시작
    * 따로 작성할 코드는 없음









