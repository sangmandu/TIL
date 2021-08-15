---
description: '210807'
---

# \[필수 과제 5\] Morsecode

### 사전 세팅

본문에는 `encoding_dict` 만 존재한다. decoding할 때 필요한 dictionary로 필요하므로 이에 대한 함수를 정의해준다

#### Get\_decoding\_dict

```python
def get_decoding_dict():
    morse_code_dict = get_morse_code_dict()
    k = morse_code_dict.keys()
    v = morse_code_dict.values()
    decoding_dict = {}
    for x, y in zip(v, k):
        decoding_dict[x] = y
    return decoding_dict
```

* 모스코드 딕셔너리를 받아 리버스하는 코드이다
  * 나중에 알았는데, reversed를 쓰면 가볍게 딕셔너리 관계를 바꿀 수 있었다



### Is\_validated\_english\_sentence

```python
def is_validated_english_sentence(user_input):
    for exp in '0123456789_@#$%^&*()-+=[]{}\"\';:\|`~':
        if exp in user_input:
            return False
    
    return True if len([word for word in user_input if word not in ' .,!?']) else False
```

* 숫자나, 정의된 특수문자가 포함되어 있으면 False 반환
* 문장부호를 제외했을 때 빈칸이면 False 반환



### Is\_validated\_morse\_code

```python
def is_validated_morse_code(user_input):
    if set(user_input) - set(["-", ".", " "]):
        return False

    decoding_dict = get_decoding_dict()
    for word in user_input.split():
        if word not in decoding_dict.keys():
            return False
    return True
```

* 정해진 모스기호 외 다른 글자가 있으면 False
* 사전에 미리 생성한 디코딩 함수로 존재하는 모스코드인지에 대한 key 여부를 조사한다



### Get\_cleaned\_english\_sentence

```python
def get_cleaned_english_sentence(raw_english_sentence):
    result = ''.join([word for word in raw_english_sentence if word not in '.,!?'])
    result = ' '.join(result.split())
    return result
```



### Decoding\_character

```python
def decoding_character(morse_character):
    decoding_dict = get_decoding_dict()
    return decoding_dict[morse_character]
```



### Encoding\_character

```python
def encoding_character(english_character):
    morse_code_dict = get_morse_code_dict()
    return morse_code_dict[english_character.upper()]
```



### Decoding\_sentence

```python
def decoding_sentence(morse_sentence):
    return ''.join([decoding_character(word) if word else ' ' for word in morse_sentence.split(' ')])
```



### Encoding\_sentence

```python
def encoding_sentence(english_sentence):
    return ' '.join(['' if word == ' ' else encoding_character(word) for word in get_cleaned_english_sentence(english_sentence)])
```



### Main

```python
def main():
    print("Morse Code Program!!")
    while True:
        codes = input("Input your message (H - Help, 0 - Exit) : ")
        if codes == "0":
            break
        if is_help_command(codes):
            print(get_help_message())
        elif is_validated_morse_code(codes):
            print(decoding_sentence(codes))
        elif is_validated_english_sentence(codes):
            print(encoding_sentence(codes))
        else:
            print("Wrong Input")
        continue  
    print("Good Bye")
    print("Morse Code Program Finished!!")

if __name__ == "__main__":
    main()
```



