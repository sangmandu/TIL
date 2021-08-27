# Day 5 : LRscheduler \| Facenet \| Save

## LRscheduler







## Facenet

현재 데이터셋은 가로 384, 세로 512의 크기를 가졌는데, GPU 리소스 문제때문에 이미지를 충분히 줄여야 한다.

* 이미지 크기가 크면 한번에 올릴 수 있는 이미지가 적어져 batch\_size가 작아진다
* 마찬가지로 num\_workers 역시 줄어들게된다.

이 때 이미지의 크기를 줄이는 것은 이미지의 정보가 상실될 수 있으므로 어느정도 성능 하락의 요인이 될 수 있다. 그러면 이미지 사이즈를 줄이면서 정보는 최대한 유지하려면 어떻게 해야할까? 바로 우리가 구별하고자 하는 대상만을 최대한 담는것이다.

![&#xC778;&#xBB3C; &#xC0AC;&#xC9C4; &#xC608;&#xC2DC;](../../../.gitbook/assets/image%20%281017%29.png)

위와 같이, 인물 사진에 대해서 상하좌우로 배경또는 옷이 존재하고 이 정보는 우리가 이 인물이 여성인지 남성인지, 또는 나이가 30대 이상인지 이하인지 구분하는데 필요하지 않다.

* 패션 스타일로 구분할 수는 있겠지만 주관적인 요소라는 것을 떠나 모델이 그런 센스가 있을까?

따라서 이 얼굴을 중심으로 이미지를 Crop 하려고 한다. 물론 `torchvision.transform` 에서 `CenterCrop` 을 할 수도 있다. 다만 주어진 데이터셋이 다음 이미지와 같이 얼굴이 항상 이미지의 가운데만 존재하지는 않았다.

![](../../../.gitbook/assets/image%20%281014%29.png)

따라서, 얼굴을 기준으로 나누는 것이 합리적이다. 그리고 이 때 모든 이미지에 대해 얼굴을 찾지 못할 수도 있으므로 이에 대한 대책을 마련해야 한다.

얼굴을 왜 찾지못했을까? 모든 사진은 전면을 바라보고 있고 얼굴이 잘리지 않았다. 예상할 수 있는 이유는 마스크때문이다. 보통의 face detector는 dlib 라이브러리의 landmarks를 사용한다. 이에 대한 설명은 여기서 볼 수 있다.

```python
dirs = os.listdir(TRAIN_IMAGE_DIR)
dirs = [item for item in dirs if item[0] == "0"]
dir_bar = tqdm(dirs)
length = len(dirs)
x = y = 0
for idx, d in enumerate(dir_bar):
    path = os.path.join(TRAIN_IMAGE_DIR, d)
    
    for image_path in glob(os.path.join(path, '*')):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
        boxes, probs = mtcnn.detect(image)
        if boxes is not None:
            x1, y1, x2, y2 = map(int, boxes[0])
            x1 = max(0, x1-30)
            y1 = max(0, y1-60)
            x2 = min(384, x2+30)
            y2 = min(512, y2+30)
            image = image[y1:y2, x1:x2]
            x += (x2-x1) 
            y += (y2-y1) 
        
    dir_bar.set_description(f'{idx} / {length}')
print(x / (7 * length), y / (7 * length))
```



![](../../../.gitbook/assets/image%20%281016%29.png)

![](../../../.gitbook/assets/image%20%281015%29.png)





## Save

