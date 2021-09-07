---
description: '210825'
---

# DAY 3 : DataSet/Lodaer \| EfficientNet

## Dataset/Loader

오늘의 미션은 데이터셋과 데이터로더를 구현하는 것. 사실 후기를 작성하는 시점에서는 어려운 부분이 없지만, 정말로 막상 처음 구현할 때는 너무 막막했다. 그만큼 내가 제대로 이해하지 못한거겠지.

일단 처음에는 캐글을 참고했다. 캐글 코드를 그대로 쓴 것은 아니고 어떠한 흐름으로 써지는 구나를 참고했다. 그러면서 알게된 부분은 CFG 라는 딕셔너리 변수에 하이퍼 파라미터를 모두 선언해놓는 방법이었다. 더 알아보니 이를 클래스로 선언하는 사람도 있었다.

### Config

```python
DATA_DIR = './input/data/train/images/'
CFG = {
    'fold_num': 5,
    'seed': 719,
    'epochs': 30,
    'train_bs': 30,
    'valid_bs': 60,
    'T_0': 10,
    'lr': 1e-5,
    'max_lr': 1e-3,
    'weight_decay':1e-6,
    'num_workers': 8,
    'accum_iter': 2, # suppoprt to do batch accumulation for backprop with effectively larger batch size
    'verbose_step': 1,
    'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
}
print(f'{CFG["device"]} is using!')
```

### Train\_test\_split

이후, Train data와 Valid data로 나누었다. 이 때는 `sklearn` 의 `train_test_split` 를 사용했다.

```python
data = pd.read_csv('./train_face.csv')
train_df, valid_df  = train_test_split(data, test_size=0.35, shuffle=True, stratify=data['label'], random_state=2021)
train_df.shape, valid_df.shape
```

* 인자로 data 하나만 주어지면 train\_data 와 valid\_data로 나누어 주며 인자로 data와 label이 주어지면 trainX, trainY, validX, validY로 나누어졌다. 나는 전자가 필요해서 data 인자 하나만 주었다.

### 

### Transform

```python
transform = transforms.Compose([
    transforms.CenterCrop((380, 380)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomChoice([transforms.ColorJitter(brightness=(0.2, 3)),
                             transforms.ColorJitter(contrast=(0.2, 3)),
                             transforms.ColorJitter(saturation=(0.2, 3)),
                             transforms.ColorJitter(hue=(-0.3, 0.3))
                            ]),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
```

* 이미지의 크기가 384 \* 512이다. 얼굴이 대부분 중앙에 위치하므로 CenterCrop을 사용하면 된다고 생각이 들었다. CenterCrop을 사용하는 이유는 다음과 같다.
  * 이미지 사이즈가 작을수록 GPU사용이 줄어든다. 실제로 원본 이미지를 입력했을 때는 batch size를 매우 작게 해야만 돌아갔다.
  * 사람의 얼굴 정보만 필요하다고 생각했다. 그 외에는 벽이나 옷등의 배경 이미지가 학습에 오히려 방해가 된다고 생각했다.
  * 다음 이미지를 참고 하면 알 수 있듯이 b4가 학습한 이미지의 크기는 380이다.

![https://github.com/lukemelas/EfficientNet-PyTorch/issues/42](../../../.gitbook/assets/image%20%281073%29.png)

* 그 외에 많은 Transform을 해주지는 않았다. RandomChoice를 사용해서 4개의 trsf 를 임의로 적용되도록 해주었고 이 안에 있는 변환은 밝기, 채도 등의 픽셀값 변환이다.
* 1/2 확률로 좌우반전이 일어나도록 했다.
* `ToTensor` 를 이용해 텐서로 변환되게 했고 정규화를 했다. 이 때 평균과 표준편차값은 train image의 평균과 표준편차 값을 아래와 같이 구했고 이를 상수로 계속 쓰도록 했다.
  * seed가 고정되어 있어서 계속 똑같은 train image로 고정된다.

```python
def get_img_stats(img_paths):
    img_info = dict(means=[], stds=[])
    for img_path in tqdm(img_paths):
        img = np.array(Image.open(glob(img_path)[0]))
        img_info['means'].append(img.mean(axis=(0,1)))
        img_info['stds'].append(img.std(axis=(0,1)))
    return img_info

img_stats = get_img_stats(train_df.path.values)
mean = np.mean(img_stats["means"], axis=0) / 255.
std = np.mean(img_stats["stds"], axis=0) / 255.
print(f'RGB Mean: {mean}')
print(f'RGB Standard Deviation: {std}')
```

* 단순히 이미지를 불러와서 모든 픽셀값을 합하고 이에대한 평균과 표준편차를 구하는 과정이다.



### Dataset

```python
class MaskDataset(Dataset):
    def __init__(self, df, transform=None):
        self.path = df['path']
        self.transform = transform
        self.label = df['label']
    
    def __len__(self):
        return len(self.path)
    
    def __getitem__(self, index):
        image = Image.open(self.path.iloc[index])
        if self.transform:
            image = self.transform(image)
        label = self.label.iloc[index]
        return image, torch.tensor(label)
    
class TestMaskDataset(Dataset):
    def __init__(self, df, transform=None):
        self.path = df['path']
        self.transform = transform
    
    def __len__(self):
        return len(self.path)
    
    def __getitem__(self, index):
        image = Image.open(self.path.iloc[index])
        if self.transform:
            image = self.transform(image)
        return image
```

* `init`
  * dataframe을 인자로 받고 그 안에 있는 특정 컬럼을 X와 y로 정했다. 여기서는 `path` 와 `label` 이다.
* `getitem`
  * `PIL` 패키지의 `Image` 라이브러리를 사용해서 이미지를 불러왔다. `cv2` 를 사용할 까 했지만 BGR로 읽어지고 이를 매번 convert 해줘야 해서 사용하지 않았다.
  * dataframe에서 index로 접근하려면 `iloc` 를 사용해야한다.
  * image는 transform에서 `ToTensor` 를 거치면서 tensor 형태가 되니까 그대로 반환해주고, label은 tensor로 캐스팅해준다.

```python
train_dataset = MaskDataset(df=train_df, transform=transform)
valid_dataset = MaskDataset(df=valid_df, transform=transform)
```

* 데이터셋을 생성한다.



### DataLoader

```python
train_loader = DataLoader(dataset = train_dataset,
                          batch_size=CFG['train_bs'],
                          shuffle=True,
                          num_workers=CFG['num_workers'],
                         )

valid_loader = DataLoader(dataset = valid_dataset,
                          batch_size=CFG['valid_bs'],
                          shuffle=False,
                          num_workers=CFG['num_workers'],
                         )
```

* 훈련 데이터의 배치 사이즈는 작게 했고 검증 데이터의 배치 사이즈는 2배로 설정했다,
  * 훈련 데이터의 배치 사이즈는 30 또는 60으로 결정했다.







## EfficientNet

사실, 여러 모델을 찾아보고 실험을 통해 결정하는 것이 맞지만, 여러 이유를 통해 EfficientNet을 제일 먼저 사용하게 되었다.

* 이전 1기 멤버의 포스팅을 참고하니 EfficientNet 사용
* 이미지넷 리더보드에서 EfficientNet이 3등이다.
  * 그래서 내일 1, 2등 모델인 ViT도 사용해볼 예정
* 멘토님의 추천

![](../../../.gitbook/assets/image%20%281066%29.png)

생각보다 모델을 불러오는 것은 쉬웠다.

```python
from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b4')
```

또, 모델들을 모아놓은 라이브러리 `timm` 을 사용해서 불러올 수도 있었다. efficientnet 모델의 종류는 굉장히 많다.

```python
import timm
timm.list_models('*eff*')
```

```text
['eca_efficientnet_b0',
 'efficientnet_b0',
 'efficientnet_b1',
 'efficientnet_b1_pruned',
 'efficientnet_b2',
 'efficientnet_b2_pruned',
 'efficientnet_b2a',
 'efficientnet_b3',
 'efficientnet_b3_pruned',
 'efficientnet_b3a',
 'efficientnet_b4',
 'efficientnet_b5',
 'efficientnet_b6',
 'efficientnet_b7',
 'efficientnet_b8',
 'efficientnet_cc_b0_4e',
 'efficientnet_cc_b0_8e',
 'efficientnet_cc_b1_8e',
 'efficientnet_el',
 'efficientnet_el_pruned',
 'efficientnet_em',
 'efficientnet_es',
 'efficientnet_es_pruned',
 'efficientnet_l2',
 'efficientnet_lite0',
 'efficientnet_lite1',
 'efficientnet_lite2',
 'efficientnet_lite3',
 'efficientnet_lite4',
 'efficientnetv2_l',
 'efficientnetv2_m',
 'efficientnetv2_rw_m',
 'efficientnetv2_rw_s',
 'efficientnetv2_s',
 'gc_efficientnet_b0',
 'tf_efficientnet_b0',
 'tf_efficientnet_b0_ap',
 'tf_efficientnet_b0_ns',
 'tf_efficientnet_b1',
 'tf_efficientnet_b1_ap',
 'tf_efficientnet_b1_ns',
 'tf_efficientnet_b2',
 'tf_efficientnet_b2_ap',
 'tf_efficientnet_b2_ns',
 'tf_efficientnet_b3',
 'tf_efficientnet_b3_ap',
 'tf_efficientnet_b3_ns',
 'tf_efficientnet_b4',
 'tf_efficientnet_b4_ap',
 'tf_efficientnet_b4_ns',
 'tf_efficientnet_b5',
 'tf_efficientnet_b5_ap',
 'tf_efficientnet_b5_ns',
 'tf_efficientnet_b6',
 'tf_efficientnet_b6_ap',
 'tf_efficientnet_b6_ns',
 'tf_efficientnet_b7',
 'tf_efficientnet_b7_ap',
 'tf_efficientnet_b7_ns',
 'tf_efficientnet_b8',
 'tf_efficientnet_b8_ap',
 'tf_efficientnet_cc_b0_4e',
 'tf_efficientnet_cc_b0_8e',
 'tf_efficientnet_cc_b1_8e',
 'tf_efficientnet_el',
 'tf_efficientnet_em',
 'tf_efficientnet_es',
 'tf_efficientnet_l2_ns',
 'tf_efficientnet_l2_ns_475',
 'tf_efficientnet_lite0',
 'tf_efficientnet_lite1',
 'tf_efficientnet_lite2',
 'tf_efficientnet_lite3',
 'tf_efficientnet_lite4',
 'tf_efficientnetv2_b0',
 'tf_efficientnetv2_b1',
 'tf_efficientnetv2_b2',
 'tf_efficientnetv2_b3',
 'tf_efficientnetv2_l',
 'tf_efficientnetv2_l_in21ft1k',
 'tf_efficientnetv2_l_in21k',
 'tf_efficientnetv2_m',
 'tf_efficientnetv2_m_in21ft1k',
 'tf_efficientnetv2_m_in21k',
 'tf_efficientnetv2_s',
 'tf_efficientnetv2_s_in21ft1k',
 'tf_efficientnetv2_s_in21k']
```

나는 이 중에서 `efficientnet_b4` 모델을 선택했다. 이유는 다음과 같다.

* `b7` 시리즈 부터는 V100 으로 돌릴 수가 없었다.
  * 이 당시 batch\_size는 30이다. 더 작게했으면 가능할 수도 있었겠지만, 15,000 장의 데이터셋을 여러 epoch를 돌려가며 확인한다고 하면 엄청난 시간이 소요될 것이기 때문에, 적어도 제일 마지막에 사용할 수 있는 방법이다.
* `b5` 시리즈 부터는 `pretrained=True` 인 모델이 없다. 즉 `b4` 까지만 `pretrained` 된 모델로 사용 가능했다.

실제로 pretrained의 힘은 엄청났는데, 장점은 다음과 같다.

* 7번 이하의 적은 epoch 수로도 train data가 완전히 학습되었다. 그만큼 적은 시간이 소요된다.
* 따라서, 적은 epoch 안에서 여러가지 실험이 가능했다. Loss 함수는 어떤것이 좋고, Lr Scheduler는 어떤 것이 좋고 등의 실험.
* 반대로 말하면, pretrained 된 모델이 아니라면 오버피팅 되는 시점의 epoch를 알기가 어려웠고, 매번 바뀔 가능성도 있다. 또, 시간도 많이 소요되어 어떤 실험을 하기가 어렵고, early stopping 같이 부수적인 부분을 추가했어야 했다.

사실, 한번 50~100 epoch 씩 돌리면서 학습 해보고 싶었지만, 제출에 대한 압박도 있었고, 여러 실험도 해보고 싶었다. 또, 성능을 높일 수 있는 다양한 테크닉을 해볼게 많다고 생각했다. 결과적으로는 아쉽지만 그래도 pretrained 된 모델을 사용해서 또 다양한 실험과 테크닉을 적용했기에 후회는 없다.

```python
from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=18)
model = model.to(CFG['device'])

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=CFG['lr'])

torch.cuda.empty_cache()
lrs = []

# for epoch in range(7):
for epoch in range(CFG['epochs']):
    model.train()
    train_batch_f1 = 0
    train_batch_accuracy = []
    train_batch_loss = []
    train_pbar = tqdm(train_loader)
    
    for n, (X, y) in enumerate(train_pbar):
        X, y = X.to(CFG['device']), y.to(CFG['device'])
        y_hat = model(X)
        loss = criterion(y_hat, y)
        pred = torch.argmax(y_hat, axis=1)
                            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lrs.append(optimizer.param_groups[0]["lr"])
        
        train_batch_accuracy.append(
            torch.sum(pred == y).cpu().numpy() / CFG['train_bs']
        )
        train_batch_loss.append(
            loss.item()
        )
        f1 = f1_score(y.cpu().numpy(), pred.cpu().numpy(), average='macro')
        train_batch_f1 += f1
        
        train_pbar.set_description(f'train : {n} / {len(train_loader)} | f1 : {f1:.5f} | accuracy : {train_batch_accuracy[-1]:.5f} | loss : {train_batch_loss[-1]:.5f}')
        
    model.eval()
    valid_batch_f1 = 0
    valid_batch_accuracy = []
    valid_batch_loss = []
    valid_pbar = tqdm(valid_loader)
    
    with torch.no_grad():
        for n, (X, y) in enumerate(valid_pbar):
            X, y = X.to(CFG['device']), y.to(CFG['device'])
            y_hat = model(X)
            loss = criterion(y_hat, y)
            pred = torch.argmax(y_hat, axis=1)
            
            valid_batch_accuracy.append(
                torch.sum(pred == y).cpu().numpy() / CFG['valid_bs']
            )
            valid_batch_loss.append(
                loss.item()
            )
            f1 = f1_score(y.cpu().numpy(), pred.cpu().numpy(), average='macro')
            valid_batch_f1 += f1
            
            valid_pbar.set_description(f'valid : {n} / {len(valid_loader)} | f1 : {f1:.5f} | accuracy : {valid_batch_accuracy[-1]:.5f} | loss : {valid_batch_loss[-1]:.5f}')


    print(f"""
epoch : {epoch+1:02d}
[train] f1 : {train_batch_f1/len(train_loader):.5f} | accuracy : {np.sum(train_batch_accuracy) / len(train_loader):.5f} | loss : {np.sum(train_batch_loss) / len(train_loader):.5f}
[valid] f1 : {valid_batch_f1/len(valid_loader):.5f} | accuracy : {np.sum(valid_batch_accuracy) / len(valid_loader):.5f} | loss : {np.sum(valid_batch_loss) / len(valid_loader):.5f}
""")
    
    if valid_batch_f1/len(valid_loader) >= 0.9:
        torch.save(model.state_dict(), f'v:f1_{valid_batch_f1/len(valid_loader):.3f}_t:f1_{train_batch_f1/len(train_loader):.5f}_efficientnet_b4_state_dict.pt')  # 모델 객체의 state_dict 저장
```

아무런 테크닉을 적용하지 않고 돌렸을 때의 f1 점수는 60점 중반 정도가 나왔다. 생각보다 점수가 낮네 싶었지만, 여러 테크닉을 고민해보고 있다.



