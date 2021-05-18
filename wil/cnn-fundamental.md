---
description: 'https://inf.run/Qf6B'
---

# 딥러닝 CNN 완벽 가이드 - Fundamental 편

### 강의 소개 및 실습 환경

강의 소개

실습 환경 설명 - 캐글 Notebook 이용하기

강의 교재와 실습 자료 다운로드

### 딥러닝 기반 이해 - 딥러닝 개요와 경사 하강법

머신러닝의 이해

딥러닝 개요

딥러닝의 장단점과 특징

퍼셉트론의 개요와 학습 메커니즘 이해

회귀 개요와 RSS, MSE의 이해

경사하강법의 이해

경사하강법을 이용하여 선형회귀 구현하기 - 01

경사하강법을 이용하여 선형회귀 구현하기 - 02

확률적 경사하강법\(Stochastic Gradient Descent\)와 미니 배치\(Mini-batch\) 경사하강법 이해

확률적 경사하강법\(Stochastic Gradient Descent\) 구현하기

미니 배치\(Mini-batch\) 경사하강법 구현하기

경사하강법의 주요 문제

### 딥러닝 기반 이해 - 오차 역전파, 활성화 함수, 손실 함수, 옵티마이저

심층신경망의 이해와 오차 역전파\(Backpropagation\) 개요

오차 역전파\(Backpropagation\)의 이해 - 미분의 연쇄 법칙

합성 함수의 연쇄 결합이 적용된 심층 신경망

오차 역전파\(Backpropagation\)의 Gradient 적용 메커니즘 - 01

오차 역전파\(Backpropagation\)의 Gradient 적용 메커니즘 - 02

활성화 함수\(Activation Function\)의 이해

Tensorflow Playground에서 딥러닝 모델의 학습 메커니즘 정리해보기

손실\(Loss\) 함수의 이해와 크로스 엔트로피\(Cross Entropy\) 상세 - 01

손실\(Loss\) 함수의 이해와 크로스 엔트로피\(Cross Entropy\) 상세 - 02

옵티마이저\(Optimizer\)의 이해 - Momentum, AdaGrad

옵티마이저\(Optimizer\)의 이해 - RMSProp, Adam

### Keras Framework

Tensorflow 2.X 와 tf.keras 소개

이미지 배열의 이해

Dense Layer로 Fashion MNIST 예측 모델 구현하기 - 이미지 데이터 확인 및 사전 데이터 처리

Dense Layer로 Fashion MNIST 예측 모델 구현하기 - 모델 설계 및 학습 수행

Keras Layer API 개요

Dense Layer로 Fashion MNIST 예측 모델 구현하기 - 예측 및 성능 평가

Dense Layer로 Fashion MNIST 예측 모델 구현하기 - 검증 데이터를 활용하여 학습 수행

Functional API 이용하여 모델 만들기

Functional API 구조 이해하기 - 01

Functional API 구조 이해하기 - 02

Dense Layer로 Fashion MNIST 예측 모델 Live Coding 으로 구현 정리 - 01

Dense Layer로 Fashion MNIST 예측 모델 Live Coding 으로 구현 정리 - 02

Keras Callback 개요

Keras Callback 실습 - ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

Numpy array와 Tensor 차이, 그리고 fit\(\) 메소드 상세 설명

### CNN의 이해

Dense Layer기반 Image 분류의 문제점

Feature Extractor와 CNN 개요

컨볼루션\(Convolution\) 연산 이해

커널\(Kernel\)과 피처맵\(Feature Map\)

스트라이드\(Stride\)와 패딩\(Padding\)

풀링\(Pooling\)

Keras를 이용한 Conv2D와 Pooling 적용 실습 01

Keras를 이용한 Conv2D와 Pooling 적용 실습 02

CNN을 이용하여 Fashion MNIST 예측 모델 구현하기

다채널 입력 데이터의 Convolution 적용 이해 - 01

다채널 입력 데이터의 Convolution 적용 이해 - 02

컨볼루션\(Convolution\) 적용 시 출력 피처맵의 크기 계산 공식 이해

### CNN 모델 구현 및 성능 향상 기본 기법 적용하기

CIFAR10 데이터세트를 이용하여 CNN 모델 구현 실습 - 01

CIFAR10 데이터세트를 이용하여 CNN 모델 구현 실습 - 02

CIFAR10 데이터세트를 이용하여 CNN 모델 구현 실습 - 03

가중치 초기화\(Weight Initialization\)의 이해와 적용 - 01

가중치 초기화\(Weight Initialization\)의 이해와 적용 - 02

배치 정규화\(Batch Normalization\) 이해와 적용 - 01

배치 정규화\(Batch Normalization\) 이해와 적용 - 02

학습 데이터 Shuffle 적용 유무에 따른 모델 성능 비교

배치크기 변경에 따른 모델 성능 비교

학습율\(Learning Rate\) 동적 변경에 따른 모델 성능 비교

필터수와 층\(Layer\) 깊이 변경에 따른 모델 성능 비교

Global Average Pooling의 이해와 적용

가중치 규제\(Weight Regularization\)의 이해와 적용

### 데이터 증강의 이해 - Keras ImageDataGenerator 활용

데이터 증강\(Data Augmentation\)의 이해

Keras의 ImageDataGenerator 특징

ImageDataGenerator로 Augmentation 적용 - 01

ImageDataGenerator로 Augmentation 적용 - 02

CIFAR10 데이터 셋에 Augmentation 적용 후 모델 성능 비교 - 01

CIFAR10 데이터 셋에 Augmentation 적용 후 모델 성능 비교 - 02

### 사전 훈련 CNN 모델의 활용과 Keras Generator 메커니즘 이해

사전 훈련 모델\(Pretrained Model\)의 이해와 전이학습\(Transfer Learning\) 개요

사전 훈련 모델 VGG16을 이용하여 CIFAR10 학습 모델 구현 후 모델 성능 비교

사전 훈련 모델 Xception을 이용하여 CIFAR10 학습 모델 구현 후 모델 성능 비교

개와 고양이\(Cat and Dog\) 이미지 분류 개요 및 파이썬 기반 주요 이미지 라이브러리 소개

개와 고양이 데이터 세트 구성 확인 및 메타 정보 생성하기

Keras Generator 기반의 Preprocessing과 Data Loading 메커니즘 이해하기 - 01

Keras Generator 기반의 Preprocessing과 Data Loading 메커니즘 이해하기 - 02

flow\_from\_directory\(\) 이용하여 개와 고양이 판별 모델 학습 및 평가 수행

flow\_from\_dataframe\(\) 이용하여 개와 고양이 판별 모델 학습 및 평가 수행

이미지 픽셀값의 Scaling 방법, tf 스타일? torch 스타일?

### Albumentation을 이용한 Augmentation기법과 Keras Sequence 활용하기

데이터 증강\(Augmentation\) 전용 패키지인 Albumentations 소개

Albumentations 사용 해보기\(Flip, Shift, Scale, Rotation 등\)

Albumentations 사용 해보기\(Crop, Bright, Contrast, HSV 등\)

Albumentations 사용 해보기\(Noise, Cutout, CLAHE, Blur, Oneof 등\)

Keras의 Sequence 클래스 이해와 활용 개요

Keras Sequence기반의 Dataset 직접 구현하기

Keras Sequence기반의 Dataset 활용하여 Albumentations 적용하고 Xception, MobileNet으로 이미지 분류 수행 - 01

Keras Sequence기반의 Dataset 활용하여 Albumentations 적용하고 Xception, MobileNet으로 이미지 분류 수행 - 02

### Advanced CNN 모델 파헤치기 - AlexNet, VGGNet, GoogLeNet

역대 주요 CNN 모델들의 경향과 특징

AlexNet의 개요와 구현 코드 이해

AlextNet 모델로 CIFAR10 학습 및 성능 테스트

VGGNet의 이해

VGGNet의 구조 상세 및 구현코드 이해하기

VGGNet16 모델 직접 구현하기

구현한 VGGNet16 모델로 CIFAR10 학습 및 성능 테스트

GoogLeNet\(Inception\) 개요

1x1 Convolution의 이해

GoogLeNet\(Inception\) 구조 상세

GoogLeNet\(Inception\) 구조 상세 및 구현 코드 이해

### Advanced CNN 모델 파헤치기 - ResNet 상세와 EfficientNet 개요

ResNet의 이해 - 깊은 신경망의 문제와 identity mapping

ResNet의 이해 - Residual Block

ResNet 아키텍처 구조 상세

ResNet50 모델 직접 구현하기 - 01

ResNet50 모델 직접 구현하기 - 02

구현한 ResNet50 모델로 CIFAR10 학습 및 성능 테스트

EfficientNet의 이해

EfficientNet 아키텍처

### 사전 훈련 모델의 미세 조정 학습과 다양한 Learning Rate Scheduler의 적용

사전 훈련 모델의 미세 조정\(Fine Tuning\) 학습 이해

사전 훈련 모델의 미세 조정\(Fine-Tuning\) 학습 수행하기

학습률\(Learning Rate\)를 동적으로 변경하는 Learning Rate Scheduler 개요

Keras LearningRateScheduler 콜백 적용하여 학습율 변경하기

Cosine Decay와 Cosine Decay Restart 기법 이해

Keras에서 Cosine Decay와 Cosine Decay Restart 적용하기

Ramp Up and Step Down Decay 이해와 Keras에서 적용하기

### 종합 실습 1 - 120종의 Dog Breed Identification 모델 최적화

Dog Breed Identification 데이터 세트 특징과 모델 최적화 개요

Dog Breed 데이터의 메타 DataFrame 생성 및 이미지 분석, Sequence 기반 Dataset 생성

Xception 모델 학습, 성능평가 및 예측 후 결과 분석하기

EfficientNetB0 모델 학습, 성능평가 및 분석

이미지 분류 모델 최적화 기법 - Augmentation과 Learning Rate 최적화 01

이미지 분류 모델 최적화 기법 - Augmentation과 Learning Rate 최적화 02

Pretrained 모델의 Fine-tuning을 통한 모델 최적화

Config Class 기반으로 함수 변경 후 EfficientNetB1 모델 학습 및 성능 평가

### 종합 실습 2 - 캐글 Plant Pathology\(나무잎 병 진단\) 경연 대회

Plant Pathology 캐글 경연대회 개요 및 데이터 세트 가공하기

Augmentation 적용 분석과 Sequence기반 Dataset 생성하기

Xception 모델 학습 후 Kaggle에 성능 평가 csv 파일 제출하기

이미지 크기 변경 후 Xception 모델 학습 및 성능 평가

EfficientNetB3와 B5 모델 학습 및 성능 평가

EfficientNetB7 모델 학습 및 성능 평가

