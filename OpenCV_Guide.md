# OpenCV 개요 및 기초 정리

## OpenCV란?

OpenCV (Open Source Computer Vision Library)는 실시간 컴퓨터 비전을 위한 오픈소스 라이브러리입니다.  이미지 및 비디오 처리, 얼굴 인식, 객체 추적, 영상 필터링 등 다양한 비전 관련 기능을 지원합니다.

- **지원 언어** : Python, C++, Java 등
- **지원 플랫폼** : Windows, macOS, Linux, Android, iOS
- **활용 분야** : 자율주행, 의료 영상 분석, 로보틱스, AR, 이미지 복원 및 개선, OCR 등

---

### 설치 방법 (Python 기준)

```
pip install opencv-python
```

### 자주 사용하는 라이브러리
| 라이브러리           | 설명                                           | 설치 |
| ------------------- | ---------------------------------------------- | ----------------------- |
| `opencv-python`      | OpenCV의 핵심 기능 사용 (이미지/영상 처리 등)               | `pip install opencv-python` |
| `numpy`              | 이미지의 배열/행렬 연산 처리 (내부적 numpy 배열 사용) | `pip install numpy`        |
| `matplotlib`         | 이미지 시각화에 사용 (선택사항)                           | `pip install matplotlib`   |
| `scipy`              | 고급 수학 및 과학 연산에 사용                       | `pip install scipy`        |

### 기본 용어 정리
| 용어                           | 설명                                                            |
| ------------------------------ | --------------------------------------------------------------- |
| **BGR**                         | OpenCV는 이미지를 기본적으로 **Blue-Green-Red** 순서로 표현 (일반 RGB와 순서가 다름) |
| **Grayscale**                   | 흑백 이미지 (1채널 이미지)                                      |
| **Channel**                     | 이미지의 색상 구성 요소 (예: B, G, R은 각각 1채널)              |
| **ROI (Region of Interest)**    | 관심 영역 – 이미지의 특정 일부분을 선택하여 처리              |
| **Kernel (커널)**               | 필터 연산을 위한 작은 행렬 (예: 블러, 샤프닝, 엣지 감지 등)       |
| **Thresholding**                | 픽셀 값을 기준으로 이진화(흑/백) 처리                          |
| **Contour**                     | 이미지 내 객체의 외곽선 경계선                                 |
| **Cascade Classifier**          | 얼굴/객체 인식에서 사용되는 학습 기반 감지 방법 (Haar 등)       |
| **Morphological Operations**    | 침식(Erosion), 팽창(Dilation) 등의 이미지 형태학 연산           |
| **Affine Transform**            | 회전, 이동, 확대/축소 등의 기하학적 변환                        |
| **Perspective Transform**       | 원근 변환을 사용하여 이미지의 시점을 변경                      |
| **Histogram Equalization**      | 이미지의 밝기 분포를 개선하기 위한 기법                        |
| **Image Blurring**              | 노이즈 제거 및 부드러운 이미지를 만들기 위한 필터 기법          |
| **Edge Detection**              | 이미지 내에서 객체의 경계를 감지하는 기법 (예: Canny edge detector) |

---

## 🖼️ 기본 코드 예제

### 이미지 읽기 및 변환
```python
import cv2
import numpy as np

# 이미지 읽기
img = cv2.imread('image.jpg')  # 이미지 경로를 지정

# 크기 조절
resized = cv2.resize(img, (300, 300))

# 그레이스케일 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 이미지 출력
cv2.imshow('Original', img)
cv2.imshow('Gray', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 이미지 필터링
```python
# 이미지 블러링 (GaussianBlur 사용)
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# 엣지 감지 (Canny)
edges = cv2.Canny(gray, 100, 200)

# 결과 출력
cv2.imshow('Blurred Image', blurred)
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 얼굴 인식 (Haar Cascade 사용)
```python
# Haar Cascade Classifier 로 얼굴 인식
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 얼굴 인식
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# 얼굴 인식 결과 출력
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow('Face Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 이미지 저장
```python
# 이미지를 새로운 파일로 저장
cv2.imwrite('output_image.jpg', img)
```

---
