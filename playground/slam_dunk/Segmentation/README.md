# Segmentation

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)

Mosaic3D와 OpenScene 모델을 활용한 3D Segmentation 및 Visualization 데모 프로젝트

## 📋 목차

- [개요](#개요)
- [주요 기능](#주요-기능)
- [설치 방법](#설치-방법)
- [사용법](#사용법)
- [프로젝트 구조](#프로젝트-구조)
- [데모](#데모)
- [참고 자료](#참고-자료)
- [기여](#기여)
- [라이센스](#라이센스)

## 🎯 개요

이 프로젝트는 **Mosaic3D**와 **OpenScene** 딥러닝 모델을 활용하여 3D 공간 데이터에 대한 세그멘테이션(Segmentation)을 수행하고, 그 결과를 시각화하는 데모 코드를 제공합니다. SLAM-Dunk-Prometheus 프로젝트의 일환으로, 3D 장면 이해 및 객체 인식에 활용됩니다.

### 주요 특징

- **Mosaic3D**: 3D 포인트 클라우드 데이터에 대한 고성능 세그멘테이션
- **OpenScene**: 오픈 어휘 기반 3D 장면 이해
- **시각화 도구**: 세그멘테이션 결과를 직관적으로 확인할 수 있는 visualization 기능

## ✨ 주요 기능

- 3D 포인트 클라우드 데이터 세그멘테이션
- Stray 데이터 포맷 변환 (`convert_stray.py`)
- 실시간 시각화 및 결과 분석
- 다양한 3D 장면 데이터셋 지원

## 🛠️ 설치 방법

### 요구사항

- Python 3.8 이상
- CUDA (GPU 사용 권장)
- 필요한 Python 패키지

### 설치 단계

1. 리포지토리 클론

```bash
git clone https://github.com/SLAM-Dunk-Prometheus/Segmentation.git
cd Segmentation
```

2. 가상 환경 생성 (선택사항)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

3. 의존성 패키지 설치

```bash
# 기본 설치 (convert_stray.py만 실행)
pip install -r requirements.txt

# 또는 pyproject.toml 사용
pip install -e .

# 전체 기능 설치 (Mosaic3D 포함)
pip install -e ".[full]"

# 개발 도구 포함
pip install -e ".[dev]"
```

4. Mosaic3D 모델 설정

```bash
cd mosaic3d
# 모델별 추가 설정이 필요한 경우 여기에 작성
```

## 🚀 사용법

### 기본 사용법

1. **Stray 데이터 변환**

```bash
python convert_stray.py --input <input_path> --output <output_path>
```

2. **세그멘테이션 실행**

```bash
# Mosaic3D를 이용한 세그멘테이션
cd mosaic3d
python run_segmentation.py --config <config_file>
```

3. **결과 시각화**

```bash
# 세그멘테이션 결과 시각화
python visualize.py --input <segmentation_result>
```

### 예제

```python
# 간단한 사용 예제
from mosaic3d import Segmentation

# 모델 초기화
model = Segmentation(model_type='mosaic3d')

# 데이터 로드 및 세그멘테이션 수행
result = model.segment(point_cloud_data)

# 결과 시각화
model.visualize(result)
```

## 📁 프로젝트 구조

```
Segmentation/
├── mosaic3d/              # Mosaic3D 모델 관련 코드
│   ├── models/           # 모델 아키텍처
│   ├── utils/            # 유틸리티 함수
│   └── configs/          # 설정 파일
├── convert_stray.py      # Stray 데이터 변환 스크립트
├── requirements.txt      # 의존성 패키지 목록
└── README.md            # 프로젝트 문서
```

## 🎬 데모

### 데모 영상

> 🎥 데모 영상은 곧 추가될 예정입니다!

### 실행 결과 예시

*(세그멘테이션 결과 이미지 또는 스크린샷을 여기에 추가할 수 있습니다)*

## 📚 참고 자료

### 관련 논문

- **Mosaic3D**: [https://arxiv.org/pdf/2502.02548]
- **OpenScene**: [https://arxiv.org/pdf/2211.15654)]

### 관련 프로젝트

- [SLAM-Dunk-Prometheus 메인 프로젝트](https://github.com/SLAM-Dunk-Prometheus)

## 🤝 기여

프로젝트에 대한 기여를 환영합니다! 기여 방법:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



---

⭐ 이 프로젝트가 유용하다면 Star를 눌러주세요!
