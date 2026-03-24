# SLAM-Pipeline

![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.9-blue.svg)
![SLAM](https://img.shields.io/badge/SLAM-LiDAR--IMU-green.svg)

StrayScanner 데이터셋을 사용한 실시간 LiDAR-IMU SLAM 및 데이터 후처리 파이프라인

---

## 📋 목차

- [개요](#-개요)
- [주요 기능](#-주요-기능)
- [시스템 요구사항](#-시스템-요구사항)
- [설치 방법](#-설치-방법)
- [프로젝트 구조](#-프로젝트-구조)
- [사용법](#-사용법)
- [데이터 형식](#-데이터-형식)
- [출력 결과](#-출력-결과)
- [트러블슈팅](#-트러블슈팅)
- [기여](#-기여)
- [라이센스](#-라이센스)
- [참고 자료](#-참고-자료)

## 🎯 개요

**SLAM-Pipeline**은 StrayScanner 앱으로 수집한 LiDAR 및 IMU 데이터를 사용하여 실시간 SLAM(Simultaneous Localization and Mapping)을 수행하는 파이프라인입니다. 이 프로젝트는 SLAM-Dunk-Prometheus 시스템의 핵심 컴포넌트로, 3D 공간 매핑 및 로봇 위치 추정에 활용됩니다.

### 특징

- ✅ **실시간 LiDAR-IMU Fusion**: LiDAR 포인트 클라우드와 IMU 데이터를 융합한 고정밀 SLAM
- ✅ **StrayScanner 데이터 지원**: iPhone/iPad의 LiDAR 센서 데이터 활용
- ✅ **자동 데이터 후처리**: 포인트 클라우드 통합 및 최적화
- ✅ **시각화 도구**: SLAM 결과 실시간 시각화

## ✨ 주요 기능

### 1. 실시간 SLAM (`LiDAR_IMU_SLAM.py`)
- LiDAR 포인트 클라우드 기반 위치 추정
- IMU 데이터를 활용한 센서 융합
- Odometry 데이터 통합
- 실시간 3D 맵 생성

### 2. 데이터 후처리 (`post_processor.py`)
- 깊이 맵(Depth map) 전처리
- 포인트 클라우드 통합
- Confidence 필터링
- 데이터 품질 향상

## 💻 시스템 요구사항

### 하드웨어
- **CPU**: Intel i5 이상 또는 동급 (멀티코어 권장)
- **RAM**: 최소 8GB (16GB 권장)
- **Storage**: 데이터셋 크기에 따라 변동 (일반적으로 1-10GB)

### 소프트웨어
- **OS**: Linux (Ubuntu 20.04+), macOS, Windows 10/11
- **Python**: 3.9 이상
- **CUDA**: GPU 사용 시 CUDA 11.0+ (선택사항)

### 필수 라이브러리
```
numpy
opencv-python
open3d
scipy
pandas
matplotlib
```

## 🛠️ 설치 방법

### 1. 리포지토리 클론

```bash
git clone https://github.com/SLAM-Dunk-Prometheus/SLAM-Pipeline.git
cd SLAM-Pipeline
```

### 2. Conda 환경 생성 (권장)

```bash
# Conda 환경 생성
conda create -n slam-pipeline python=3.9
conda activate slam-pipeline
```

### 3. 의존성 설치

**방법 1: pip 사용**
```bash
pip install -r requirements.txt
```

**방법 2: 개발 모드 설치**
```bash
pip install -e .
```

### 4. 설치 확인

```bash
python -c "import numpy, cv2, open3d; print('설치 완료!')"
```

## 📁 프로젝트 구조

```
SLAM-Pipeline/
├── LiDAR_IMU_SLAM.py      # 메인 SLAM 실행 스크립트
├── post_processor.py      # 데이터 전처리 및 후처리 도구
├── requirements.txt       # Python 의존성 패키지
├── pyproject.toml        # 프로젝트 메타데이터
├── LICENSE               # Apache-2.0 라이센스
├── README.md            # 프로젝트 문서 (본 파일)
└── data/                # StrayScanner 데이터 디렉토리 (사용자가 준비)
    ├── rgb.mp4          # RGB 비디오 데이터
    ├── camera_matrix.csv # 카메라 내부 파라미터
    ├── odometry.csv     # 오도메트리 데이터
    ├── imu.csv          # IMU 센서 데이터
    ├── depth/           # 깊이 맵 디렉토리
    │   ├── 000000.npy
    │   ├── 000001.npy
    │   └── ...
    └── confidence/      # 깊이 신뢰도 맵 디렉토리
        ├── 000000.png
        ├── 000001.png
        └── ...
```

## 🚀 사용법

### 기본 실행

#### 1. SLAM 실행

```bash
# 기본 실행
python LiDAR_IMU_SLAM.py data/

# 상세 로그 출력
python LiDAR_IMU_SLAM.py data/ --verbose

# 특정 프레임 범위 지정
python LiDAR_IMU_SLAM.py data/ --start 0 --end 1000
```

#### 2. 데이터 전처리

```bash
# 기본 전처리
python post_processor.py data/

# 포인트 클라우드 통합
python post_processor.py data/ --integrate

# 다운샘플링 적용
python post_processor.py data/ --downsample 0.01
```

### 고급 사용법

#### Python 스크립트에서 사용

```python
from LiDAR_IMU_SLAM import SLAM
from post_processor import PostProcessor

# SLAM 초기화
slam = SLAM(data_path='data/')

# SLAM 실행
trajectory, point_cloud = slam.run()

# 후처리
processor = PostProcessor(data_path='data/')
processed_cloud = processor.integrate_point_clouds()

# 결과 저장
slam.save_results('output/')
```

#### 커스텀 파라미터 설정

```python
# SLAM 파라미터 설정
slam_params = {
    'voxel_size': 0.05,
    'max_correspondence_distance': 0.1,
    'imu_weight': 0.3
}

slam = SLAM(data_path='data/', params=slam_params)
slam.run()
```

## 📊 데이터 형식

### StrayScanner 데이터셋 구조

SLAM-Pipeline은 [StrayScanner](https://apps.apple.com/app/stray-scanner/id1557051662) 앱으로 수집한 데이터를 사용합니다.

#### 필수 파일

| 파일명 | 형식 | 설명 |
|--------|------|------|
| `rgb.mp4` | Video | RGB 카메라 영상 |
| `camera_matrix.csv` | CSV | 카메라 내부 파라미터 (fx, fy, cx, cy) |
| `odometry.csv` | CSV | 프레임별 오도메트리 (x, y, z, qw, qx, qy, qz) |
| `imu.csv` | CSV | IMU 센서 데이터 (가속도, 각속도) |
| `depth/*.npy` | NumPy | 각 프레임의 깊이 맵 |
| `confidence/*.png` | Image | 깊이 신뢰도 맵 |

#### 데이터 수집 가이드

1. iPhone/iPad에 StrayScanner 앱 설치
2. LiDAR 스캔 수행
3. 데이터 내보내기 (Export)
4. `data/` 디렉토리에 압축 해제

## 📈 출력 결과

### SLAM 출력

```
output/
├── trajectory.txt        # 카메라 궤적 (x, y, z, qw, qx, qy, qz)
├── point_cloud.pcd      # 통합 포인트 클라우드
├── map.ply              # 3D 맵 (색상 포함)
├── visualization.png    # 결과 시각화 이미지
└── metrics.json         # 성능 메트릭
```

### 시각화 예시

![SLAM Output](slam_output.png)

*SLAM 실행 결과: 3D 포인트 클라우드와 카메라 궤적*

### 성능 지표

- **처리 속도**: ~10-30 FPS (하드웨어에 따라 변동)
- **맵 해상도**: Voxel size 0.01-0.05m
- **궤적 정확도**: RMSE < 5cm (양질의 데이터 기준)

## 🔧 트러블슈팅

### 일반적인 문제

#### 1. ImportError: No module named 'open3d'

**해결방법:**
```bash
pip install open3d --upgrade
```

#### 2. CUDA Out of Memory

**해결방법:**
```bash
# CPU 모드로 실행
python LiDAR_IMU_SLAM.py data/ --device cpu

# 또는 배치 크기 줄이기
python LiDAR_IMU_SLAM.py data/ --batch-size 1
```

#### 3. 데이터 로드 실패

**확인사항:**
- `data/` 디렉토리 구조가 올바른지 확인
- 모든 필수 파일이 존재하는지 확인
- 파일 권한 확인 (`chmod -R 755 data/`)

#### 4. 메모리 부족 오류

**해결방법:**
```bash
# 다운샘플링 적용
python post_processor.py data/ --downsample 0.05

# 프레임 간격 증가
python LiDAR_IMU_SLAM.py data/ --frame-skip 2
```

### 로그 확인

```bash
# 디버그 모드 실행
python LiDAR_IMU_SLAM.py data/ --debug --log-file slam.log
```

## 🤝 기여

프로젝트에 기여를 환영합니다!

### 기여 방법

1. **Fork** 이 리포지토리
2. **Feature Branch** 생성 (`git checkout -b feature/AmazingFeature`)
3. **Commit** 변경사항 (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the Branch (`git push origin feature/AmazingFeature`)
5. **Pull Request** 생성

### 코딩 컨벤션

- PEP 8 스타일 가이드 준수
- 함수 및 클래스에 docstring 작성
- Type hints 사용 권장

## 📚 참고 자료

### 관련 논문

- **LOAM**: J. Zhang and S. Singh, "LOAM: Lidar Odometry and Mapping in Real-time"
- **FAST-LIO**: W. Xu and F. Zhang, "FAST-LIO: A Fast, Robust LiDAR-inertial Odometry Package"

### 관련 프로젝트

- [StrayScanner](https://apps.apple.com/app/stray-scanner/id1557051662) - iOS LiDAR 데이터 수집 앱
- [SLAM-Dunk-Prometheus](https://github.com/SLAM-Dunk-Prometheus) - 메인 프로젝트
- [Segmentation](https://github.com/SLAM-Dunk-Prometheus/Segmentation) - 3D 세그멘테이션 모듈


---

⭐ 이 프로젝트가 유용하다면 Star를 눌러주세요!
