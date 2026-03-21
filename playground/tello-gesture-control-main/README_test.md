# 환경 설정

## 가상환경 실행
```powershell
conda env create -n drone_gesture python=3.7
conda activate drone_gesture
```
## 의존성 설치
기존의 `requirements.txt`는 `mediapip == 0.8.2`를 요구.

하지만 0.8.2 버전은 구글에서 더 이상 제공 X
로컬에 `whl` 파일을 직접 받아 설치해야함 <= [링크](https://dashboard.stablebuild.com/pypi-deleted-packages/pkg/mediapipe/0.8.2)

***cp37***로 받아야함!!! (python 3.7버전 호환)

```powershell
pip install [PATH to .whl]
pip install -r requirements.txt
```

# 테스트
```powershell
python tests/webcam_gesture_test.py
```