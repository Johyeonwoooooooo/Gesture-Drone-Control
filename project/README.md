# 언어 기반 3D 객체 위치 탐색

실내 3D 포인트 클라우드에서 자연어 쿼리에 해당하는 객체를 찾고, 해당 객체의 3D bounding box를 시각화하는 데모입니다.

UniDet3D로 3D 객체 후보를 검출하고, CLIP을 사용해 텍스트 쿼리와 각 bbox 후보의 유사도를 계산합니다. 결과는 viser 기반 3D 뷰어에서 확인할 수 있습니다.

## Pipeline

Point Cloud
→ UniDet3D 3D Object Detection
→ 3D Bounding Box Proposals
→ CLIP Text-Object Matching
→ Interactive 3D Visualization

## 주요 기능

- 3D point cloud 기반 객체 검출
- ScanNet / ScanNet++ label head 지원
- CLIP 기반 자연어 쿼리 검색
- 쿼리와 유사한 bbox 강조 표시
- viser 기반 3D 시각화

## 파일 구조

project/
├── infer.py              # UniDet3D inference
├── build_clip_index.py   # bbox별 CLIP embedding 생성
├── app.py                # 3D 시각화 및 검색 앱
└── README.md

## 실행 방법

1. 3D 객체 검출

python infer.py

2. CLIP index 생성

python build_clip_index.py

3. 시각화 앱 실행

python app.py

브라우저에서 접속:

http://localhost:8080

## 입력 데이터 형식

입력 point cloud는 .bin 파일이며, 각 point는 다음과 같은 9차원 feature를 가집니다.

(N, 9) = x, y, z, r, g, b, nx, ny, nz

## 참고

이 저장소에는 UniDet3D 원본 코드, pretrained checkpoint, 대용량 point cloud 데이터는 포함하지 않습니다. 실행 전 별도로 UniDet3D 환경과 입력 데이터를 준비해야 합니다.
