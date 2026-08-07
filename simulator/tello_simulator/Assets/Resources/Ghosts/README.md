# Ghosts

`GhostWanderer.cs` 가 `Resources.LoadAll<GameObject>("Ghosts")` 로 이 폴더를 통째로
읽어 유령마다 하나씩 무작위로 골라 쓴다. **.glb 를 여기 떨어뜨리면 그게 곧 추가**다 —
씬 수정도, 코드 수정도 없다. 스케일은 스폰할 때 `ghostMeters` 키에 맞춰 다시
맞추므로 모델 파일의 단위가 뭐든 상관없다.

임포트는 `org.khronos.unitygltf` (Packages/manifest.json) 가 한다. 집 glb
(`Assets/Qpor2mEya8F.glb`) 와 같은 경로다.

## 출처 / 라이선스

셋 다 **CC0 (Public Domain)** — 저작자 표시 의무 없음. 그래도 남겨둔다.

| 파일 | 원본 | 제작 |
|---|---|---|
| `ghost_sheet.glb` | [Ghost](https://poly.pizza/m/Iip30bDHmu) | Quaternius |
| `ghost_skull.glb` | [Ghost Skull](https://poly.pizza/m/TX8r9WBXpe) | Quaternius |
| `ghost_figure.glb` | [Ghost Character](https://poly.pizza/m/CKLHPoYhE9) | Polygonal Mind |
| `zombie.glb` | [Zombie](https://poly.pizza/m/JoBvxIUpZP) | Quaternius |
| `skeleton.glb` | [Skeleton](https://poly.pizza/m/DM4QScSmbS) | Quaternius |

빼고 싶은 모델은 파일만 지우면 된다. 반대로 특정 모델만 쓰려면 인스펙터의
`Ghost Models` 배열에 넣는다 — 배열이 비어 있을 때만 이 폴더를 읽는다.

## 애니메이션

전부 리깅된 모델이고 클립(`CharacterArmature|Idle`, `|Walk`, `|Death` …)이 같이
들어온다. glTF 임포터가 `_addAnimatorComponent: 0` 으로 넣어서 **그냥 두면 바인드
포즈(T 자세)로 굳는다.** `GhostWanderer.PlayIdle()` 이 런타임에 `Animator` 를
붙이고 Playable 로 클립 하나를 돌린다 — AnimatorController 는 필요 없다.

클립은 폴더 전체가 한 뭉치로 로드되므로 `CharacterArmature|` 앞부분(리그 이름)을
모델의 본 이름과 대조해 자기 것만 고른다. 새로 넣는 모델의 리그 이름이 클립
접두어와 다르면 매칭이 안 되니, 그럴 땐 인스펙터 `Clip Preference` 로 우선순위를
조절하거나 `Play Animation` 을 끈다.

## 무서운 모델을 더 넣고 싶다면

poly.pizza 의 CC0 재고는 Quaternius 로우폴리라 여기까지가 한계다. 실사풍
좀비·수녀귀신 같은 건 **Sketchfab (CC0 필터)** 나 **Mixamo** 에 있는데 둘 다
로그인이 필요해서 직접 받아야 한다. 받은 `.glb`(또는 `.fbx`)를 이 폴더에 넣으면
그걸로 끝 — 코드도 씬도 안 건드린다. 키는 스폰할 때 `ghostMeters` 로 다시
맞춰지므로 모델 스케일도 신경 쓸 필요 없다.
