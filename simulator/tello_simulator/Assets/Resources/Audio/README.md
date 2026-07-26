# 호러 사운드 에셋 놓는 곳

`HorrorAudio.cs`가 여기서 클립을 자동으로 집어간다 (`Resources.Load`).
Inspector에서 직접 드래그해 넣으면 그 값이 우선이고, 이 폴더는 폴백이다.

| 파일 | 역할 | 권장 길이 / 성격 |
|---|---|---|
| `ambient.wav` | 2D 룸톤. 계속 루프 | 30~120초, 이음매 없는 loop. 저주파 웅웅거림 / 바람 / 빈 건물 공조음 |
| `heartbeat.wav` | 2D 심박. 후보(preview)에 가까워질수록 볼륨·피치 상승 | 1~2초, loop 가능한 두근 1~2회 |
| `Stingers/*.wav` | 12~35초 랜덤 간격으로 드론 주변 3D 랜덤 위치에서 재생 | 0.5~3초 one-shot. 삐걱, 발소리, 문 닫힘, 속삭임, 금속 긁힘 등 여러 개 |
| `drone.wav` | 드론 로터 루프. 드론에 붙은 3D 소스라 거리·방향에 따라 들린다. 속도에 따라 피치·볼륨 상승, 착륙하면 잦아든다 | 2~5초, 이음매 없는 loop. 프로펠러 웅웅거림 |
| `drone_takeoff.wav` | 이륙 순간 1회 | 1~2초 스핀업 |
| `drone_land.wav` | 착륙 순간 1회 | 1~2초 스핀다운 |

`Stingers/` 안의 wav는 **전부** 로드되므로 (`Resources.LoadAll`), 파일명은 자유.

포맷: Unity가 읽는 것이면 아무거나 (`.wav` / `.ogg` / `.mp3`). `.wav` 권장.

## 클립이 없어도 된다

모든 레이어가 null-safe다. 파일을 안 넣으면 그 레이어만 조용히 꺼지고
씬은 정상 동작한다. 조명/포그/포스트FX부터 먼저 확인해도 무방.

## 출처 (CC0 / free)

- freesound.org — CC0 필터 걸고 `horror ambience`, `heartbeat`, `creak`, `footstep`
- pixabay.com/sound-effects — 전부 무료, 가입 불필요
- opengameart.org — 게임용 CC0 사운드팩

라이선스가 CC-BY면 `README-integration.md`에 출처 한 줄 남길 것.

## 볼륨 조절

`HorrorAtmosphere` 오브젝트의 `HorrorAudio` 컴포넌트 Inspector에서:
`ambientVolume`, `stingerVolume`, `heartbeatMaxVolume`,
`stingerMinDelay`/`stingerMaxDelay` (스팅어 빈도),
`heartbeatFarDistance`/`heartbeatNearDistance` (심박이 붙는 거리 범위).

드론 로터: `droneVolumeIdle`(호버링 0.25) / `droneVolumeMax`(전속 0.55),
`dronePitchIdle` 0.85 / `dronePitchMax` 1.3, `droneFullSpeed`(전속 기준 15 u/s,
`TelloSimulator.moveSpeed` 와 맞춘 값), `droneMinDistance`/`droneMaxDistance`
(3인칭 카메라가 드론 뒤 약 4유닛이라 minDistance 3이 기본).
