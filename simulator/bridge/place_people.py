#!/usr/bin/env python3
"""Pick where people stand in the house and write it out for Unity.

`PersonPlacer.cs` reads the JSON this emits and clones the NPC prefabs already
sitting in `test.unity` onto those spots at Play. Nothing here touches Unity and
nothing in Unity recomputes this — the building is fixed (00809), so the spots
are decided once, checked in, and read back.

Why not place them by hand in the Editor: the scene's three hand-placed NPCs are
what showed the problem. Two of them sit 3 m *below* the ground floor (Unity
y=0.18 → world z=-3.06 m) where no patrol can ever see them, and all three are
at scale 3 in a house imported at scale 5 — 1.08 m tall people. Deriving the
spots from the room index instead makes both mistakes unrepresentable: the floor
comes from `RoomInfo.floor_z` and the scale from the house's own.

Spots are placed around each room's scan pose — the exact point the drone hovers
at to sweep that room (`room_index.scan_pose`) — so whatever the patrol visits,
it sees. People face that point, because a frontal view is what YOLO is best at.

    python simulator/bridge/place_people.py            # 기본 경로에 쓴다
    python simulator/bridge/place_people.py --dry-run  # 세지 않고 표만 본다
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from patrol import room_index                      # noqa: E402
from patrol.litept_backend import LitePTBackend    # noqa: E402
from patrol import planner                         # noqa: E402
from simulator.bridge import coord_transform       # noqa: E402


# 방 넓이(m²) → 세울 사람 수. 벽장만 한 방에 사람을 세우면 드론이 들어가지도
# 못하고, 큰 거실에 하나만 두면 한 바퀴 도는 동안 안 보이는 각도가 생긴다.
AREA_BUCKETS = [(5.0, 0), (20.0, 1), (60.0, 2), (float("inf"), 3)]

PERSON_HEIGHT_M = 1.8
# 사람이 설 자리가 만족해야 하는 것들. 전부 월드 미터.
MIN_DIST_FROM_SCAN = 1.5   # 드론 바로 아래면 카메라에 안 들어온다
MIN_DIST_FLOOR = 0.9       # 좁은 방에서 여기까지는 좁힌다 (아래 참고)
MAX_DIST_FROM_SCAN = 5.0   # 너무 멀면 어두워서 안 잡힌다
MIN_SEPARATION = 1.2       # 사람끼리
BODY_RADIUS = 0.35

# 사람 뒤(방 중심 반대쪽)를 이만큼 훑어 벽이나 가구가 있는 자리만 받는다.
#
# 문간은 방에서 **가장 확실하게 비어 있는 자리**라 아래 링 탐색이 제일 먼저
# 찾아낸다 — 그래서 사람이 입구에 서고 드론이 방에 들어오다 부딪혔다. 벽을
# 등진 자리만 받으면 사람은 저절로 사이드로 가고 문간은 빠진다: 문간은 뒤가
# 계속 뚫려 있어서 이 검사를 통과하지 못한다.
#
# 자기 몸 반경 안은 이미 standing_spot_is_clear 가 비어 있다고 확인한 뒤라
# 훑기는 BODY_RADIUS 밖에서 시작한다.
BACKING_MAX_M = 1.2
BACKING_STEP_M = 0.15      # 복셀 해상도와 같게
BACKING_HEIGHTS = (0.3, 0.9, 1.5)


def min_dist_for(room) -> float:
    """1.5 m 를 고정으로 요구하면 2.3×2.4 m 짜리 방에서는 그 반경이 이미 방
    밖이라 아무도 못 선다. 방의 짧은 변에 맞춰 좁히되, 너무 붙으면 화각
    (FOV 100°, 세로 약 62°)에 몸이 다 안 들어오므로 0.9 m 아래로는 안 간다."""
    short_side = float(min(room.size_xy))
    return float(min(MIN_DIST_FROM_SCAN, max(MIN_DIST_FLOOR, short_side * 0.35)))


def people_for_area(area_m2: float) -> int:
    for limit, n in AREA_BUCKETS:
        if area_m2 < limit:
            return n
    return AREA_BUCKETS[-1][1]


def _free(gm, p: np.ndarray) -> bool:
    """Is this world point empty in the voxel grid?"""
    try:
        return planner.grid_at(gm, *planner.world_to_voxel(gm, p)) == 0
    except Exception:
        return False


def standing_spot_is_clear(gm, xy: np.ndarray, floor_z: float) -> bool:
    """A person is a 1.8 m column, not a point — check along it, and check a
    ring at chest height so we do not stand them inside a sofa."""
    for dz in (0.3, 0.9, 1.5):
        if not _free(gm, np.array([xy[0], xy[1], floor_z + dz])):
            return False
    chest = floor_z + 0.9
    for ang in range(0, 360, 90):
        a = math.radians(ang)
        probe = np.array([xy[0] + BODY_RADIUS * math.cos(a),
                          xy[1] + BODY_RADIUS * math.sin(a), chest])
        if not _free(gm, probe):
            return False
    return True


def has_backing(gm, xy: np.ndarray, floor_z: float,
                outward: np.ndarray) -> bool:
    """사람 등 뒤(방 중심 반대쪽)에 벽이나 가구가 있는가 — 즉 '사이드'인가.

    문간은 뒤가 계속 뚫려 있으므로 여기서 걸러진다. 벽·가구·창틀 어느 것이든
    막혀 있기만 하면 통과다. 격자 밖도 막힌 것으로 친다 (건물 바깥이다)."""
    n = int(round((BACKING_MAX_M - BODY_RADIUS) / BACKING_STEP_M))
    for i in range(1, n + 1):
        p = xy + outward * (BODY_RADIUS + i * BACKING_STEP_M)
        for dz in BACKING_HEIGHTS:
            if not _free(gm, np.array([p[0], p[1], floor_z + dz])):
                return True
    return False


def _ring_search(room, gm, scan: np.ndarray, want: int,
                 chosen: list[np.ndarray], require_backing: bool) -> None:
    """스캔 지점에서 바깥으로 링을 넓혀가며 자리를 고른다 (`chosen` 을 채운다)."""
    lo = room.bbox_min[:2] + BODY_RADIUS
    hi = room.bbox_max[:2] - BODY_RADIUS
    for radius in np.arange(min_dist_for(room), MAX_DIST_FROM_SCAN + 0.01, 0.25):
        # 각도를 라디우스마다 어긋나게 돌려 같은 방향으로만 줄 서지 않게 한다.
        phase = (radius * 37.0) % 30.0
        for ang in np.arange(phase, phase + 360.0, 15.0):
            if len(chosen) >= want:
                return
            a = math.radians(ang)
            direction = np.array([math.cos(a), math.sin(a)])
            xy = scan[:2] + radius * direction
            if np.any(xy < lo) or np.any(xy > hi):
                continue
            if not standing_spot_is_clear(gm, xy, room.floor_z):
                continue
            if require_backing and not has_backing(gm, xy, room.floor_z,
                                                   direction):
                continue
            if any(np.linalg.norm(xy - c[:2]) < MIN_SEPARATION for c in chosen):
                continue
            chosen.append(np.array([xy[0], xy[1], room.floor_z]))


def spots_for_room(room, gm, hover_height: float, want: int) -> list[np.ndarray]:
    """Ring search outward from the scan pose, keeping spots that are clear,
    far enough from the drone, far enough from each other, and **backed by a
    wall** so nobody stands in the doorway the drone flies through.

    벽을 등진 자리를 먼저 다 훑고, 그래도 인원이 모자랄 때만 등 검사를 풀고
    다시 훑는다. 작고 뻥 뚫린 방에서 '입구를 피하려다 아무도 못 세우는' 쪽이
    더 나쁘기 때문이다 — 그 방은 원래도 부딪힐 자리가 없다."""
    if want <= 0:
        return []

    scan = room_index.scan_pose(room, hover_height, gm)
    chosen: list[np.ndarray] = []
    _ring_search(room, gm, scan, want, chosen, require_backing=True)
    if len(chosen) < want:
        _ring_search(room, gm, scan, want, chosen, require_backing=False)
    return chosen


def unity_yaw_facing(sim_tf, person_world: np.ndarray,
                     target_world: np.ndarray) -> float:
    """Unity eulerAngles.y that points the person at `target_world`.
    0° faces +Z and grows toward +X, so it is atan2(dx, dz)."""
    pu, tu = sim_tf.mosaic_to_unity(np.stack([person_world, target_world]))
    dx, dz = float(tu[0] - pu[0]), float(tu[2] - pu[2])
    if abs(dx) < 1e-6 and abs(dz) < 1e-6:
        return 0.0
    return math.degrees(math.atan2(dx, dz)) % 360.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(REPO / "data" / "final_npy"))
    ap.add_argument("--room-aliases", default=str(REPO / "patrol" / "room_aliases.json"))
    ap.add_argument("--building", default="00809_Qpor2mEya8F")
    ap.add_argument("--out", default=str(
        REPO / "simulator" / "tello_simulator" / "Assets" / "Resources"
        / "person_spawn_points.json"))
    ap.add_argument("--hover-height", type=float, default=1.2,
                    help="api_server 기본값과 같아야 스캔 지점이 일치한다")
    ap.add_argument("--point-stride", type=int, default=4)
    ap.add_argument("--resolution", type=float, default=0.15)
    ap.add_argument("--margin", type=int, default=1)
    ap.add_argument("--sample", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true", help="파일을 쓰지 않는다")
    args = ap.parse_args()

    backend = LitePTBackend(Path(args.data_dir))
    rooms = room_index.build_room_index(backend, args.room_aliases)
    print(f"[place] {len(rooms)} rooms")

    points = backend.load_points(stride=args.point_stride)
    gm = planner.voxelize(points, args.resolution, args.margin, args.sample)
    print(f"[place] grid {gm.shape}")

    sim_tf = coord_transform.load_building_transform(args.building)
    # 집이 scale 5 로 들어와 있으므로 실측 1.8 m 로 만들어진 프리팹도 5 배여야
    # 키가 맞는다. 변환 행렬에서 직접 읽어 건물 배치를 바꿔도 따라가게 한다.
    probe = sim_tf.mosaic_to_unity(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    unity_per_meter = float(np.linalg.norm(probe[1] - probe[0]))

    out: list[dict] = []
    table: list[tuple] = []
    for room in sorted(rooms.values(), key=lambda r: (r.floor, r.room_name)):
        area = float(room.size_xy[0] * room.size_xy[1])
        want = people_for_area(area)
        spots = spots_for_room(room, gm, args.hover_height, want)
        table.append((room.display, area, want, len(spots)))

        scan = room_index.scan_pose(room, args.hover_height, gm)
        for i, world in enumerate(spots):
            u = sim_tf.mosaic_to_unity(world.reshape(1, 3))[0]
            out.append({
                "room": room.room_name,
                "display": room.display,
                "index": i,
                "unity_position": [round(float(v), 3) for v in u],
                "unity_yaw": round(unity_yaw_facing(sim_tf, world, scan), 1),
                "unity_scale": round(unity_per_meter, 3),
                "world": [round(float(v), 3) for v in world],
            })

    print(f"\n{'면적m2':>7} {'요청':>4} {'배치':>4}  방")
    for disp, area, want, got in table:
        flag = "" if want == got else "   <- 자리 부족"
        print(f"{area:7.1f} {want:4d} {got:4d}  {disp}{flag}")
    print(f"\n총 {len(out)} 명 / {len(rooms)} 개 방, "
          f"사람 스케일 {unity_per_meter:.2f} (프리팹 1 m 당 유니티 유닛)")

    if args.dry_run:
        print("[place] --dry-run: 파일을 쓰지 않았다")
        return 0

    payload = {
        "building": args.building,
        "person_height_m": PERSON_HEIGHT_M,
        "hover_height": args.hover_height,
        "note": "simulator/bridge/place_people.py 가 생성. 직접 고치지 말 것.",
        "spawns": out,
    }
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(f"[place] wrote {dest} ({dest.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
