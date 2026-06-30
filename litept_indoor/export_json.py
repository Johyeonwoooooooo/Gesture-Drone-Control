"""
모든 방의 centers.pkl → detections.json 통합 저장
가구 구성을 보고 방 타입을 자동 추정해서 room_type 필드에 저장

사용법:
    python export_json.py --npy_root /shareHost/minyoy/project/data/npy
    python export_json.py --npy_root /shareHost/minyoy/project/data/npy --out detections.json
"""
import argparse
import json
import pickle
from collections import Counter
from pathlib import Path


# ── 방 타입 추정 규칙 ─────────────────────────────────────────────────────────
# 각 가구가 어느 방에 "기여"하는지 점수 정의
ROOM_SCORES: dict = {
    'bed':            {'bedroom': 10},
    'bathtub':        {'bathroom': 10},
    'toilet':         {'bathroom': 8},
    'sink':           {'bathroom': 3, 'kitchen': 3},
    'refrigerator':   {'kitchen': 10},
    'counter':        {'kitchen': 6},
    'sofa':           {'living': 6},
    'table':          {'living': 3, 'dining': 4},
    'chair':          {'living': 2, 'dining': 3, 'office': 2},
    'desk':           {'office': 8},
    'bookshelf':      {'office': 5, 'bedroom': 2},
    'curtain':        {'living': 1, 'bedroom': 1},
    'picture':        {'living': 1},
    'cabinet':        {'living': 1, 'bedroom': 2, 'kitchen': 2},
    'door':           {},
    'window':         {},
    'shower curtain': {'bathroom': 8},
}

ROOM_TYPE_KO: dict = {
    'bedroom':  '침실',
    'bathroom': '화장실',
    'kitchen':  '주방',
    'living':   '거실',
    'dining':   '식당',
    'office':   '서재',
    'unknown':  '알 수 없음',
}


def infer_room_type(labels: list) -> str:
    """가구 라벨 리스트로 방 타입 추정."""
    score: Counter = Counter()
    for lbl in labels:
        for room, pts in ROOM_SCORES.get(lbl, {}).items():
            score[room] += pts
    if not score:
        return 'unknown'
    return score.most_common(1)[0][0]


def pkl_to_entries(pkl_path: Path) -> list:
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    room = data['room']
    labels_in_room = [obj['class_name'] for obj in data['centers']]
    room_type = infer_room_type(labels_in_room)

    entries = []
    for obj in data['centers']:
        cx, cy, cz = obj['center'].tolist()
        entries.append({
            'room':       room,
            'room_type':  room_type,
            'label':      obj['class_name'],
            'label_idx':  obj['label_idx'],
            'center':     [round(cx, 4), round(cy, 4), round(cz, 4)],
            'n_points':   int(obj['n_points']),
        })
    return entries, room_type


def assign_room_names(room_ids: list, room_types: list,
                      existing: dict) -> dict:
    """room_id → 한국어 이름 자동 할당. 기존 이름은 유지."""
    names = dict(existing)
    type_count: Counter = Counter()
    for rid, rtype in zip(room_ids, room_types):
        if rid in names:
            continue
        ko = ROOM_TYPE_KO.get(rtype, '알 수 없음')
        type_count[ko] += 1
        names[rid] = f'{ko}{type_count[ko]}'
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy_root', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()

    out      = args.out or (args.npy_root / 'detections.json')
    names_path = args.npy_root / 'room_names.json'

    pkl_files = sorted(args.npy_root.glob('*/centers.pkl'))
    if not pkl_files:
        print('centers.pkl 파일을 찾을 수 없습니다. 먼저 infer_centers.py를 실행하세요.')
        return

    # 기존 room_names.json 로드 (수동 수정 이름 보존)
    existing_names = {}
    if names_path.exists():
        with open(names_path, encoding='utf-8') as f:
            existing_names = json.load(f)

    all_entries = []
    room_ids, room_types = [], []
    for pkl in pkl_files:
        entries, room_type = pkl_to_entries(pkl)
        all_entries.extend(entries)
        room_ids.append(pkl.parent.name)
        room_types.append(room_type)

    room_names = assign_room_names(room_ids, room_types, existing_names)

    # detections.json에 room_name 필드 추가
    for e in all_entries:
        e['room_name'] = room_names.get(e['room'], e['room'])

    with open(out, 'w', encoding='utf-8') as f:
        json.dump(all_entries, f, ensure_ascii=False, indent=2)

    with open(names_path, 'w', encoding='utf-8') as f:
        json.dump(room_names, f, ensure_ascii=False, indent=2)

    for rid, rtype in zip(room_ids, room_types):
        ko = ROOM_TYPE_KO.get(rtype, rtype)
        print(f'  {rid}: {ko}  →  "{room_names[rid]}"')

    print(f'\n총 {len(all_entries)}개 인스턴스 → {out}')
    print(f'방 이름 → {names_path}')


if __name__ == '__main__':
    main()
