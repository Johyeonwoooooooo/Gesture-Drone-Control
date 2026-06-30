"""
자연어 쿼리로 JSON에서 물체 검색

사용법:
    python query.py "거실에서 소파 찾아줘"
    python query.py "침실 침대" --json detections.json
    python query.py --interactive
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# ── 한국어 방 타입 → room_type 키 매핑 ───────────────────────────────────────
ROOM_KW_MAP: dict = {
    '거실':   'living',
    '리빙룸': 'living',
    '침실':   'bedroom',
    '안방':   'bedroom',
    '방':     'bedroom',
    '침방':   'bedroom',
    '주방':   'kitchen',
    '부엌':   'kitchen',
    '키친':   'kitchen',
    '화장실': 'bathroom',
    '욕실':   'bathroom',
    '욕조':   'bathroom',
    '바스룸': 'bathroom',
    '서재':   'office',
    '사무실': 'office',
    '오피스': 'office',
    '식당':   'dining',
    '다이닝': 'dining',
    '식사':   'dining',
}

# ── 한국어 → ScanNet class_name 매핑 ─────────────────────────────────────────
LABEL_KW_MAP: dict = {
    '의자':   'chair',
    '체어':   'chair',
    '좌석':   'chair',
    '테이블': 'table',
    '탁자':   'table',
    '식탁':   'table',
    '책상':   'desk',
    '데스크': 'desk',
    '소파':   'sofa',
    '쇼파':   'sofa',
    '카우치': 'sofa',
    '침대':   'bed',
    '베드':   'bed',
    '캐비닛': 'cabinet',
    '서랍장': 'cabinet',
    '장롱':   'cabinet',
    '붙박이': 'cabinet',
    '수납장': 'cabinet',
    '책장':   'bookshelf',
    '책꽂이': 'bookshelf',
    '선반':   'bookshelf',
    '문':     'door',
    '도어':   'door',
    '창문':   'window',
    '창':     'window',
    '그림':   'picture',
    '액자':   'picture',
    '카운터': 'counter',
    '조리대': 'counter',
    '커튼':   'curtain',
    '블라인드':'curtain',
    '냉장고': 'refrigerator',
    '변기':   'toilet',
    '세면대': 'sink',
    '싱크대': 'sink',
    '싱크':   'sink',
    '욕조':   'bathtub',
    '가구':   'otherfurniture',
}


def parse_query(text: str, room_names: Optional[list] = None):
    """쿼리 문자열에서 (room_name_exact, room_type, label) 추출.

    room_names가 주어지면 구체적 방 이름(거실1, 침실2 등) 먼저 매칭.
    매칭 없으면 room_type 키워드(거실, 침실 등)로 폴백.
    """
    detected_room_name = None
    detected_room_type = None
    detected_label     = None

    # 구체적 방 이름 먼저 매칭 (거실1, 침실2 등)
    if room_names:
        for name in sorted(room_names, key=len, reverse=True):
            if name in text:
                detected_room_name = name
                break

    # 방 이름이 없으면 room_type 키워드
    if detected_room_name is None:
        for kw, canonical in sorted(ROOM_KW_MAP.items(), key=lambda x: -len(x[0])):
            if kw in text:
                detected_room_type = canonical
                break

    for kw, canonical in sorted(LABEL_KW_MAP.items(), key=lambda x: -len(x[0])):
        if kw in text:
            detected_label = canonical
            break

    return detected_room_name, detected_room_type, detected_label


def search(entries: list, room_name: Optional[str], room_type: Optional[str],
           label: Optional[str]) -> list:
    results = entries
    if label:
        results = [e for e in results if e['label'] == label]
    if room_name:
        results = [e for e in results if e.get('room_name') == room_name]
    elif room_type:
        results = [e for e in results if e.get('room_type') == room_type]
    return results


def format_result(entry: dict, idx: int) -> str:
    cx, cy, cz = entry['center']
    room_type = entry.get('room_type', '?')
    return (
        f"  [{idx+1}] {entry['label']:20s}"
        f" | 방타입: {room_type:10s}"
        f" | room: {entry['room']}"
        f" | center: ({cx:.2f}, {cy:.2f}, {cz:.2f})"
        f" | points: {entry['n_points']}"
    )


def run_query(text: str, entries: list) -> list:
    room_names = list({e.get('room_name') for e in entries if e.get('room_name')})
    room_name, room_type, label = parse_query(text, room_names)

    hints = []
    if label:      hints.append(f'라벨={label}')
    if room_name:  hints.append(f'방={room_name}')
    elif room_type: hints.append(f'방타입={room_type}')
    if not hints:
        print('  ※ 방 이름이나 물체 종류를 인식하지 못했습니다.')
        print('  예: "거실1 소파", "침실2에서 침대 찾아줘", "주방 냉장고"')
        return []

    print(f'  검색 조건: {", ".join(hints)}')
    results = search(entries, room_name, room_type, label)

    if not results:
        print('  검색 결과 없음.')
        return []

    print(f'  {len(results)}개 검출:')
    for i, e in enumerate(results):
        print(format_result(e, i))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('query', nargs='?', default=None)
    parser.add_argument('--json', type=Path,
                        default=Path(__file__).parent.parent / 'project/data/npy/detections.json')
    parser.add_argument('--interactive', '-i', action='store_true')
    args = parser.parse_args()

    if not args.json.exists():
        print(f'JSON 파일을 찾을 수 없습니다: {args.json}')
        print('먼저 export_json.py를 실행해 detections.json을 생성하세요.')
        sys.exit(1)

    with open(args.json, encoding='utf-8') as f:
        entries = json.load(f)

    print(f'로드 완료: {len(entries)}개 인스턴스 ({args.json})\n')

    if args.interactive or not args.query:
        print('쿼리를 입력하세요 (종료: q 또는 Ctrl-C)')
        while True:
            try:
                text = input('> ').strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text.lower() in ('q', 'quit', 'exit'):
                break
            if not text:
                continue
            run_query(text, entries)
            print()
    else:
        run_query(args.query, entries)


if __name__ == '__main__':
    main()
