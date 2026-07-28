# web_assets -> data.js (평면도 PNG base64 + 메타 인라인)
# 점을 다시 찍거나 방이 바뀌면: export_web_assets.py 실행 후 이 스크립트 재실행.
# 인라인하는 이유: index.html 을 서버 없이 더블클릭으로 열 수 있게 (file:// 에선 fetch 불가)
import base64
import json
from pathlib import Path

ASSETS = Path(__file__).parent.parent / 'web_assets'
meta = json.load(open(ASSETS / 'web_meta.json', encoding='utf-8'))

imgs = {}
for f, fl in meta['floors'].items():
    b64 = base64.b64encode((ASSETS / fl['png']).read_bytes()).decode()
    imgs[f] = 'data:image/png;base64,' + b64

out = Path(__file__).parent / 'data.js'
with open(out, 'w', encoding='utf-8') as fp:
    fp.write('// make_data.py 로 자동 생성 — 직접 수정하지 말 것\n')
    fp.write('const META = ' + json.dumps(meta, ensure_ascii=False) + ';\n')
    fp.write('const FLOOR_IMG = ' + json.dumps(imgs) + ';\n')
print(f'data.js: {out.stat().st_size / 1e6:.1f}MB')
