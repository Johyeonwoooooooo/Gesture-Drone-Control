"""순찰 보고서 — LLM-written patrol report from a finished mission.

Output (one folder per patrol, under `out/reports/`)::

    20260801_143251_patrol/
        report.md      relative image links, good for git/PR review
        report.html    base64-embedded images, self-contained, open in a browser
        report.json    machine-readable, for a future web UI
        events/        the detector's photos, copied at detection time

The factual sections (날짜 / 순찰 구역 / 탐지 건수 / 이벤트 상세 / 비행 통계) are
built deterministically — the LLM only writes the 요약 paragraph, and a template
sentence takes over if it returns nothing usable. A report is never blocked by
the model.
"""
from __future__ import annotations

import base64
import html
import json
import mimetypes
import time
from pathlib import Path
from typing import List, Optional, Sequence

REPORT_SYSTEM_PROMPT = """당신은 실내 순찰 드론의 보고서를 작성하는 보안 관제 어시스턴트입니다.

입력은 방금 끝난 순찰의 사실 요약 JSON입니다. 이를 바탕으로 한국어 요약을
3~5문장으로 작성하세요.

규칙:
- JSON에 있는 사실만 사용하고, 없는 내용을 지어내지 마세요.
- 탐지가 0건이면 이상 없음을 분명히 밝히세요.
- 탐지가 있으면 어느 방에서 몇 건이 언제 발견됐는지 먼저 쓰세요.
- 도달하지 못한 방이나 충돌이 있었다면 마지막에 한 문장으로 덧붙이세요.
- 마크다운 제목이나 목록 없이, 평문 문단으로만 답하세요.
"""


# ------------------------------------------------------------------- assembly

def _facts(result, intent, rooms, query: str) -> dict:
    """The compact, LLM-safe view of a PatrolResult."""
    return {
        "날짜": time.strftime("%Y-%m-%d %H:%M", time.localtime(result.started_at)),
        "사용자_명령": query,
        "순찰_구역": [r.display for r in rooms],
        "도달한_구역": [v.display for v in result.rooms if v.reached],
        "미도달_구역": [f"{v.display} ({v.reason})"
                    for v in result.rooms if not v.reached],
        "탐지_건수": len(result.events),
        "탐지_상세": [
            {"시각": e.iso_time, "방": e.room_display or e.room_name,
             "대상": e.label, "신뢰도": round(e.conf, 2),
             "사진": bool(e.saved_image)}
            for e in result.events
        ],
        "비행_거리_m": round(result.distance_m, 1),
        "소요_시간_초": int(result.duration_s),
        "충돌_횟수": result.collisions,
        "복귀_완료": result.returned_home,
        "중단_사유": result.aborted_reason or None,
    }


def _fallback_summary(facts: dict) -> str:
    n = facts["탐지_건수"]
    zones = ", ".join(facts["순찰_구역"]) or "지정 구역"
    if n == 0:
        s = (f"{facts['날짜']} 순찰에서 {zones}을(를) 점검했으며 탐지된 인원은 "
             f"없었습니다. 총 {facts['비행_거리_m']}m를 "
             f"{facts['소요_시간_초']}초간 비행했습니다.")
    else:
        first = facts["탐지_상세"][0]
        s = (f"{facts['날짜']} 순찰 중 {zones}에서 총 {n}건의 탐지가 있었습니다. "
             f"최초 탐지는 {first['시각']} {first['방']}에서 발생했습니다. "
             f"총 {facts['비행_거리_m']}m를 {facts['소요_시간_초']}초간 비행했습니다.")
    if facts["미도달_구역"]:
        s += f" 도달하지 못한 구역: {', '.join(facts['미도달_구역'])}."
    if facts["충돌_횟수"]:
        s += f" 비행 중 충돌 {facts['충돌_횟수']}회가 기록되었습니다."
    return s


def _llm_summary(llm, facts: dict) -> str:
    if llm is None:
        return _fallback_summary(facts)
    try:
        text = llm.generate(REPORT_SYSTEM_PROMPT,
                            json.dumps(facts, ensure_ascii=False, indent=1),
                            max_new_tokens=320).strip()
    except Exception:
        return _fallback_summary(facts)
    # Strip fences/headings a small model sometimes adds anyway.
    text = text.replace("```", "").strip()
    lines = [l.strip() for l in text.splitlines()
             if l.strip() and not l.strip().startswith("#")]
    text = " ".join(lines).strip()
    return text if len(text) >= 20 else _fallback_summary(facts)


# --------------------------------------------------------------------- writers

def _markdown(facts: dict, summary: str, result, rooms) -> str:
    L: List[str] = ["# 순찰 보고서", ""]
    L.append(f"- **날짜**: {facts['날짜']}")
    L.append(f"- **순찰 구역**: {', '.join(facts['순찰_구역']) or '-'}")
    L.append(f"- **사용자 명령**: {facts['사용자_명령']}")
    L.append(f"- **총 탐지 이벤트**: {facts['탐지_건수']}건")
    L.append(f"- **비행**: {facts['비행_거리_m']} m / {facts['소요_시간_초']}초 "
             f"/ 충돌 {facts['충돌_횟수']}회"
             + ("  / 복귀 완료" if facts["복귀_완료"] else "  / 복귀 미완료"))
    if facts["중단_사유"]:
        L.append(f"- **중단 사유**: {facts['중단_사유']}")
    L += ["", "## 요약", "", summary, "", "## 탐지 상세", ""]

    if not result.events:
        L.append("탐지된 인원이 없습니다.")
    for i, e in enumerate(result.events, 1):
        L.append(f"### 이벤트 #{i}")
        L.append("")
        L.append(f"- 시각: {e.iso_time}")
        L.append(f"- 방: {e.room_display or e.room_name}")
        L.append(f"- 대상: {e.label} (신뢰도 {e.conf:.2f})")
        if e.drone_world:
            xyz = ", ".join(f"{v:.2f}" for v in e.drone_world)
            L.append(f"- 드론 위치(world m): ({xyz})")
        if e.bbox:
            L.append(f"- bbox: {[round(v, 1) for v in e.bbox]}")
        if e.saved_image:
            L.append("")
            L.append(f"![탐지 #{i}](events/{Path(e.saved_image).name})")
        else:
            L.append("- 사진: 없음")
        L.append("")

    L += ["## 구역별 결과", "", "| 구역 | 도달 | 스캔 | 탐지 | 이동거리 | 소요 |",
          "|---|---|---|---|---|---|"]
    for v in result.rooms:
        L.append(f"| {v.display} | {'O' if v.reached else f'X ({v.reason})'} "
                 f"| {v.scan_degrees:.0f}° | {v.events}건 "
                 f"| {v.leg_length_m:.1f} m | {v.duration_s:.0f}s |")
    if result.listener_stats:
        s = result.listener_stats
        L += ["", f"> 탐지 수신 통계: 수신 {s.get('received', 0)} / "
                  f"기록 {s.get('accepted', 0)} / "
                  f"이동 중 무시 {s.get('ignored_in_transit', 0)} / "
                  f"필터 {s.get('filtered', 0)}"]
    L.append("")
    return "\n".join(L)


_HTML_CSS = """
:root { color-scheme: light dark; }
body { font-family: -apple-system, "Noto Sans KR", "Malgun Gothic", sans-serif;
       max-width: 820px; margin: 2rem auto; padding: 0 1rem; line-height: 1.65; }
h1 { border-bottom: 3px solid #e04b4b; padding-bottom: .3rem; }
h2 { margin-top: 2.2rem; border-bottom: 1px solid #8884; padding-bottom: .2rem; }
.meta { list-style: none; padding: 0; }
.meta li { padding: .15rem 0; }
.summary { background: #8881; border-left: 4px solid #e04b4b;
           padding: .9rem 1.1rem; border-radius: 0 6px 6px 0; }
.event { border: 1px solid #8884; border-radius: 8px; padding: 1rem;
         margin: 1rem 0; }
.event h3 { margin-top: 0; }
.event img { max-width: 100%; border-radius: 6px; display: block;
             margin-top: .6rem; }
.badge { background: #e04b4b; color: #fff; border-radius: 999px;
         padding: .1rem .6rem; font-size: .85em; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #8884; padding: .4rem .6rem; text-align: left; }
th { background: #8882; }
.none { color: #888; }
footer { margin-top: 2.5rem; font-size: .85em; color: #888; }
"""


# Above this, a photo is downscaled before embedding (and if that is not
# possible, linked relatively instead). Nine full-res PNGs inline made a 11 MB
# report.html in testing — self-contained is only useful if it still opens.
MAX_EMBED_BYTES = 600_000
EMBED_MAX_PX = 1280


def _shrink(path: Path) -> Optional[bytes]:
    """JPEG-recompress an oversized photo for embedding. None if Pillow is
    unavailable or the image cannot be read."""
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        import io
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((EMBED_MAX_PX, EMBED_MAX_PX))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=80, optimize=True)
        return buf.getvalue()
    except Exception:
        return None


def _data_uri(path: str) -> Optional[str]:
    """Base64 data URI for the report's inline image, or None to link instead."""
    p = Path(path)
    try:
        raw = p.read_bytes()
    except OSError:
        return None
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    if len(raw) > MAX_EMBED_BYTES:
        small = _shrink(p)
        if small is None:
            return None          # caller falls back to a relative link
        raw, mime = small, "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def _html(facts: dict, summary: str, result) -> str:
    e_ = html.escape
    P: List[str] = [
        "<!doctype html><html lang='ko'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>순찰 보고서 {e_(facts['날짜'])}</title>",
        f"<style>{_HTML_CSS}</style></head><body>",
        "<h1>순찰 보고서</h1>",
        "<ul class='meta'>",
        f"<li><b>날짜</b> · {e_(facts['날짜'])}</li>",
        f"<li><b>순찰 구역</b> · {e_(', '.join(facts['순찰_구역']) or '-')}</li>",
        f"<li><b>사용자 명령</b> · {e_(facts['사용자_명령'])}</li>",
        f"<li><b>총 탐지 이벤트</b> · <span class='badge'>"
        f"{facts['탐지_건수']}건</span></li>",
        f"<li><b>비행</b> · {facts['비행_거리_m']} m / "
        f"{facts['소요_시간_초']}초 / 충돌 {facts['충돌_횟수']}회</li>",
        "</ul>",
        f"<h2>요약</h2><p class='summary'>{e_(summary)}</p>",
        "<h2>탐지 상세</h2>",
    ]
    if not result.events:
        P.append("<p class='none'>탐지된 인원이 없습니다.</p>")
    for i, ev in enumerate(result.events, 1):
        P.append("<div class='event'>")
        P.append(f"<h3>이벤트 #{i} — {e_(ev.label)} "
                 f"<span class='badge'>{ev.conf:.2f}</span></h3>")
        P.append(f"<div>{e_(ev.iso_time)} · {e_(ev.room_display or ev.room_name)}</div>")
        if ev.drone_world:
            xyz = ", ".join(f"{v:.2f}" for v in ev.drone_world)
            P.append(f"<div>드론 위치(world m): ({e_(xyz)})</div>")
        if ev.saved_image:
            # Prefer an embedded copy (the file travels alone); if it is too big
            # to shrink, link relatively — still works next to events/.
            uri = _data_uri(ev.saved_image) or f"events/{Path(ev.saved_image).name}"
            P.append(f"<img src='{e_(uri)}' alt='탐지 #{i}'>")
        else:
            P.append("<div class='none'>사진 없음</div>")
        P.append("</div>")

    P.append("<h2>구역별 결과</h2><table><tr><th>구역</th><th>도달</th>"
             "<th>스캔</th><th>탐지</th><th>이동거리</th><th>소요</th></tr>")
    for v in result.rooms:
        P.append(f"<tr><td>{e_(v.display)}</td>"
                 f"<td>{'O' if v.reached else 'X (' + e_(v.reason) + ')'}</td>"
                 f"<td>{v.scan_degrees:.0f}°</td><td>{v.events}건</td>"
                 f"<td>{v.leg_length_m:.1f} m</td><td>{v.duration_s:.0f}s</td></tr>")
    P.append("</table>")
    P.append("<footer>patrol patrol agent · 자동 생성</footer>")
    P.append("</body></html>")
    return "\n".join(P)


# ------------------------------------------------------------------ entry point

def build_report(result, rooms: Sequence, query: str, out_dir: Path,
                 llm=None, intent=None) -> Path:
    """Write report.md / report.html / report.json into `out_dir`.

    `out_dir` is the SAME folder the mission already copied its photos into
    (`PatrolConfig.events_dir` == out_dir/"events"), so the markdown's relative
    links resolve without another copy. Returns `out_dir`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    facts = _facts(result, intent, rooms, query)
    summary = _llm_summary(llm, facts)

    (out_dir / "report.md").write_text(
        _markdown(facts, summary, result, rooms), encoding="utf-8")
    (out_dir / "report.html").write_text(
        _html(facts, summary, result), encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps({
        "facts": facts,
        "summary": summary,
        "events": [e.to_json() for e in result.events],
        "rooms": [vars(v) for v in result.rooms],
        "listener_stats": result.listener_stats,
        "planned_path_world": result.planned_path_world,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return out_dir


def report_dir_for(root: str | Path, started_at: Optional[float] = None) -> Path:
    """out/reports/<ts>_patrol — created before the flight so detection photos
    can be copied into it as they arrive."""
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(started_at or time.time()))
    return Path(root) / f"{ts}_patrol"
