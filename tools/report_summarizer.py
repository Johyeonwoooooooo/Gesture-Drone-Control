# tools/report_summarizer.py
"""
PoC Ollama-based report summarizer.

Usage:
    from tools import report_summarizer
    report = report_summarizer.summarize(raw_data, model="qwen3:1.7b")

Behavior:
- Tries to call local Ollama via the ollama Python client.
- Prompts the model to return a single JSON object following the schema.
- Attempts to parse the model output as JSON; falls back to deterministic summarizer on failure.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, Optional

SYSTEM_PROMPT = """
You are a concise report generator for autonomous drone patrol missions.
Given a raw patrol / flight JSON payload, produce a single JSON object (no markdown,
no prose, nothing else) that conforms to the schema described below.
If a field is not available in the input, use null or an empty list as appropriate.

Output JSON schema (keys and expected types):
{
  "id": string,                       # unique report id
  "mission_start": string|null,
  "mission_end": string|null,
  "total_detections": integer,
  "detections": [
      {"room_id": string, "room_label": string, "floor": string|null,
       "first_seen": string|null, "count": integer}
  ],
  "first_detection": {"room_id":string, "room_label":string, "time":string}|null,
  "flight": {"distance_m": number|null, "duration_s": number|null, "collisions": integer|null},
  "failures": [ {"room_id":string, "room_label":string, "reason":string} ],
  "summary_text_ko": string,
  "recommended_actions_ko": [ string ],
  "generated_at": string
}

Return only the JSON object. Do not include extra commentary.
"""

def _extract_json_from_text(text: str) -> Optional[str]:
    """Find the largest JSON object substring in text by locating first '{' and last '}'."""
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start:end + 1]
    return candidate

def _fallback_summarize(d: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic summarizer used if LLM path fails."""
    now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    rid = d.get('id') or f'report-{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}'

    detections = []
    if isinstance(d.get('locations'), list):
        for loc in d['locations']:
            detections.append({
                'room_id': loc.get('room_id') or loc.get('id') or '',
                'room_label': loc.get('label') or loc.get('room_label') or '',
                'floor': loc.get('floor'),
                'first_seen': loc.get('first_seen') or None,
                'count': int(loc.get('detections') or loc.get('count') or 0),
            })
    elif isinstance(d.get('detections'), list):
        for loc in d['detections']:
            detections.append({
                'room_id': loc.get('room_id') or loc.get('id') or '',
                'room_label': loc.get('label') or loc.get('room_label') or '',
                'floor': loc.get('floor'),
                'first_seen': loc.get('first_seen') or None,
                'count': int(loc.get('count') or 0),
            })

    total_detections = int(d.get('total_detections') or sum(x['count'] for x in detections))
    first_det = d.get('first_detection') or (detections[0] if detections else None)
    flight = d.get('flight') or {}
    failures = d.get('failures') or []

    if isinstance(first_det, dict):
        fd_txt = f"{first_det.get('room_label') or first_det.get('room_id') or ''} {first_det.get('first_seen') or first_det.get('time') or ''}".strip()
    else:
        fd_txt = str(first_det)

    summary_text = f"순찰에서 {total_detections}건의 탐지. 최초 탐지: {fd_txt or '알 수 없음'}."

    report = {
        'id': rid,
        'mission_start': d.get('mission_start') or d.get('start_time') or None,
        'mission_end': d.get('mission_end') or None,
        'total_detections': total_detections,
        'detections': detections,
        'first_detection': first_det,
        'flight': {
            'distance_m': flight.get('distance_m'),
            'duration_s': flight.get('duration_s'),
            'collisions': int(flight.get('collisions', 0) or 0),
        },
        'failures': failures,
        'summary_text_ko': summary_text,
        'recommended_actions_ko': [],
        'generated_at': now,
        'raw': d,
    }
    return report

def summarize(raw: Dict[str, Any], model: str = "qwen3:1.7b", timeout: int = 30) -> Dict[str, Any]:
    """Summarize the given raw patrol/flight JSON into the report schema using Ollama.
    If Ollama is not available or parsing fails, falls back to a deterministic summarizer.
    """
    try:
        import ollama
    except Exception:
        return _fallback_summarize(raw)

    user_content = json.dumps(raw, ensure_ascii=False, indent=2)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Here is the raw patrol payload. Produce the report JSON per the schema.\n\nPayload:\n{user_content}"}
    ]

    try:
        resp = ollama.chat(model=model, messages=messages)
        content = None
        if isinstance(resp, dict):
            msg = resp.get('message') or resp.get('choices')
            if isinstance(msg, dict):
                content = msg.get('content') or msg.get('message') or None
            elif isinstance(msg, list) and msg:
                c = msg[0]
                content = c.get('content') or c.get('message') or None
        if content is None and isinstance(resp, str):
            content = resp

        if not content:
            raise RuntimeError('No content in Ollama response')

        if not isinstance(content, str):
            content = str(content)

        # Try direct JSON parse
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                parsed.setdefault('generated_at', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
                return parsed
        except json.JSONDecodeError:
            cand = _extract_json_from_text(content)
            if cand:
                try:
                    parsed = json.loads(cand)
                    parsed.setdefault('generated_at', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
                    return parsed
                except json.JSONDecodeError:
                    pass

        return _fallback_summarize(raw)

    except Exception:
        return _fallback_summarize(raw)

if __name__ == '__main__':
    import sys
    if sys.stdin.isatty():
        print('Usage: echo payload.json | python tools/report_summarizer.py')
        sys.exit(1)
    raw = json.load(sys.stdin)
    r = summarize(raw)
    print(json.dumps(r, ensure_ascii=False, indent=2))