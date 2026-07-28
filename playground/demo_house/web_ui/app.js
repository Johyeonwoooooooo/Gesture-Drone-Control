// 드론 관제 UI (목업) — 출발지 = 드론 현재 위치(고정), 사용자는 목적지 1곳만 클릭 선택.
// 백엔드 연동 지점: planBtn 클릭 시 payload 를 POST /plan 으로 보내면 됨.

'use strict';

const DRONE_ROOM = '007'; // 드론 현재 위치 (목업): 이 방의 center 에 있다고 가정
const FLOOR_NAME = { 0: '1층', 1: '2층', 2: '3층' };

const drone = (() => {
  const r = META.rooms[DRONE_ROOM];
  return { room: DRONE_ROOM, floor: r.floor, pos: [...r.center] };
})();

const state = {
  floor: drone.floor,          // 보고 있는 층
  view: { scale: 1, ox: 0, oy: 0 },  // 이미지px -> 화면px: s = p*scale + o
  hoverRoom: null,
  dest: null,                  // {room, floor, pos:[x,y,z]}
  planned: false,              // '경로 계획' 눌렀는지 (목업 점선 표시용)
  imgs: {},                    // floor -> HTMLImageElement
};

// ── DOM ──────────────────────────────────────────
const canvas = document.getElementById('map');
const ctx = canvas.getContext('2d');
const wrap = document.getElementById('map-wrap');
const tooltip = document.getElementById('tooltip');
const optRooms = document.getElementById('opt-rooms');
const optDoors = document.getElementById('opt-doors');
const optPassages = document.getElementById('opt-passages');
const planBtn = document.getElementById('plan-btn');

// ── 좌표 변환 ────────────────────────────────────
const fl = () => META.floors[String(state.floor)];
const worldToImg = (x, y) => [(x - fl().xmin) / META.res, (fl().ymax - y) / META.res];
const imgToWorld = (c, r) => [fl().xmin + c * META.res, fl().ymax - r * META.res];
const imgToScreen = (c, r) => [c * state.view.scale + state.view.ox, r * state.view.scale + state.view.oy];
const screenToImg = (sx, sy) => [(sx - state.view.ox) / state.view.scale, (sy - state.view.oy) / state.view.scale];
const worldToScreen = (x, y) => imgToScreen(...worldToImg(x, y));

// 월드좌표 (x,y) 가 속한 방 (현재 층에서). bbox 겹치면 center 가 가까운 쪽.
function roomAt(x, y) {
  let best = null, bestD = Infinity;
  for (const id of fl().rooms) {
    const r = META.rooms[id];
    if (x < r.bbox_min[0] || x > r.bbox_max[0] || y < r.bbox_min[1] || y > r.bbox_max[1]) continue;
    const d = (x - r.center[0]) ** 2 + (y - r.center[1]) ** 2;
    if (d < bestD) { bestD = d; best = id; }
  }
  return best;
}
const roomSize = id => {
  const r = META.rooms[id];
  return `${(r.bbox_max[0] - r.bbox_min[0]).toFixed(1)} × ${(r.bbox_max[1] - r.bbox_min[1]).toFixed(1)} m`;
};

// ── 뷰 조작 ──────────────────────────────────────
function fitView() {
  const f = fl();
  const s = Math.min(wrap.clientWidth / f.width, wrap.clientHeight / f.height) * 0.92;
  state.view.scale = s;
  state.view.ox = (wrap.clientWidth - f.width * s) / 2;
  state.view.oy = (wrap.clientHeight - f.height * s) / 2;
}
function zoomAt(sx, sy, factor) {
  const v = state.view;
  const s = Math.max(0.15, Math.min(10, v.scale * factor));
  v.ox = sx - (sx - v.ox) * (s / v.scale);
  v.oy = sy - (sy - v.oy) * (s / v.scale);
  v.scale = s;
}
function focusRoom(id) {
  const r = META.rooms[id];
  if (r.floor !== state.floor) setFloor(r.floor);
  const [c0, r0] = worldToImg(r.bbox_min[0], r.bbox_max[1]); // 좌상단
  const [c1, r1] = worldToImg(r.bbox_max[0], r.bbox_min[1]); // 우하단
  const s = Math.min(wrap.clientWidth / (c1 - c0), wrap.clientHeight / (r1 - r0)) * 0.6;
  state.view.scale = Math.min(s, 6);
  state.view.ox = wrap.clientWidth / 2 - (c0 + c1) / 2 * state.view.scale;
  state.view.oy = wrap.clientHeight / 2 - (r0 + r1) / 2 * state.view.scale;
}

// ── 그리기 ───────────────────────────────────────
function resize() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = wrap.clientWidth * dpr;
  canvas.height = wrap.clientHeight * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function draw(t) {
  ctx.clearRect(0, 0, wrap.clientWidth, wrap.clientHeight);
  const img = state.imgs[String(state.floor)];
  const v = state.view;
  if (img) {
    ctx.imageSmoothingEnabled = v.scale < 2;
    ctx.drawImage(img, v.ox, v.oy, img.width * v.scale, img.height * v.scale);
  }

  // 방 경계
  if (optRooms.checked) {
    for (const id of fl().rooms) {
      const r = META.rooms[id];
      const [x0, y0] = worldToScreen(r.bbox_min[0], r.bbox_max[1]);
      const [x1, y1] = worldToScreen(r.bbox_max[0], r.bbox_min[1]);
      const hov = id === state.hoverRoom;
      if (hov) { ctx.fillStyle = 'rgba(57,208,164,.10)'; ctx.fillRect(x0, y0, x1 - x0, y1 - y0); }
      ctx.strokeStyle = hov ? 'rgba(57,208,164,.9)' : 'rgba(123,138,156,.35)';
      ctx.lineWidth = hov ? 1.6 : 1;
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      if (v.scale > 0.55 || hov) {
        ctx.fillStyle = hov ? 'rgba(57,208,164,.95)' : 'rgba(219,228,238,.55)';
        ctx.font = '600 11px sans-serif';
        ctx.fillText(`방 ${id}`, x0 + 6, y0 + 15);
      }
    }
  }

  // 문 / 계단
  if (optDoors.checked) {
    for (const e of META.edges) {
      const fa = META.rooms[e.a].floor, fb = META.rooms[e.b].floor;
      if (fa !== state.floor && fb !== state.floor) continue;
      const [sx, sy] = worldToScreen(e.door_center[0], e.door_center[1]);
      if (e.type === 'stairs') {
        ctx.fillStyle = '#b487f0';
        ctx.save();
        ctx.translate(sx, sy); ctx.rotate(Math.PI / 4);
        ctx.fillRect(-4.5, -4.5, 9, 9);
        ctx.restore();
        ctx.fillStyle = 'rgba(180,135,240,.9)';
        ctx.font = '600 10px sans-serif';
        ctx.fillText('계단', sx + 8, sy + 3);
      } else {
        ctx.fillStyle = '#e8c34a';
        ctx.beginPath(); ctx.arc(sx, sy, 3.5, 0, 7); ctx.fill();
      }
    }
  }

  // 통로 지정점
  if (optPassages.checked) {
    ctx.fillStyle = 'rgba(74,184,232,.85)';
    for (const id of fl().rooms)
      for (const p of META.rooms[id].passages || []) {
        const [sx, sy] = worldToScreen(p[0], p[1]);
        ctx.beginPath(); ctx.arc(sx, sy, 2.5, 0, 7); ctx.fill();
      }
  }

  // 경로 미리보기 (목업: 같은 층이면 직선 점선만)
  if (state.planned && state.dest && state.dest.floor === drone.floor && state.floor === drone.floor) {
    const a = worldToScreen(drone.pos[0], drone.pos[1]);
    const b = worldToScreen(state.dest.pos[0], state.dest.pos[1]);
    ctx.setLineDash([7, 6]);
    ctx.lineDashOffset = -(t / 40) % 13;
    ctx.strokeStyle = 'rgba(255,107,94,.85)';
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    ctx.setLineDash([]);
  }

  // 드론 마커 (현재 층일 때, 펄스)
  if (drone.floor === state.floor) {
    const [sx, sy] = worldToScreen(drone.pos[0], drone.pos[1]);
    const pulse = 8 + 6 * (0.5 + 0.5 * Math.sin(t / 350));
    ctx.strokeStyle = 'rgba(57,208,164,.45)';
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(sx, sy, pulse, 0, 7); ctx.stroke();
    ctx.fillStyle = '#39d0a4';
    ctx.beginPath(); ctx.arc(sx, sy, 5.5, 0, 7); ctx.fill();
    ctx.fillStyle = '#04120d';
    ctx.beginPath(); ctx.arc(sx, sy, 2, 0, 7); ctx.fill();
  }

  // 목적지 마커 (크로스헤어)
  if (state.dest && state.dest.floor === state.floor) {
    const [sx, sy] = worldToScreen(state.dest.pos[0], state.dest.pos[1]);
    ctx.strokeStyle = '#ff6b5e'; ctx.fillStyle = '#ff6b5e'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(sx, sy, 9, 0, 7); ctx.stroke();
    ctx.beginPath(); ctx.arc(sx, sy, 2.5, 0, 7); ctx.fill();
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      ctx.beginPath();
      ctx.moveTo(sx + dx * 6, sy + dy * 6); ctx.lineTo(sx + dx * 13, sy + dy * 13);
      ctx.stroke();
    }
  }

  requestAnimationFrame(draw);
}

// ── 입력 (팬/줌/클릭/호버) ────────────────────────
let dragging = false, moved = false, lastX = 0, lastY = 0;
canvas.addEventListener('mousedown', e => { dragging = true; moved = false; lastX = e.offsetX; lastY = e.offsetY; });
window.addEventListener('mouseup', () => { dragging = false; canvas.classList.remove('panning'); });
canvas.addEventListener('mousemove', e => {
  if (dragging) {
    const dx = e.offsetX - lastX, dy = e.offsetY - lastY;
    if (Math.abs(dx) + Math.abs(dy) > 2) { moved = true; canvas.classList.add('panning'); }
    state.view.ox += dx; state.view.oy += dy;
    lastX = e.offsetX; lastY = e.offsetY;
    return;
  }
  const [x, y] = imgToWorld(...screenToImg(e.offsetX, e.offsetY));
  setHover(roomAt(x, y));
  if (state.hoverRoom) {
    const r = META.rooms[state.hoverRoom];
    tooltip.innerHTML =
      `<div class="tt-room">방 ${state.hoverRoom}</div>` +
      `<div class="tt-line">${FLOOR_NAME[r.floor]} · ${roomSize(state.hoverRoom)}</div>` +
      `<div class="tt-line">통로 ${(r.passages || []).length}개 · 클릭하여 목적지 지정</div>`;
    tooltip.hidden = false;
    tooltip.style.left = Math.min(e.offsetX + 16, wrap.clientWidth - 170) + 'px';
    tooltip.style.top = (e.offsetY + 16) + 'px';
  } else tooltip.hidden = true;
});
canvas.addEventListener('mouseleave', () => { tooltip.hidden = true; setHover(null); });
canvas.addEventListener('click', e => {
  if (moved) return; // 드래그였음
  const [x, y] = imgToWorld(...screenToImg(e.offsetX, e.offsetY));
  const room = roomAt(x, y);
  if (!room) { toast('방 안쪽을 클릭해 주세요'); return; }
  state.dest = { room, floor: META.rooms[room].floor, pos: [+x.toFixed(2), +y.toFixed(2), META.rooms[room].center[2]] };
  state.planned = false;
  renderMission();
});
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  zoomAt(e.offsetX, e.offsetY, e.deltaY < 0 ? 1.15 : 1 / 1.15);
}, { passive: false });
document.getElementById('zoom-in').onclick = () => zoomAt(wrap.clientWidth / 2, wrap.clientHeight / 2, 1.3);
document.getElementById('zoom-out').onclick = () => zoomAt(wrap.clientWidth / 2, wrap.clientHeight / 2, 1 / 1.3);
document.getElementById('zoom-fit').onclick = fitView;

function setHover(id) {
  if (id === state.hoverRoom) return;
  state.hoverRoom = id;
  document.querySelectorAll('#room-list li').forEach(li => li.classList.toggle('hover', li.dataset.id === id));
}

// ── 층 탭 / 방 목록 ──────────────────────────────
function setFloor(f) {
  state.floor = f;
  document.querySelectorAll('.floor-tab').forEach(b => b.classList.toggle('active', +b.dataset.floor === f));
  renderRoomList();
  fitView();
}
function renderFloorTabs() {
  const el = document.getElementById('floor-tabs');
  el.innerHTML = '';
  for (const f of Object.keys(META.floors).map(Number).sort()) {
    const b = document.createElement('button');
    b.className = 'floor-tab'; b.dataset.floor = f;
    b.innerHTML = `${FLOOR_NAME[f]}<span class="ft-n">${META.floors[f].rooms.length}개 방</span>`;
    b.onclick = () => setFloor(f);
    el.appendChild(b);
  }
}
function renderRoomList() {
  const ul = document.getElementById('room-list');
  document.getElementById('room-count').textContent = `— ${FLOOR_NAME[state.floor]}`;
  ul.innerHTML = '';
  for (const id of fl().rooms) {
    const li = document.createElement('li');
    li.dataset.id = id;
    li.innerHTML = `<span class="rl-id">방 ${id}</span><span class="rl-size">${roomSize(id)}</span>`;
    if (id === drone.room) li.classList.add('drone-here');
    if (state.dest && state.dest.room === id) li.classList.add('dest');
    li.onmouseenter = () => { state.hoverRoom = id; };
    li.onmouseleave = () => { state.hoverRoom = null; };
    li.onclick = () => focusRoom(id);
    ul.appendChild(li);
  }
}

// ── 미션 패널 ────────────────────────────────────
const fmt = p => `(${p[0].toFixed(2)}, ${p[1].toFixed(2)}, ${p[2].toFixed(2)})`;
function renderMission() {
  document.getElementById('drone-loc').textContent = `방 ${drone.room} · ${FLOOR_NAME[drone.floor]}`;
  document.getElementById('drone-coord').textContent = fmt(drone.pos);
  document.getElementById('wp-start-room').textContent = `방 ${drone.room} · ${FLOOR_NAME[drone.floor]}`;
  document.getElementById('wp-start-coord').textContent = fmt(drone.pos);

  const room = document.getElementById('wp-dest-room');
  const coord = document.getElementById('wp-dest-coord');
  const clear = document.getElementById('dest-clear');
  const cross = document.getElementById('cross-floor');
  const payload = document.getElementById('payload');
  if (state.dest) {
    room.textContent = `방 ${state.dest.room} · ${FLOOR_NAME[state.dest.floor]}`;
    room.classList.remove('empty');
    coord.textContent = fmt(state.dest.pos);
    clear.hidden = false;
    cross.hidden = state.dest.floor === drone.floor;
    planBtn.disabled = false;
    payload.textContent = JSON.stringify({
      start: drone.pos, goal: state.dest.pos,
      start_room: drone.room, goal_room: state.dest.room,
    }, null, 1);
  } else {
    room.textContent = '평면도에서 지점을 클릭하세요';
    room.classList.add('empty');
    coord.textContent = '';
    clear.hidden = true;
    cross.hidden = true;
    planBtn.disabled = true;
    payload.textContent = '목적지를 선택하면 표시됩니다';
  }
  renderRoomList();
}
document.getElementById('dest-clear').onclick = () => { state.dest = null; state.planned = false; renderMission(); };
planBtn.onclick = () => {
  state.planned = true;
  toast('경로계획 요청 (목업) — 백엔드 POST /plan 연동 지점입니다');
};

// ── 자연어 명령 (목업) ───────────────────────────
document.getElementById('nl-form').addEventListener('submit', e => {
  e.preventDefault();
  const q = document.getElementById('nl-input').value.trim();
  if (q) toast(`"${q}" — 자연어 명령 해석은 백엔드 연동 예정입니다`);
});

// ── 토스트 ───────────────────────────────────────
let toastTimer;
function toast(msg) {
  const el = document.getElementById('toast');
  el.textContent = msg; el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 2600);
}

// ── 초기화 ───────────────────────────────────────
for (const [f, src] of Object.entries(FLOOR_IMG)) {
  const im = new Image();
  im.src = src;
  state.imgs[f] = im;
}
window.addEventListener('resize', () => { resize(); fitView(); });
resize();
renderFloorTabs();
setFloor(drone.floor);
renderMission();
requestAnimationFrame(draw);
