"""web/*.dc.html 정적 점검: JS 구문 / 템플릿 변수 / 태그 균형 / this.X 미정의"""
import re, io, os, sys, subprocess, tempfile
FILES = ["web/HAUNTED OPS.dc.html", "web/드론 관제.dc.html"]
ok = True
for f in FILES:
    s = io.open(f, encoding='utf-8').read()
    tpl = s.split('</x-dc>')[0]
    js = re.search(r'<script type="text/x-dc"[^>]*>(.*?)</script>', s, re.S).group(1)
    print(f)
    p = os.path.join(tempfile.gettempdir(), re.sub(r'\W', '_', f) + ".js")
    io.open(p, 'w', encoding='utf-8').write(
        "class DCLogic{}\nvar React={createRef(){},createElement(){}};\n"
        "var window={},navigator={},fetch=()=>{},setInterval=()=>{},clearInterval=()=>{},"
        "setTimeout=()=>{},clearTimeout=()=>{},localStorage={getItem(){},setItem(){}};\n" + js)
    r = subprocess.run(["node", "--check", p], capture_output=True, text=True)
    print("  JS 구문:", "OK" if r.returncode == 0 else "FAIL\n" + r.stderr[:600])
    ok &= r.returncode == 0

    al = set(re.findall(r'as="(\w+)"', tpl))
    toks = {t for t in re.findall(r'\{\{\s*([A-Za-z_$][\w$]*)', tpl) if '.' not in t} - al - {'true', 'false'}
    prov = (set(re.findall(r'vals\.(\w+)\s*=', js)) | set(re.findall(r'[{,]\s*(\w+)\s*:', js))
            | set(re.findall(r'\b(\w+),', js)) | set(re.findall(r'^\s*(\w+):', js, re.M)))
    miss = sorted(t for t in toks if t not in prov)
    print("  템플릿 변수:", "OK" if not miss else f"미제공 {miss}")
    ok &= not miss

    bal = all(tpl.count("<" + t) == tpl.count("</" + t + ">") for t in ("sc-if", "sc-for"))
    print("  태그 균형:", "OK" if bal else "FAIL")
    ok &= bal

    # ★ this.X 가 실제로 존재하는가 (상수 이름 바꾸고 놓친 참조를 잡는다)
    body = js[js.index('class Component'):]
    defined = set(re.findall(r'^  ([A-Za-z_$][\w$]*)\s*[=(]', body, re.M))
    defined |= set(re.findall(r'\bthis\.([A-Za-z_$][\w$]*)\s*=', body))   # 메서드 안에서 대입
    used = set(re.findall(r'\bthis\.([A-Za-z_$][\w$]*)', body))
    miss2 = sorted(used - defined - {'state', 'setState', 'props', 'forceUpdate'})
    print("  this.X 참조:", "OK" if not miss2 else f"미정의 {miss2}")
    ok &= not miss2
print("=>", "전체 OK" if ok else "실패")
sys.exit(0 if ok else 1)
