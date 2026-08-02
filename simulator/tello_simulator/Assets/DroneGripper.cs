using System.Collections;
using UnityEngine;

/// <summary>
/// DroneGripper — 드론에 붙이는 4-핑거 L자형 집게발 컴포넌트.
/// 집기: 그리퍼가 아래로 뻗어 물체에 닿은 후 손가락을 닫고 올라옴.
/// 놓기: 손가락을 열고 물체를 분리(중력 적용).
/// </summary>
[ExecuteAlways]
public class DroneGripper : MonoBehaviour
{
    // ─── 위치 ────────────────────────────────────────────
    [Header("그리퍼 위치 오프셋")]
    [Tooltip("드론 기준 그리퍼 루트 로컬 위치. Y 음수 = 아래로.")]
    public Vector3 gripperOffset = new Vector3(0f, -0.3f, 0f);

    // ─── 크기 ────────────────────────────────────────────
    [Header("그리퍼 크기")]
    public float bodyLength    = 0.4f;
    public float bodyRadius    = 0.05f;
    public float fingerLength  = 0.3f;
    public float fingerThickness = 0.04f;
    public float fingerSpread  = 0.1f;

    // ─── 여닫힘 ─────────────────────────────────────────
    [Header("여닫힘 각도 & 속도")]
    [Tooltip("열림 각도 (°)")]
    public float openAngle  = 35f;
    [Tooltip("닫힘 각도 (°) — 음수 = 안쪽")]
    public float closeAngle = -12f;
    [Tooltip("여닫힘 Lerp 속도")]
    public float lerpSpeed  = 8f;

    // ─── 집기 ────────────────────────────────────────────
    [Header("집기 설정")]
    [Tooltip("HoldPoint 기준 집기 판정 반경")]
    public float grabRange = 0.5f;
    [Tooltip("그리퍼가 최대로 뻗을 수 있는 거리")]
    public float maxExtend = 2f;
    [Tooltip("뻗기/접기 속도")]
    public float extendSpeed = 4f;
    public LayerMask grabMask = ~0;

    // ─── 색상 ────────────────────────────────────────────
    [Header("색상")]
    public Color gripperColor = new Color(0.18f, 0.18f, 0.18f);

    // ─── 내부 상태 ───────────────────────────────────────
    private Transform   holdPoint;
    private Transform[] fingerPivots;
    private Vector3[]   fingerTiltAxes;
    private float currentAngle;
    private float targetAngle;

    private Transform neckTransform;
    private Transform fingersTransform;
    private float neckDefaultScaleY;
    private float neckDefaultLocalY;
    private float fingersDefaultLocalY;

    private GameObject heldObject;
    private Rigidbody  heldRb;
    private Collider[] heldColliders;
    private Collider[] gripperColliders;
    private bool       isAnimating = false;
    private Coroutine  activeCoroutine;

    // ─── 공개 프로퍼티 ───────────────────────────────────
    public bool IsHolding  => heldObject != null;
    public bool IsAnimating => isAnimating;
    public bool CanGrab()  => FindPickable() != null;

    // ════════════════════════════════════════════════════
    void Awake()
    {
        RebuildGripper();
    }

    void OnValidate()
    {
        // Inspector에서 값 바꿀 때 Edit 모드에서도 즉시 반영
#if UNITY_EDITOR
        if (!Application.isPlaying)
            UnityEditor.EditorApplication.delayCall += RebuildGripper;
#endif
    }

    void RebuildGripper()
    {
        if (this == null) return; // 오브젝트가 삭제된 경우 방어
        Transform existing = transform.Find("Gripper");
        if (existing != null) DestroyImmediate(existing.gameObject);
        BuildGripperHierarchy();
        gripperColliders = null;
        currentAngle = openAngle;
        targetAngle  = openAngle;
    }

    void Update()
    {
        if (!Application.isPlaying) return;
        HandleInput();
        AnimateFingers();
    }

    // ─── 스페이스바 입력 ──────────────────────────────────
    void HandleInput()
    {
        var kb = UnityEngine.InputSystem.Keyboard.current;
        if (kb != null && kb.spaceKey.wasPressedThisFrame)
            Toggle();
    }

    // ─── 손가락 보간 ─────────────────────────────────────
    void AnimateFingers()
    {
        if (fingerPivots == null) return;
        currentAngle = Mathf.Lerp(currentAngle, targetAngle, Time.deltaTime * lerpSpeed);
        for (int i = 0; i < fingerPivots.Length; i++)
            if (fingerPivots[i] != null)
                fingerPivots[i].localRotation =
                    Quaternion.AngleAxis(currentAngle, fingerTiltAxes[i]);
    }

    // ════════════════════════════════════════════════════
    // 공개 메서드
    // ════════════════════════════════════════════════════

    public void Grab()
    {
        if (IsHolding || isAnimating) return;
        GameObject target = FindPickable();
        if (target == null) return;
        if (activeCoroutine != null) StopCoroutine(activeCoroutine);
        activeCoroutine = StartCoroutine(GrabSequence(target));
    }

    public void Release()
    {
        if (!IsHolding || isAnimating) return;
        if (activeCoroutine != null) StopCoroutine(activeCoroutine);
        activeCoroutine = StartCoroutine(ReleaseSequence());
    }

    public void Toggle()
    {
        if (isAnimating) return;
        if (IsHolding) Release();
        else Grab();
    }

    // ════════════════════════════════════════════════════
    // 집기 코루틴: 뻗기 → 손가락 닫기 → 부착 → 접기
    // ════════════════════════════════════════════════════
    IEnumerator GrabSequence(GameObject target)
    {
        isAnimating = true;

        // 수직 거리 계산 — 물체 상단(bounds.max.y) 기준
        Collider targetCol = target.GetComponent<Collider>()
                          ?? target.GetComponentInChildren<Collider>();
        float topY    = targetCol != null ? targetCol.bounds.max.y : target.transform.position.y;
        float pivotY  = target.transform.position.y;
        // holdPoint에 부착 시 피벗이 아닌 상단이 holdPoint에 오도록 할 오프셋
        float hangOffset = topY - pivotY;   // 피벗→상단 거리 (월드)

        float extend = Mathf.Clamp(fingersTransform.position.y - topY, 0f, maxExtend);

        // 1. 아래로 뻗기
        yield return StartCoroutine(SetExtend(extend));

        // 2. 손가락 닫기
        targetAngle = closeAngle;
        yield return new WaitForSeconds(0.35f);

        // 3. 오브젝트 부착 — 상단이 holdPoint에 오도록 아래로 오프셋
        heldObject = target;
        heldRb = target.GetComponent<Rigidbody>() ?? target.GetComponentInParent<Rigidbody>();
        if (heldRb != null) heldRb.isKinematic = true;
        heldColliders = heldObject.GetComponentsInChildren<Collider>(true);
        SetCollidersEnabled(heldColliders, false);
        heldObject.transform.SetParent(holdPoint);
        heldObject.transform.localPosition = Vector3.down * hangOffset;
        heldObject.transform.localRotation = Quaternion.identity;

        // 4. 접기
        yield return StartCoroutine(SetExtend(0f));

        isAnimating = false;
    }

    // ════════════════════════════════════════════════════
    // 놓기 코루틴: 손가락 열기 → 분리(중력) → 접기
    // ════════════════════════════════════════════════════
    IEnumerator ReleaseSequence()
    {
        isAnimating = true;

        // 1. 손가락 열기
        targetAngle = openAngle;
        yield return new WaitForSeconds(0.3f);

        // 2. 분리
        GameObject dropping   = heldObject;
        Rigidbody  droppingRb = heldRb;
        heldObject = null;
        heldRb     = null;

        Collider[] droppingCols = heldColliders;
        heldColliders = null;

        dropping.transform.SetParent(null, true);

        // 콜라이더를 꺼둔 채 중력을 켜면 바닥을 통과해 사라진 것처럼 보인다.
        // 분리 직후 복원하고, 드론/집게발과의 순간 겹침만 짧게 무시한다.
        SetCollidersEnabled(droppingCols, true);
        IgnoreCollisionsWithGripper(droppingCols, true);

        if (droppingRb != null)
        {
            droppingRb.isKinematic    = false;
            droppingRb.useGravity     = true;
            droppingRb.linearVelocity = Vector3.zero;
        }

        // 3. 그리퍼 접기
        yield return StartCoroutine(SetExtend(0f));

        // 4. 추가 대기 — 물체가 중력으로 드론 아래 완전히 벗어날 때까지
        yield return new WaitForSeconds(0.8f);

        // 5. 드론/집게발 충돌 복원
        IgnoreCollisionsWithGripper(droppingCols, false);

        isAnimating = false;
    }

    void SetCollidersEnabled(Collider[] colliders, bool enabled)
    {
        if (colliders == null) return;
        foreach (Collider c in colliders)
            if (c != null) c.enabled = enabled;
    }

    void IgnoreCollisionsWithGripper(Collider[] objectColliders, bool ignore)
    {
        if (objectColliders == null) return;
        if (gripperColliders == null || gripperColliders.Length == 0)
            gripperColliders = GetComponentsInChildren<Collider>(true);

        foreach (Collider objectCol in objectColliders)
        {
            if (objectCol == null || !objectCol.enabled) continue;
            foreach (Collider gripperCol in gripperColliders)
            {
                if (gripperCol == null || !gripperCol.enabled || objectCol == gripperCol) continue;
                Physics.IgnoreCollision(objectCol, gripperCol, ignore);
            }
        }
    }


    // ─── 목통 + Fingers 위치 애니메이션 ─────────────────
    IEnumerator SetExtend(float targetExtend)
    {
        float startFingersY   = fingersTransform.localPosition.y;
        float targetFingersY  = fingersDefaultLocalY - targetExtend;

        float startNeckScaleY = neckTransform.localScale.y;
        float targetNeckScaleY = neckDefaultScaleY + targetExtend;

        float startNeckY  = neckTransform.localPosition.y;
        float targetNeckY = neckDefaultLocalY - targetExtend * 0.5f;

        float t = 0f;
        while (t < 1f)
        {
            t += Time.deltaTime * extendSpeed;
            float e = Mathf.SmoothStep(0f, 1f, Mathf.Clamp01(t));

            fingersTransform.localPosition = new Vector3(
                0f, Mathf.Lerp(startFingersY, targetFingersY, e), 0f);

            neckTransform.localScale = new Vector3(
                neckTransform.localScale.x,
                Mathf.Lerp(startNeckScaleY, targetNeckScaleY, e),
                neckTransform.localScale.z);

            neckTransform.localPosition = new Vector3(
                0f, Mathf.Lerp(startNeckY, targetNeckY, e), 0f);

            yield return null;
        }
    }

    // ─── Pickable 탐색 ───────────────────────────────────
    GameObject FindPickable()
    {
        if (holdPoint == null) return null;
        Collider[] hits = Physics.OverlapSphere(
            holdPoint.position, grabRange, grabMask,
            QueryTriggerInteraction.Ignore);
        GameObject closest = null;
        float minDist = float.MaxValue;
        foreach (Collider col in hits)
        {
            if (!col.CompareTag("Pickable")) continue;

            Rigidbody rb = col.attachedRigidbody;
            GameObject candidate = rb != null ? rb.gameObject : col.gameObject;
            float d = Vector3.Distance(holdPoint.position, col.transform.position);
            if (d < minDist) { minDist = d; closest = candidate; }
        }
        return closest;
    }

    // ─── HUD ─────────────────────────────────────────────
    void OnGUI()
    {
        if (!IsHolding) return;
        GUIStyle style = new GUIStyle(GUI.skin.box);
        style.fontSize  = 22;
        style.alignment = TextAnchor.MiddleCenter;
        GUI.color = new Color(0.2f, 1f, 0.4f);
        float w = 340f;
        GUI.Box(new Rect((Screen.width - w) / 2f, Screen.height - 60f, w, 44f),
                $"Carrying: {heldObject.name}", style);
        GUI.color = Color.white;
    }

    // ════════════════════════════════════════════════════
    // 그리퍼 계층 자동 생성 (이미지 참고: 플레이트+목통+L자형 4발톱)
    // ════════════════════════════════════════════════════
    float CalcDroneScale()
    {
        // 드론 Renderer 전체 bounds에서 XZ 최대값 기준 스케일 계산
        // 그리퍼가 없는 Renderer만 대상으로 함
        Bounds b = new Bounds(transform.position, Vector3.zero);
        bool found = false;
        foreach (Renderer r in GetComponentsInChildren<Renderer>(true))
        {
            // 그리퍼 자식은 제외
            if (r.transform.IsChildOf(transform.Find("Gripper") ?? transform) &&
                r.gameObject.name.StartsWith("Gripper")) continue;
            if (!found) { b = r.bounds; found = true; }
            else b.Encapsulate(r.bounds);
        }
        float xzSpan = Mathf.Max(b.size.x, b.size.z);
        // 기준 드론 XZ 크기 1.5m → 그 배율만큼 그리퍼도 키움
        return Mathf.Clamp(xzSpan / 1.5f, 0.3f, 10f);
    }

    void BuildGripperHierarchy()
    {
        Material mat = MakeMaterial(gripperColor);

        float s = CalcDroneScale();

        float plateW  = 0.28f * s;
        float plateH  = 0.03f * s;
        float neckW   = 0.06f * s;
        float neckH   = 0.18f * s;
        float fW      = 0.04f * s;
        float fUpperH = 0.12f * s;
        float fLowerH = 0.09f * s;
        float fSpread = 0.05f * s;

        // 루트 — offset도 스케일에 비례
        GameObject root = new GameObject("Gripper");
        root.transform.SetParent(transform);
        root.transform.localPosition = gripperOffset * s;
        root.transform.localRotation = Quaternion.identity;

        // 상단 플레이트
        GameObject plate = GameObject.CreatePrimitive(PrimitiveType.Cube);
        plate.name = "TopPlate";
        DestroyImmediate(plate.GetComponent<Collider>());
        plate.transform.SetParent(root.transform);
        plate.transform.localPosition = Vector3.zero;
        plate.transform.localScale    = new Vector3(plateW, plateH, plateW);
        plate.transform.localRotation = Quaternion.identity;
        ApplyMaterial(plate, mat);

        // 목통 (Neck) — 뻗기 애니메이션에서 스케일 조절됨
        GameObject neck = GameObject.CreatePrimitive(PrimitiveType.Cube);
        neck.name = "Neck";
        DestroyImmediate(neck.GetComponent<Collider>());
        neck.transform.SetParent(root.transform);
        neck.transform.localPosition = new Vector3(0f, -(plateH * 0.5f + neckH * 0.5f), 0f);
        neck.transform.localScale    = new Vector3(neckW, neckH, neckW);
        neck.transform.localRotation = Quaternion.identity;
        ApplyMaterial(neck, mat);
        neckTransform     = neck.transform;
        neckDefaultScaleY = neckH;          // Cylinder/Cube scale.y = 실제 높이 (Cube는 1:1)
        neckDefaultLocalY = neck.transform.localPosition.y;

        // Fingers 기준점
        GameObject fingersGO = new GameObject("Fingers");
        fingersGO.transform.SetParent(root.transform);
        fingersGO.transform.localPosition = new Vector3(0f, -(plateH * 0.5f + neckH), 0f);
        fingersGO.transform.localRotation = Quaternion.identity;
        fingersTransform     = fingersGO.transform;
        fingersDefaultLocalY = fingersGO.transform.localPosition.y;

        // HoldPoint
        GameObject holdGO = new GameObject("HoldPoint");
        holdGO.transform.SetParent(fingersGO.transform);
        holdGO.transform.localPosition = new Vector3(0f, -fUpperH * 0.6f, 0f);
        holdGO.transform.localRotation = Quaternion.identity;
        holdPoint = holdGO.transform;

        // L자형 손가락 4개
        Vector3[] dirs = { Vector3.forward, Vector3.back, Vector3.left, Vector3.right };
        fingerPivots   = new Transform[4];
        fingerTiltAxes = new Vector3[4];

        for (int i = 0; i < 4; i++)
        {
            Vector3 outward = dirs[i];
            fingerTiltAxes[i] = Vector3.Cross(outward, Vector3.up).normalized;

            GameObject pivot = new GameObject($"FingerPivot_{i}");
            pivot.transform.SetParent(fingersGO.transform);
            pivot.transform.localPosition = outward * fSpread;
            pivot.transform.localRotation = Quaternion.identity;
            fingerPivots[i] = pivot.transform;

            // 세로 세그먼트
            GameObject upper = GameObject.CreatePrimitive(PrimitiveType.Cube);
            upper.name = $"FingerUpper_{i}";
            DestroyImmediate(upper.GetComponent<Collider>());
            upper.transform.SetParent(pivot.transform);
            upper.transform.localPosition = new Vector3(0f, -fUpperH * 0.5f, 0f);
            upper.transform.localScale    = new Vector3(fW, fUpperH, fW);
            upper.transform.localRotation = Quaternion.identity;
            ApplyMaterial(upper, mat);

            // 가로 세그먼트 (L자 꺾인 부분)
            GameObject lower = GameObject.CreatePrimitive(PrimitiveType.Cube);
            lower.name = $"FingerLower_{i}";
            DestroyImmediate(lower.GetComponent<Collider>());
            lower.transform.SetParent(pivot.transform);
            lower.transform.localPosition = new Vector3(
                outward.x * fLowerH * 0.5f, -fUpperH,
                outward.z * fLowerH * 0.5f);
            lower.transform.localScale    = new Vector3(fW, fLowerH, fW);
            lower.transform.localRotation = Quaternion.FromToRotation(Vector3.up, outward);
            ApplyMaterial(lower, mat);
        }
    }

    // ─── 머티리얼 ─────────────────────────────────────────
    Material MakeMaterial(Color color)
    {
        Shader urp = Shader.Find("Universal Render Pipeline/Lit")
                  ?? Shader.Find("Universal Render Pipeline/Simple Lit");
        if (urp != null)
        {
            var mat = new Material(urp);
            mat.SetColor("_BaseColor", color);
            return mat;
        }
        Shader builtin = Shader.Find("Standard") ?? Shader.Find("Sprites/Default");
        var m = new Material(builtin);
        m.color = color;
        return m;
    }

    void ApplyMaterial(GameObject go, Material mat)
    {
        var r = go.GetComponent<Renderer>();
        if (r != null) r.material = mat;
    }

    // ─── 기즈모 ──────────────────────────────────────────
    void OnDrawGizmosSelected()
    {
        Vector3 center = holdPoint != null
            ? holdPoint.position
            : transform.position + gripperOffset + Vector3.down * 0.5f;
        Gizmos.color = IsHolding ? Color.green : new Color(1f, 0.6f, 0f, 0.5f);
        Gizmos.DrawWireSphere(center, grabRange);
    }
}
