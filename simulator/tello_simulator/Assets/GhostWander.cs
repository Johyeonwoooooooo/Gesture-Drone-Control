using UnityEngine;

/// <summary>
/// GhostWander — 집 안을 부드럽게 배회하는 유령 AI.
/// 속도 벡터를 서서히 꺾어 자연스러운 곡선 경로를 만든다.
/// </summary>
public class GhostWander : MonoBehaviour
{
    [Header("배회 범위")]
    public Vector3 wanderExtents = new Vector3(10f, 3f, 10f);
    [Tooltip("XZ 범위 이탈 시 중심으로 귀환하는 힘")]
    public float returnForce = 2f;

    private Vector3 wanderCenter;

    [Header("이동")]
    public float moveSpeed  = 2f;
    [Tooltip("초당 최대 방향 전환 각도 (낮을수록 완만한 곡선)")]
    public float turnSpeed  = 55f;

    [Header("부유 애니메이션")]
    public float bobAmplitude = 0.2f;
    public float bobFrequency = 0.5f;

    [Header("벽 회피")]
    public float     wallCheckDist = 2f;
    public LayerMask wallMask      = ~0;

    private Vector3 moveDir;      // 현재 이동 방향 (수평)
    private Vector3 targetPos;    // 목표 위치
    private float   baseY;        // bob 기준 높이 (서서히 목표 높이로 이동)

    void Awake()
    {
        wanderCenter = transform.position;
    }

    void Start()
    {
        moveDir = Random.insideUnitSphere;
        moveDir.y = 0f;
        if (moveDir.sqrMagnitude < 0.01f) moveDir = Vector3.forward;
        moveDir.Normalize();

        baseY = transform.position.y;
        PickNewTarget();
    }

    void Update()
    {
        // ── 목표 도착 시 새 목표 선택 (정지 없이) ─────────
        Vector3 pos       = transform.position;
        Vector3 toTarget  = targetPos - pos;
        toTarget.y = 0f;
        if (toTarget.magnitude < 1f)
            PickNewTarget();

        // ── 범위 이탈 시 중심 방향으로 귀환 ───────────────
        Vector3 offset = pos - wanderCenter;
        offset.y = 0f;
        bool outOfBounds = Mathf.Abs(offset.x) > wanderExtents.x
                        || Mathf.Abs(offset.z) > wanderExtents.z
                        || pos.y > wanderCenter.y + wanderExtents.y
                        || pos.y < wanderCenter.y;
        if (outOfBounds)
        {
            Vector3 toCenter = wanderCenter - pos;
            toCenter.y = 0f;
            moveDir = Vector3.RotateTowards(
                moveDir, toCenter.normalized,
                returnForce * Mathf.Deg2Rad * Time.deltaTime * 100f,
                0f).normalized;
            PickNewTarget();  // 범위 안 새 목표 즉시 선택
        }

        // ── 벽 감지: 전방 ray → 감지 시 목표 재설정 ───────
        if (Physics.Raycast(pos, moveDir, wallCheckDist, wallMask, QueryTriggerInteraction.Ignore))
            PickNewTarget();

        // ── 방향을 목표 쪽으로 서서히 꺾기 ────────────────
        Vector3 desired = toTarget.magnitude > 0.1f ? toTarget.normalized : moveDir;
        moveDir = Vector3.RotateTowards(
            moveDir, desired,
            turnSpeed * Mathf.Deg2Rad * Time.deltaTime,
            0f).normalized;

        // ── 수평 이동 ──────────────────────────────────────
        float dx = moveDir.x * moveSpeed * Time.deltaTime;
        float dz = moveDir.z * moveSpeed * Time.deltaTime;

        // ── 높이: baseY를 목표 높이로 서서히 접근 + bob 합산 ─
        baseY = Mathf.Lerp(baseY, targetPos.y, Time.deltaTime * 0.4f);
        float bob = Mathf.Sin(Time.time * bobFrequency * Mathf.PI * 2f) * bobAmplitude;

        transform.position = new Vector3(pos.x + dx, baseY + bob, pos.z + dz);

        // ── 회전: 이동 방향을 바라봄 ──────────────────────
        Quaternion targetRot = Quaternion.LookRotation(moveDir, Vector3.up);
        transform.rotation = Quaternion.Slerp(
            transform.rotation, targetRot, Time.deltaTime * 3f);
    }

    void PickNewTarget()
    {
        targetPos = wanderCenter + new Vector3(
            Random.Range(-wanderExtents.x, wanderExtents.x),
            Random.Range(0f, wanderExtents.y),          // Y는 위쪽으로만
            Random.Range(-wanderExtents.z, wanderExtents.z));
    }

    void OnTriggerEnter(Collider other)
    {
        TelloSimulator sim = other.GetComponentInParent<TelloSimulator>();
        if (sim != null)
            sim.RegisterGhostCollision();
    }

    void OnDrawGizmosSelected()
    {
        Vector3 center = Application.isPlaying ? wanderCenter : transform.position;
        // Y는 위쪽으로만: 박스 중심을 extents.y 절반만큼 올림
        Vector3 boxCenter = center + Vector3.up * (wanderExtents.y * 0.5f);
        Vector3 boxSize   = new Vector3(wanderExtents.x * 2f, wanderExtents.y, wanderExtents.z * 2f);
        Gizmos.color = new Color(0.5f, 0f, 1f, 0.2f);
        Gizmos.DrawCube(boxCenter, boxSize);
        Gizmos.color = Color.magenta;
        Gizmos.DrawSphere(targetPos, 0.25f);
        Gizmos.color = Color.cyan;
        Gizmos.DrawRay(transform.position, moveDir * wallCheckDist);
    }
}
