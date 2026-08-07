using UnityEngine;

// A "ghost" (유령) that free-roams the scanned environment on its own. It floats at a
// fixed height above the floor and wanders continuously, steering away from walls and
// furniture using raycasts (the scan has no baked NavMesh). It carries no collider, so
// it never trips the drone's collision probe and is never picked up as cargo — it is
// purely a free-roaming presence you can import into the scene at Play time.
//
// Spawned by SimPropManager via the "ghost" UDP command. Tunables are public so they can
// be adjusted in the Inspector on the spawned GhostRoamer if desired.
public class GhostRoamer : MonoBehaviour
{
    [Header("Roaming")]
    [Tooltip("Horizontal drift speed (m/s in the 5x-scaled scan world).")]
    public float moveSpeed = 3.5f;
    [Tooltip("How fast the ghost swings toward a new heading (deg/s).")]
    public float turnSpeed = 120f;
    [Tooltip("Height held above the floor directly below the ghost.")]
    public float hoverHeight = 2.5f;
    [Tooltip("Ghost stays within this radius of where it was spawned.")]
    public float wanderRadius = 25f;

    [Header("Obstacle Avoidance")]
    [Tooltip("Distance ahead scanned for walls/furniture before steering away.")]
    public float lookAhead = 4f;
    [Tooltip("Thickness of the forward probe (spherecast radius).")]
    public float bodyRadius = 0.6f;
    [Tooltip("Layers treated as the environment to avoid and to stand above.")]
    public LayerMask environmentMask = ~0;

    [Header("Feel")]
    [Tooltip("Amplitude of the gentle vertical bob.")]
    public float bobAmplitude = 0.35f;
    public float bobSpeed = 1.5f;

    private Vector3 home;
    private float currentYaw;
    private float targetYaw;
    private float repathTimer;
    private float bobPhase;
    private float smoothBaseY;

    void Start()
    {
        home = transform.position;
        currentYaw = transform.eulerAngles.y;
        targetYaw = currentYaw;
        bobPhase = Random.Range(0f, Mathf.PI * 2f);
        smoothBaseY = transform.position.y;
        PickNewHeading();
    }

    void Update()
    {
        float dt = Time.deltaTime;

        repathTimer -= dt;
        if (repathTimer <= 0f)
        {
            PickNewHeading();
        }

        // Steer back toward home if the ghost has wandered past its leash.
        Vector3 flatToHome = home - transform.position;
        flatToHome.y = 0f;
        if (flatToHome.magnitude > wanderRadius)
        {
            targetYaw = Quaternion.LookRotation(flatToHome).eulerAngles.y;
        }

        // Avoid walls / furniture straight ahead: turn toward whichever side is clearer.
        if (Physics.SphereCast(transform.position, bodyRadius, transform.forward,
                               out _, lookAhead, environmentMask,
                               QueryTriggerInteraction.Ignore))
        {
            bool leftClear = !Physics.SphereCast(transform.position, bodyRadius,
                Quaternion.Euler(0f, -55f, 0f) * transform.forward, out _, lookAhead,
                environmentMask, QueryTriggerInteraction.Ignore);
            float turn = leftClear ? -1f : 1f;
            targetYaw = currentYaw + turn * Random.Range(70f, 130f);
            repathTimer = Mathf.Max(repathTimer, 1.2f);
        }

        // Smoothly rotate toward the target heading and drift forward.
        currentYaw = Mathf.MoveTowardsAngle(currentYaw, targetYaw, turnSpeed * dt);
        transform.rotation = Quaternion.Euler(0f, currentYaw, 0f);
        Vector3 next = transform.position + transform.forward * moveSpeed * dt;

        // Hold altitude above the floor below, with a gentle bob. smoothBaseY tracks the
        // floor-relative baseline; the sine is added as an absolute offset (not scaled by
        // dt) so the bob is actually visible and never accumulates drift.
        bobPhase += bobSpeed * dt;
        float desiredBaseY = smoothBaseY;
        if (Physics.Raycast(next + Vector3.up * 3f, Vector3.down, out RaycastHit floor,
                            60f, environmentMask, QueryTriggerInteraction.Ignore))
        {
            desiredBaseY = floor.point.y + hoverHeight;
        }
        smoothBaseY = Mathf.Lerp(smoothBaseY, desiredBaseY, dt * 2f);
        next.y = smoothBaseY + Mathf.Sin(bobPhase) * bobAmplitude;

        transform.position = next;
    }

    void PickNewHeading()
    {
        targetYaw = Random.Range(0f, 360f);
        repathTimer = Random.Range(2.5f, 5.5f);
    }
}
