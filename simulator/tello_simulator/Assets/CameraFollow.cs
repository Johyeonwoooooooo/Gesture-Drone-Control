using UnityEngine;

// Drone camera with two modes:
//   ThirdPerson — game-style chase cam behind the drone's MOVEMENT direction
//   FirstPerson — drone nose cam facing the movement direction; C key toggles
//
// The simulated Tello translates without yawing (follow_path sends yaw=0), so
// the drone's transform rotation is NOT its heading. The chase cam derives the
// heading from the drone's velocity (position delta) instead.
//
// Exception — the patrol 360° scan (patrol_mission.scan_360 sends `rc 0 0 0 yaw`):
// there the body spins while the velocity is ~0, so a purely velocity-derived
// heading would leave the view frozen and the sweep invisible. While hovering
// AND actually yawing, the heading follows the drone's own rotation instead.
public class CameraFollow : MonoBehaviour
{
    public enum CamMode { ThirdPerson, FirstPerson }

    [Header("Target to Follow")]
    public Transform target; // 드론 오브젝트를 Inspector에서 여기에 드래그 앤 드롭 하세요.

    [Header("Mode")]
    public CamMode mode = CamMode.ThirdPerson;
    public KeyCode toggleKey = KeyCode.C;

    [Header("Third Person")]
    public Vector3 offset = new Vector3(0, 1.5f, -4.0f); // 이동방향 기준 (위 1.5, 뒤 4)

    [Header("First Person")]
    public Vector3 fpvOffset = new Vector3(0, 0.2f, 0.4f); // 드론 앞머리 기준

    [Header("Heading (velocity-based)")]
    [Tooltip("How fast the chase direction turns to follow the drone's motion.")]
    public float headingSmoothSpeed = 4.0f;
    [Tooltip("Below this speed (u/s) the heading is held (drone hovering).")]
    public float minMoveSpeed = 0.15f;
    [Tooltip("A single-frame position jump larger than this is a teleport, not motion.")]
    public float teleportThreshold = 5f;

    [Header("Heading (in-place yaw)")]
    [Tooltip("While hovering, turn the camera with the drone's own yaw. This is " +
             "what makes the patrol 360° room scan visible — during it the body " +
             "rotates but the velocity is ~0. Turn off to restore the old " +
             "velocity-only behaviour.")]
    public bool followYawWhileHovering = true;
    [Tooltip("Minimum yaw rate (deg/s) that counts as a deliberate in-place turn. " +
             "Gating on the RATE (not just on hovering) matters: the drone flies " +
             "with yaw fixed at 0, so adopting its rotation at every waypoint " +
             "arrival would swing the camera each time it stops.")]
    public float minYawRate = 5f;

    [Header("Follow Physics")]
    public float smoothSpeed = 5.0f; // 위치 추적 속도
    public float rotationSmoothSpeed = 5.0f; // 회전 추적 속도

    private Vector3 prevTargetPos;
    private float prevTargetYaw;
    private Vector3 heading = Vector3.forward; // smoothed horizontal travel direction

    void Start()
    {
        if (target != null)
        {
            prevTargetPos = target.position;
            prevTargetYaw = target.eulerAngles.y;
            if (target.forward.sqrMagnitude > 0.001f)
            {
                Vector3 f = target.forward; f.y = 0f;
                if (f.sqrMagnitude > 0.001f) heading = f.normalized;
            }
        }
    }

    void Update()
    {
        if (TogglePressed())
        {
            mode = mode == CamMode.ThirdPerson ? CamMode.FirstPerson : CamMode.ThirdPerson;
        }
    }

    void LateUpdate()
    {
        if (target == null) return;

        // A `setpos` teleport (home spawn, mission start) moves the drone far in one
        // frame; treating that as velocity whips the chase cam around. Swallow it —
        // and swallow the frame's yaw change with it, since `setpos x y z yaw` can
        // rewrite the rotation in the same frame.
        Vector3 delta = target.position - prevTargetPos;
        bool teleported = delta.magnitude > teleportThreshold;
        if (teleported)
        {
            prevTargetPos = target.position;
            delta = Vector3.zero;
        }

        float targetYaw = target.eulerAngles.y;
        float yawRate = teleported
            ? 0f
            : Mathf.DeltaAngle(prevTargetYaw, targetYaw) / Mathf.Max(Time.deltaTime, 1e-4f);
        prevTargetYaw = targetYaw;

        // Heading: from horizontal velocity while moving, from the drone's own
        // rotation while hovering AND turning (patrol scan), held otherwise.
        Vector3 vel = delta / Mathf.Max(Time.deltaTime, 1e-4f);
        vel.y = 0f;
        if (vel.magnitude > minMoveSpeed)
        {
            heading = Vector3.Slerp(heading, vel.normalized,
                                    Time.deltaTime * headingSmoothSpeed);
        }
        else if (followYawWhileHovering && Mathf.Abs(yawRate) > minYawRate)
        {
            Vector3 forward = target.forward;
            forward.y = 0f;
            if (forward.sqrMagnitude > 0.001f)
            {
                heading = Vector3.Slerp(heading, forward.normalized,
                                        Time.deltaTime * headingSmoothSpeed);
            }
        }
        prevTargetPos = target.position;

        Vector3 desiredPosition;
        Quaternion desiredRotation;

        // Chase orientation from the travel direction (not the drone's yaw).
        Quaternion headRot = Quaternion.LookRotation(heading, Vector3.up);

        if (mode == CamMode.FirstPerson)
        {
            desiredPosition = target.position + headRot * fpvOffset;
            desiredRotation = headRot;
        }
        else
        {
            desiredPosition = target.position + headRot * offset;
            desiredRotation = Quaternion.LookRotation(target.position - desiredPosition, Vector3.up);
        }

        transform.position = Vector3.Lerp(transform.position, desiredPosition, Time.deltaTime * smoothSpeed);
        transform.rotation = Quaternion.Slerp(transform.rotation, desiredRotation, Time.deltaTime * rotationSmoothSpeed);
    }

    // Works whether the project's Active Input Handling is the legacy Input
    // Manager, the new Input System package, or Both.
    bool TogglePressed()
    {
#if ENABLE_INPUT_SYSTEM
        var kb = UnityEngine.InputSystem.Keyboard.current;
        if (kb != null && kb.cKey.wasPressedThisFrame)
        {
            return true;
        }
#endif
#if ENABLE_LEGACY_INPUT_MANAGER
        if (Input.GetKeyDown(toggleKey))
        {
            return true;
        }
#endif
        return false;
    }
}
