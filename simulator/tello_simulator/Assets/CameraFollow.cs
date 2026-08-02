using UnityEngine;

// Drone camera with three modes:
//   ThirdPerson — game-style chase cam behind the drone's MOVEMENT direction
//   FirstPerson — drone nose cam facing the movement direction; C key toggles
//   Preview     — while TelloSimulator.previewActive, the camera flies to
//                 frame the candidate target so the user can confirm it
//
// The simulated Tello translates without yawing (follow_path sends yaw=0), so
// the drone's transform rotation is NOT its heading. The chase cam derives the
// heading from the drone's velocity (position delta) instead.
public class CameraFollow : MonoBehaviour
{
    public enum CamMode { ThirdPerson, FirstPerson }

    [Header("Target to Follow")]
    public Transform target; // 드론 오브젝트를 Inspector에서 여기에 드래그 앤 드롭 하세요.

    [Header("Mode")]
    public CamMode mode = CamMode.ThirdPerson;
    public KeyCode toggleKey = KeyCode.C;

    [Header("Third Person")]
    public Vector3 offset = new Vector3(0, 0.5f, -10f); // 3인칭 원거리

    [Header("First Person")]
    public Vector3 fpvOffset = new Vector3(0, 0.2f, 0.4f); // 드론 앞머리 기준

    [Header("Preview")]
    [Tooltip("Auto-resolved from `target` when left empty.")]
    public TelloSimulator sim;
    public float previewDistance = 8f;   // 타겟에서 수평으로 물러나는 거리
    public float previewHeight = 4f;     // 타겟 위로 올라가는 높이

    [Header("Heading (velocity-based)")]
    [Tooltip("How fast the chase direction turns to follow the drone's motion.")]
    public float headingSmoothSpeed = 4.0f;
    [Tooltip("Below this speed (u/s) the heading is held (drone hovering).")]
    public float minMoveSpeed = 0.15f;

    [Header("Follow Physics")]
    public float smoothSpeed = 5.0f; // 위치 추적 속도
    public float rotationSmoothSpeed = 5.0f; // 회전 추적 속도

    private Vector3 prevTargetPos;
    private Vector3 heading = Vector3.forward; // smoothed horizontal travel direction

    void Start()
    {
        if (sim == null && target != null)
        {
            sim = target.GetComponent<TelloSimulator>();
        }
        if (target != null)
        {
            prevTargetPos = target.position;
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

        // Update the heading from horizontal velocity; hold it while hovering.
        Vector3 vel = (target.position - prevTargetPos) / Mathf.Max(Time.deltaTime, 1e-4f);
        vel.y = 0f;
        if (vel.magnitude > minMoveSpeed)
        {
            heading = Vector3.Slerp(heading, vel.normalized,
                                    Time.deltaTime * headingSmoothSpeed);
        }
        prevTargetPos = target.position;

        Vector3 desiredPosition;
        Quaternion desiredRotation;

        if (sim != null && sim.previewActive)
        {
            // Frame the candidate: back off horizontally toward the drone so
            // the flight direction is implied, then look at the target.
            Vector3 back = target.position - sim.previewTarget;
            back.y = 0f;
            back = back.sqrMagnitude < 0.01f ? Vector3.back : back.normalized;
            desiredPosition = sim.previewTarget + back * previewDistance + Vector3.up * previewHeight;
            desiredRotation = Quaternion.LookRotation(sim.previewTarget - desiredPosition);
        }
        else
        {
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
