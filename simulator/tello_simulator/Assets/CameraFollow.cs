using UnityEngine;

// Drone camera with three modes:
//   ThirdPerson — game-style chase cam behind the drone's heading (default)
//   FirstPerson — drone nose cam; toggle with the C key
//   Preview     — while TelloSimulator.previewActive, the camera flies to
//                 frame the candidate target so the user can confirm it
public class CameraFollow : MonoBehaviour
{
    public enum CamMode { ThirdPerson, FirstPerson }

    [Header("Target to Follow")]
    public Transform target; // 드론 오브젝트를 Inspector에서 여기에 드래그 앤 드롭 하세요.

    [Header("Mode")]
    public CamMode mode = CamMode.ThirdPerson;
    public KeyCode toggleKey = KeyCode.C;

    [Header("Third Person")]
    public Vector3 offset = new Vector3(0, 1.5f, -4.0f); // 드론 기준 상대적 위치 (위로 1.5, 뒤로 4)

    [Header("First Person")]
    public Vector3 fpvOffset = new Vector3(0, 0.2f, 0.4f); // 드론 앞머리 기준

    [Header("Preview")]
    [Tooltip("Auto-resolved from `target` when left empty.")]
    public TelloSimulator sim;
    public float previewDistance = 8f;   // 타겟에서 수평으로 물러나는 거리
    public float previewHeight = 4f;     // 타겟 위로 올라가는 높이

    [Header("Follow Physics")]
    public float smoothSpeed = 5.0f; // 위치 추적 속도
    public float rotationSmoothSpeed = 5.0f; // 회전 추적 속도

    void Start()
    {
        if (sim == null && target != null)
        {
            sim = target.GetComponent<TelloSimulator>();
        }
    }

    void Update()
    {
        if (Input.GetKeyDown(toggleKey))
        {
            mode = mode == CamMode.ThirdPerson ? CamMode.FirstPerson : CamMode.ThirdPerson;
        }
    }

    void LateUpdate()
    {
        if (target == null) return;

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
            // Yaw-only follow rotation: the drone visual may inherit tilt from
            // the URDF children, and the camera should never roll with it.
            Quaternion yawRot = Quaternion.Euler(0f, target.eulerAngles.y, 0f);

            if (mode == CamMode.FirstPerson)
            {
                desiredPosition = target.position + yawRot * fpvOffset;
                desiredRotation = yawRot;
            }
            else
            {
                desiredPosition = target.position + yawRot * offset;
                desiredRotation = Quaternion.LookRotation(target.position - desiredPosition);
            }
        }

        transform.position = Vector3.Lerp(transform.position, desiredPosition, Time.deltaTime * smoothSpeed);
        transform.rotation = Quaternion.Slerp(transform.rotation, desiredRotation, Time.deltaTime * rotationSmoothSpeed);
    }
}
