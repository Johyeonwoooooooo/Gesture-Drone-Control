using UnityEngine;

public class CameraFollow : MonoBehaviour
{
    [Header("Target to Follow")]
    public Transform target; // 드론 오브젝트를 Inspector에서 여기에 드래그 앤 드롭 하세요.

    [Header("Offset Settings")]
    public Vector3 offset = new Vector3(0, 1.5f, -4.0f); // 드론 기준 상대적 위치 (위로 1.5, 뒤로 4)
    
    [Header("Follow Physics")]
    public float smoothSpeed = 5.0f; // 위치 추적 속도
    public float rotationSmoothSpeed = 5.0f; // 회전 추적 속도

    void LateUpdate()
    {
        if (target == null) return;

        // 1. 목표 위치 계산 (드론의 회전 상태를 반영)
        Vector3 desiredPosition = target.TransformPoint(offset);
        
        // 2. 부드러운 위치 이동 (Lerp)
        transform.position = Vector3.Lerp(transform.position, desiredPosition, Time.deltaTime * smoothSpeed);

        // 3. 목표 회전 계산 (드론을 부드럽게 바라보기)
        Quaternion targetRotation = Quaternion.LookRotation(target.position - transform.position);
        transform.rotation = Quaternion.Slerp(transform.rotation, targetRotation, Time.deltaTime * rotationSmoothSpeed);
    }
}
