using UnityEngine;

// Dynamic-obstacle avoidance assist for the drone. Every frame while flying it looks for
// moving ghosts (GhostRoamer) near/ahead of the drone and nudges the CharacterController
// sideways-and-up to slip around them — layered ON TOP of the pilot's / mission's rc
// motion, so it works in both manual flight and the autonomous delivery mission.
//
// Ghosts carry no collider (so they never trip the drone's collision probe), which is why
// avoidance is proactive here rather than relying on physics blocking. Added to the drone
// at runtime by SimPropManager. Toggle with the "avoid on" / "avoid off" UDP command.
[RequireComponent(typeof(CharacterController))]
public class DroneAvoidance : MonoBehaviour
{
    [Tooltip("Master switch. Toggled at runtime by the 'avoid on/off' UDP command.")]
    public bool avoidanceEnabled = true;
    [Tooltip("Distance (5x-scaled world) at which a ghost starts pushing the drone away.")]
    public float safeDistance = 4.0f;
    [Tooltip("Peak dodge speed (m/s) applied when a ghost is right on top of the drone.")]
    public float avoidStrength = 6.0f;
    [Tooltip("How much extra the drone lifts while dodging (fraction of the dodge vector).")]
    public float upwardBias = 0.3f;
    [Tooltip("Extra weight given to a ghost that sits ahead, in the drone's flight path.")]
    public float frontBias = 1.0f;
    [Tooltip("Only dodge while the drone is airborne.")]
    public bool onlyWhenFlying = true;

    private CharacterController cc;
    private TelloSimulator sim;
    private bool avoidingNow;
    private int threatCount;

    void Awake()
    {
        cc = GetComponent<CharacterController>();
        sim = GetComponent<TelloSimulator>();
    }

    // LateUpdate runs after TelloSimulator's rc move (Update), so the dodge is applied on
    // top of the commanded motion for the frame instead of being overwritten by it.
    void LateUpdate()
    {
        avoidingNow = false;
        threatCount = 0;

        if (!avoidanceEnabled || cc == null || !cc.enabled)
        {
            return;
        }
        if (onlyWhenFlying && sim != null && !sim.IsFlying)
        {
            return;
        }

        Vector3 push = Vector3.zero;
        foreach (GhostRoamer ghost in Object.FindObjectsByType<GhostRoamer>(FindObjectsSortMode.None))
        {
            if (ghost == null)
            {
                continue;
            }

            Vector3 toDrone = transform.position - ghost.transform.position;
            float dist = toDrone.magnitude;
            if (dist > safeDistance || dist < 1e-3f)
            {
                continue;
            }

            threatCount++;
            avoidingNow = true;

            // Closer ghost -> stronger push (1 at contact, 0 at the edge of safeDistance).
            float strength = (safeDistance - dist) / safeDistance;

            // Horizontal "back away" component, always applied so a ghost drifting in from
            // any side still shoves the drone clear.
            Vector3 awayH = Vector3.ProjectOnPlane(toDrone, Vector3.up);
            awayH = awayH.sqrMagnitude > 1e-4f ? awayH.normalized : -transform.forward;

            // Lateral side-step: pick the side that moves the drone off the ghost's line.
            // When the ghost is dead ahead this makes the drone slide around it, not just brake.
            Vector3 rel = ghost.transform.position - transform.position;
            float side = Vector3.Dot(rel, transform.right);          // >0: ghost on the right
            Vector3 dodgeDir = (side > 0f ? -transform.right : transform.right);
            float front = Mathf.Clamp01(Vector3.Dot(transform.forward, rel.normalized)); // ghost ahead?

            Vector3 v = awayH * (0.6f + 0.4f * front)
                      + dodgeDir * front
                      + Vector3.up * upwardBias;

            push += v * strength * (1f + front * frontBias);
        }

        if (push.sqrMagnitude > 1e-4f)
        {
            Vector3 move = Vector3.ClampMagnitude(push, 1f) * avoidStrength * Time.deltaTime;
            cc.Move(move);
        }
    }

    public bool IsAvoiding => avoidingNow;

    void OnDrawGizmosSelected()
    {
        Gizmos.color = new Color(1f, 0.5f, 0.1f, 0.5f);
        Gizmos.DrawWireSphere(transform.position, safeDistance);
    }

    void OnGUI()
    {
        GUIStyle style = new GUIStyle(GUI.skin.label) { fontSize = 14 };
        if (avoidingNow)
        {
            GUI.color = new Color(1f, 0.5f, 0.1f);
            GUI.Label(new Rect(10, 210, 620, 25),
                $"[Avoid] EVADING {threatCount} ghost(s) ahead", style);
        }
        else
        {
            GUI.color = Color.white;
            GUI.Label(new Rect(10, 210, 620, 25),
                $"[Avoid] {(avoidanceEnabled ? "armed" : "OFF")}  (UDP: avoid on / avoid off)", style);
        }
        GUI.color = Color.white;
    }
}
