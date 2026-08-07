using UnityEngine;

// A custom object placed in the environment that the drone can pick up and carry.
// Created at runtime by SimPropManager (default layout, cargo_layout.json, or the
// "spawn" UDP command). Expects a Rigidbody and a Collider on the same GameObject.
public class CarryableProp : MonoBehaviour
{
    public string propName = "prop";
    public string shape = "box";

    public bool isCarried { get; private set; }
    public bool isDelivered { get; private set; }

    [HideInInspector] public TextMesh label;

    private Rigidbody rb;
    private Collider col;
    private Renderer rend;
    private Color baseColor = Color.white;
    private Vector3 spawnPosition;

    void Awake()
    {
        rb = GetComponent<Rigidbody>();
        col = GetComponent<Collider>();
        rend = GetComponent<Renderer>();
        if (rend != null)
        {
            baseColor = rend.material.color;
        }
        spawnPosition = transform.position;
    }

    // Called once by the spawner after it has assigned the final material color.
    public void CaptureBaseState()
    {
        if (rend != null)
        {
            baseColor = rend.material.color;
        }
        spawnPosition = transform.position;
    }

    public void Attach(Transform holder, Vector3 worldPosition)
    {
        isCarried = true;
        // Picking a delivered prop back up un-delivers it; the state stream always
        // reflects where the cargo currently is, not mission history.
        isDelivered = false;
        rb.isKinematic = true;
        // The drone is a CharacterController; its own cargo must never block it.
        col.enabled = false;
        // Clamp rigidly onto the drone: parented + kinematic means the prop shares the
        // drone's transform exactly (no swinging, no lag). Kept upright and aligned to
        // the drone's heading so it reads as clamped to the body, not slung underneath.
        transform.SetParent(holder, true);
        transform.position = worldPosition;
        transform.rotation = Quaternion.Euler(0f, holder.eulerAngles.y, 0f);
        if (rend != null)
        {
            rend.material.color = baseColor;
        }
        if (label != null)
        {
            label.text = propName;
        }
    }

    public void Release()
    {
        isCarried = false;
        Transform holder = transform.parent;
        transform.SetParent(null, true);
        col.enabled = true;

        // Set the cargo DOWN cleanly on whatever surface is directly below, instead of
        // letting it free-fall and bounce. This is the "내려놓기" (place-down) behaviour:
        // the prop is snapped to rest on the ground/pad, upright, with zero velocity.
        PlaceOnSurfaceBelow(holder);

        // Write the resolved pose into the body BEFORE re-enabling dynamics so the prop
        // settles from where it is placed, not from where it was grabbed.
        rb.position = transform.position;
        rb.rotation = transform.rotation;
        rb.isKinematic = false;
        rb.linearVelocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
        if (holder != null)
        {
            StartCoroutine(TemporarilyIgnoreHolder(holder));
        }
    }

    // Snap the prop so it rests on the highest surface directly beneath it, upright.
    // The drone's own colliders and the prop's own collider are ignored. If nothing is
    // found within reach the prop keeps its current pose and simply falls (old behaviour).
    void PlaceOnSurfaceBelow(Transform holder)
    {
        float halfHeight = (rend != null) ? rend.bounds.extents.y : 0.2f;

        // Keep the prop upright, preserving only its heading.
        transform.rotation = Quaternion.Euler(0f, transform.eulerAngles.y, 0f);

        Vector3 origin = transform.position + Vector3.up * 0.05f;
        RaycastHit[] hits = Physics.RaycastAll(origin, Vector3.down, 12f, ~0,
                                               QueryTriggerInteraction.Ignore);
        float surfaceY = float.NegativeInfinity;
        bool found = false;
        foreach (RaycastHit h in hits)
        {
            if (h.collider == col)
            {
                continue;   // the prop's own (re-enabled) collider
            }
            if (holder != null && (h.collider.transform == holder
                                   || h.collider.transform.IsChildOf(holder)))
            {
                continue;   // the drone
            }
            if (h.point.y > surfaceY)
            {
                surfaceY = h.point.y;   // topmost surface below = ground / pad
                found = true;
            }
        }

        if (found)
        {
            Vector3 p = transform.position;
            p.y = surfaceY + halfHeight + 0.01f;
            transform.position = p;
        }
    }

    // At the moment of release the prop can overlap one of the drone's colliders
    // (the URDF import leaves collision meshes on the model). If the collider is
    // enabled while overlapping, the physics engine's depenetration hurls the prop
    // several meters. Ignore collisions with the drone until the prop has fallen clear.
    System.Collections.IEnumerator TemporarilyIgnoreHolder(Transform holder)
    {
        Collider[] holderColliders = holder.GetComponentsInChildren<Collider>(false);
        foreach (Collider hc in holderColliders)
        {
            if (hc != null && hc.enabled && hc.gameObject.activeInHierarchy && hc != col)
            {
                Physics.IgnoreCollision(col, hc, true);
            }
        }
        yield return new WaitForSeconds(1.5f);
        foreach (Collider hc in holderColliders)
        {
            if (hc != null && hc.enabled && hc.gameObject.activeInHierarchy
                && col != null && col.enabled && hc != col)
            {
                Physics.IgnoreCollision(col, hc, false);
            }
        }
    }

    public void MarkDelivered(Color zoneTint)
    {
        if (isDelivered)
        {
            return;
        }
        isDelivered = true;
        if (rend != null)
        {
            rend.material.color = Color.Lerp(baseColor, zoneTint, 0.65f);
        }
        if (label != null)
        {
            label.text = propName + " [OK]";
        }
        Debug.Log($"[Cargo] '{propName}' delivered.");
    }

    void FixedUpdate()
    {
        // Safety net: a prop that tunnels through the environment and falls into the
        // void respawns at its original position instead of falling forever.
        if (!isCarried && transform.position.y < -20f)
        {
            rb.linearVelocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
            transform.position = spawnPosition;
        }
    }
}
