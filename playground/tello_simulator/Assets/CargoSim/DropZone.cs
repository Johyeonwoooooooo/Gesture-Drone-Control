using UnityEngine;

// Delivery target pad. A carryable prop that comes to rest inside the trigger
// volume is marked as delivered. Visuals (pad + label) are built by SimPropManager.
public class DropZone : MonoBehaviour
{
    public string zoneName = "dropzone";
    public float radius = 1.5f;
    public Color zoneColor = new Color(0.15f, 1f, 0.45f, 0.45f);

    [HideInInspector] public Renderer padRenderer;
    [HideInInspector] public TextMesh label;

    void Update()
    {
        // Soft pulse so the pad reads as an active target.
        if (padRenderer != null)
        {
            Color c = zoneColor;
            c.a = 0.35f + 0.15f * Mathf.Sin(Time.time * 3f);
            padRenderer.material.color = c;
        }
    }

    void OnTriggerStay(Collider other)
    {
        Rigidbody rb = other.attachedRigidbody;
        if (rb == null)
        {
            return;
        }

        CarryableProp prop = rb.GetComponent<CarryableProp>();
        if (prop == null || prop.isCarried || prop.isDelivered)
        {
            return;
        }

        // Only count the delivery once the prop has settled on the pad.
        if (rb.linearVelocity.sqrMagnitude > 0.05f)
        {
            return;
        }

        prop.MarkDelivered(zoneColor);
    }
}
