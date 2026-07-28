using UnityEngine;

// Gives the simulated drone a magnetic-winch style cargo hook: "grab" attaches the
// nearest carryable prop below the drone, "drop" releases it. Added to the drone at
// runtime by SimPropManager; commands arrive through TelloSimulator's UDP hook.
public class DroneCargo : MonoBehaviour
{
    [Tooltip("Maximum distance from the drone to a prop for a grab to succeed.")]
    public float grabRadius = 2.5f;
    [Tooltip("How far below the drone the carried prop hangs.")]
    public float carryDistance = 1.1f;

    public CarryableProp carried { get; private set; }

    private LineRenderer tether;

    void Awake()
    {
        GameObject tetherGO = new GameObject("CargoTether");
        tetherGO.transform.SetParent(transform, false);
        tether = tetherGO.AddComponent<LineRenderer>();
        tether.positionCount = 2;
        tether.startWidth = 0.06f;
        tether.endWidth = 0.06f;
        tether.material = new Material(Shader.Find("Sprites/Default"));
        tether.startColor = new Color(1f, 0.8f, 0.2f, 0.9f);
        tether.endColor = new Color(1f, 0.8f, 0.2f, 0.9f);
        tether.useWorldSpace = true;
        tether.enabled = false;
    }

    public bool TryGrab()
    {
        if (carried != null)
        {
            return false;
        }

        CarryableProp best = null;
        float bestDist = float.MaxValue;
        foreach (CarryableProp prop in Object.FindObjectsByType<CarryableProp>(FindObjectsSortMode.None))
        {
            if (prop.isCarried)
            {
                continue;
            }
            float dist = Vector3.Distance(transform.position, prop.transform.position);
            if (dist <= grabRadius && dist < bestDist)
            {
                best = prop;
                bestDist = dist;
            }
        }

        if (best == null)
        {
            Debug.Log($"[Cargo] Grab failed: no carryable prop within {grabRadius:F1}m.");
            return false;
        }

        best.Attach(transform, transform.position + Vector3.down * carryDistance);
        carried = best;
        tether.enabled = true;
        Debug.Log($"[Cargo] Grabbed '{best.propName}' ({bestDist:F2}m).");
        return true;
    }

    public bool Drop()
    {
        if (carried == null)
        {
            Debug.Log("[Cargo] Drop ignored: not carrying anything.");
            return false;
        }

        Debug.Log($"[Cargo] Dropped '{carried.propName}'.");
        carried.Release();
        carried = null;
        tether.enabled = false;
        return true;
    }

    void LateUpdate()
    {
        if (carried != null)
        {
            tether.SetPosition(0, transform.position);
            tether.SetPosition(1, carried.transform.position);
        }
    }

    void OnGUI()
    {
        GUIStyle style = new GUIStyle(GUI.skin.label);
        style.fontSize = 14;
        GUI.color = carried != null ? new Color(1f, 0.75f, 0.2f) : Color.white;
        string text = carried != null ? $"[Cargo] Carrying: {carried.propName}" : "[Cargo] Carrying: -";
        GUI.Label(new Rect(10, 160, 520, 25), text, style);
        GUI.color = Color.white;
    }
}
