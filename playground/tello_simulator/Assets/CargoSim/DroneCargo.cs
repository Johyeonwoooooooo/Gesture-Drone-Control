using UnityEngine;

// Gives the simulated drone a rigid cargo clamp: "grab" snaps the nearest carryable
// prop directly onto the drone's underside (no dangling tether), "drop" releases it and
// sets it down cleanly on the surface below. Added to the drone at runtime by
// SimPropManager; commands arrive through TelloSimulator's UDP hook.
//
// Previous behaviour hung the prop 1.1 m below the drone and drew a LineRenderer
// "tether", so the cargo read as a sling load swinging on a string. It now clamps flush
// to the belly and moves as one rigid body with the drone.
public class DroneCargo : MonoBehaviour
{
    [Tooltip("Maximum distance from the drone to a prop for a grab to succeed.")]
    public float grabRadius = 2.5f;
    [Tooltip("Extra gap left between the drone's underside and the top of the carried " +
             "prop. 0 = the prop touches the belly. A small value avoids z-fighting.")]
    public float clampClearance = 0.02f;
    [Tooltip("Fallback centre-to-centre carry offset used only if the drone/prop bounds " +
             "cannot be measured (e.g. a prop with no renderer).")]
    public float carryDistance = 0.35f;

    public CarryableProp carried { get; private set; }

    // Signed offset from the drone root's position down to the lowest point of its own
    // model, measured once and reused so every grab clamps the prop to the same belly line.
    private float undersideOffset;
    private bool undersideMeasured;

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

        Vector3 mount = ComputeMountPosition(best);
        best.Attach(transform, mount);
        carried = best;
        Debug.Log($"[Cargo] Grabbed '{best.propName}' ({bestDist:F2}m) — clamped flush to the drone.");
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
        return true;
    }

    // World position at which the carried prop's centre should sit so its top face is
    // clamped against the drone's belly, directly under the drone's XZ centre.
    Vector3 ComputeMountPosition(CarryableProp prop)
    {
        float droneBottom = transform.position.y + MeasureUndersideOffset();

        float propHalfHeight = carryDistance;   // fallback if the prop has no renderer
        Renderer propRend = prop.GetComponentInChildren<Renderer>();
        if (propRend != null)
        {
            propHalfHeight = propRend.bounds.extents.y;
        }

        float centreY = droneBottom - clampClearance - propHalfHeight;
        return new Vector3(transform.position.x, centreY, transform.position.z);
    }

    // Distance from the drone root down to the bottom of its own visible model. Cached:
    // the model does not change shape, and the carried prop must never be counted here.
    float MeasureUndersideOffset()
    {
        if (undersideMeasured)
        {
            return undersideOffset;
        }

        bool found = false;
        float minY = float.MaxValue;
        foreach (Renderer r in GetComponentsInChildren<Renderer>())
        {
            // Skip billboards/labels and anything belonging to a carried prop.
            if (r.GetComponentInParent<CarryableProp>() != null)
            {
                continue;
            }
            if (r is TrailRenderer || r is LineRenderer)
            {
                continue;
            }
            minY = Mathf.Min(minY, r.bounds.min.y);
            found = true;
        }

        undersideOffset = found ? (minY - transform.position.y) : -carryDistance;
        undersideMeasured = true;
        return undersideOffset;
    }
}
