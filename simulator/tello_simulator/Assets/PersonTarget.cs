using UnityEngine;

public class PersonTarget : MonoBehaviour
{
    [SerializeField, HideInInspector] private string personId = "";
    [SerializeField, HideInInspector] private bool autoAddCollider = true;

    public string PersonId
    {
        get
        {
            EnsureId();
            return personId;
        }
    }

    private static bool assignedSceneIds;
    private static int fallbackNextId = 1;

    // World-space box around this person, for PatrolPersonDetection's on-screen
    // overlay. The detector's box is in drone-camera pixels and means nothing on
    // the third-person view, so the box the player sees is projected from here
    // instead. Recomputed on demand so it follows a person that moves.
    public bool TryGetWorldBounds(out Bounds bounds)
    {
        Renderer[] renderers = GetComponentsInChildren<Renderer>();
        if (renderers.Length == 0)
        {
            bounds = default;
            return false;
        }

        bounds = renderers[0].bounds;
        for (int i = 1; i < renderers.Length; i++)
        {
            bounds.Encapsulate(renderers[i].bounds);
        }
        return true;
    }

    private void Awake()
    {
        AssignMissingSceneIds();
        EnsureId();
        EnsureCollider();
    }

    private static void AssignMissingSceneIds()
    {
        if (assignedSceneIds)
        {
            return;
        }
        assignedSceneIds = true;
        fallbackNextId = 1;

        PersonTarget[] targets = FindObjectsByType<PersonTarget>(FindObjectsSortMode.None);
        System.Array.Sort(targets, (a, b) => string.CompareOrdinal(GetHierarchyPath(a.transform), GetHierarchyPath(b.transform)));

        foreach (PersonTarget target in targets)
        {
            if (target == null || !string.IsNullOrWhiteSpace(target.personId))
            {
                continue;
            }
            target.personId = $"person_{fallbackNextId:000}";
            fallbackNextId += 1;
        }
    }

    private void OnEnable()
    {
        EnsureId();
    }

    private void EnsureId()
    {
        if (!string.IsNullOrWhiteSpace(personId))
        {
            return;
        }
        personId = $"person_{fallbackNextId:000}";
        fallbackNextId += 1;
    }

    private void EnsureCollider()
    {
        if (!autoAddCollider || GetComponentInChildren<Collider>() != null)
        {
            return;
        }

        CapsuleCollider capsule = gameObject.AddComponent<CapsuleCollider>();
        capsule.isTrigger = true;

        Renderer[] renderers = GetComponentsInChildren<Renderer>();
        if (renderers.Length == 0)
        {
            capsule.center = new Vector3(0f, 0.9f, 0f);
            capsule.height = 1.8f;
            capsule.radius = 0.3f;
            return;
        }

        Bounds bounds = renderers[0].bounds;
        for (int i = 1; i < renderers.Length; i++)
        {
            bounds.Encapsulate(renderers[i].bounds);
        }

        Vector3 localCenter = transform.InverseTransformPoint(bounds.center);
        capsule.center = localCenter;
        capsule.height = Mathf.Max(0.3f, bounds.size.y / Mathf.Max(0.001f, transform.lossyScale.y));
        float xRadius = bounds.size.x / Mathf.Max(0.001f, transform.lossyScale.x);
        float zRadius = bounds.size.z / Mathf.Max(0.001f, transform.lossyScale.z);
        capsule.radius = Mathf.Max(0.1f, Mathf.Max(xRadius, zRadius) * 0.5f);
    }

    private static string GetHierarchyPath(Transform t)
    {
        string path = t.name;
        while (t.parent != null)
        {
            t = t.parent;
            path = t.name + "/" + path;
        }
        return path;
    }
}
