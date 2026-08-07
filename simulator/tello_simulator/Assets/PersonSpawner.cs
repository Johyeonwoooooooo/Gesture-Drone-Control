using System.Collections.Generic;
using UnityEngine;

// Randomly places people into the scanned house every time the simulation runs
// (Play). A RuntimeInitializeOnLoadMethod bootstraps one automatically, so no scene
// setup is needed — each Play gets a fresh, different layout. People are placed on
// valid floor spots around the drone's home area, sized to the scene automatically,
// and given trigger colliders so they never trip the drone's collision counter.
//
// The drone's own position is reset on Play by TelloSimulator.spawnAtHome (home
// teleport in its Start) — this component only handles the people half.
//
// To tune values in the Inspector, drop a PersonSpawner onto any GameObject in the
// scene; the bootstrap then defers to it instead of creating its own.
public class PersonSpawner : MonoBehaviour
{
    [Header("Count / Randomness")]
    public bool enableSpawning = true;
    public int personCount = 5;
    [Tooltip("0 = 매 실행마다 다른 무작위 배치. 다른 값이면 재현 가능한 시드.")]
    public int randomSeed = 0;

    [Header("Placement Area")]
    [Tooltip("배치 중심(월드). (0,0,0)이면 드론 홈(TelloSimulator.spawnPosition)을 자동 사용.")]
    public Vector3 areaCenter = Vector3.zero;
    [Tooltip("중심에서 이 반경(월드 단위) 안의 바닥에 배치.")]
    public float areaRadius = 18f;
    [Tooltip("바닥 탐색을 시작하는, 중심 Y 기준 위쪽 오프셋.")]
    public float searchAboveCenter = 3f;
    [Tooltip("그 지점에서 아래로 이만큼까지 바닥을 찾는다.")]
    public float maxFloorDrop = 12f;
    [Tooltip("사람들 사이 최소 간격(월드 단위).")]
    public float minSeparation = 3f;
    public int maxTriesPerPerson = 40;
    [Tooltip("바닥/장애물로 취급할 레이어.")]
    public LayerMask environmentMask = ~0;

    [Header("Person Size (scene 스케일에 자동 대응)")]
    [Tooltip("드론 크기를 재서 사람 키를 자동 산출한다.")]
    public bool autoScaleFromDrone = true;
    [Tooltip("사람 키 = 드론 높이 × 이 비율 (autoScale on).")]
    public float humanToDroneRatio = 8.5f;
    [Tooltip("autoScale off이거나 측정 실패 시 쓰는 고정 키(월드 단위).")]
    public float fallbackPersonHeight = 8f;

    [Header("Appearance")]
    public Color[] palette =
    {
        new Color(0.86f, 0.82f, 0.78f),   // pale
        new Color(0.55f, 0.60f, 0.70f),   // slate
        new Color(0.70f, 0.45f, 0.45f),   // dull red
        new Color(0.45f, 0.55f, 0.50f),   // muted green
        new Color(0.60f, 0.58f, 0.66f),   // grey-violet
    };

    private readonly List<GameObject> spawned = new List<GameObject>();
    private TelloSimulator sim;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void Bootstrap()
    {
        // A scene-placed spawner wins (so its Inspector values are used).
        if (FindFirstObjectByType<PersonSpawner>() != null)
        {
            return;
        }
        if (FindFirstObjectByType<TelloSimulator>() == null)
        {
            return; // not the sim scene
        }
        GameObject go = new GameObject("PersonSpawner");
        go.AddComponent<PersonSpawner>();
    }

    void Start()
    {
        sim = FindFirstObjectByType<TelloSimulator>();
        Respawn();
    }

    [ContextMenu("Respawn People")]
    public void Respawn()
    {
        Clear();
        if (!enableSpawning || personCount <= 0)
        {
            return;
        }

        if (randomSeed != 0)
        {
            Random.InitState(randomSeed);
        }

        // Center on the drone's home. Read spawnPosition (not the live transform) so we
        // do not depend on whether TelloSimulator.Start ran before this one.
        Vector3 center = areaCenter;
        if (center == Vector3.zero && sim != null)
        {
            center = (sim.spawnAtHome) ? sim.spawnPosition : sim.transform.position;
        }

        float height = ResolvePersonHeight();
        float radius = Mathf.Max(0.1f, height * 0.16f);

        int placed = 0;
        for (int i = 0; i < personCount; i++)
        {
            if (TryFindSpot(center, height, radius, out Vector3 feet))
            {
                Color color = (palette != null && palette.Length > 0)
                    ? palette[placed % palette.Length]
                    : new Color(0.75f, 0.75f, 0.8f);
                BuildPerson($"person_{placed + 1:000}", feet, height, radius, color);
                placed++;
            }
        }

        Debug.Log($"<color=cyan>[People] Placed {placed}/{personCount} person(s) " +
                  $"(height {height:F1}) around {center}.</color>");
        if (placed < personCount)
        {
            Debug.LogWarning($"[People] Only {placed} spot(s) found — try a larger areaRadius " +
                             "or smaller minSeparation.");
        }
    }

    // Person height auto-scaled to the scene from the drone's measured size, so it works
    // whether the house glb is imported at scale 5, 30, or anything else.
    float ResolvePersonHeight()
    {
        if (autoScaleFromDrone && sim != null)
        {
            bool found = false;
            Bounds b = new Bounds();
            foreach (Renderer r in sim.GetComponentsInChildren<Renderer>())
            {
                if (r is TrailRenderer || r is LineRenderer)
                {
                    continue;
                }
                if (!found) { b = r.bounds; found = true; }
                else { b.Encapsulate(r.bounds); }
            }
            if (found && b.size.y > 0.0001f)
            {
                return Mathf.Max(0.5f, b.size.y * humanToDroneRatio);
            }
        }
        return Mathf.Max(0.5f, fallbackPersonHeight);
    }

    // Sample the area for a floor spot the person fits on without being buried in geometry
    // or overlapping another placed person.
    bool TryFindSpot(Vector3 center, float height, float radius, out Vector3 feet)
    {
        for (int t = 0; t < maxTriesPerPerson; t++)
        {
            // Uniform sample over the disk.
            float ang = Random.Range(0f, Mathf.PI * 2f);
            float r = Mathf.Sqrt(Random.value) * areaRadius;
            Vector3 xz = center + new Vector3(Mathf.Cos(ang), 0f, Mathf.Sin(ang)) * r;
            Vector3 origin = new Vector3(xz.x, center.y + searchAboveCenter, xz.z);

            if (!Physics.Raycast(origin, Vector3.down, out RaycastHit hit,
                                 searchAboveCenter + maxFloorDrop, environmentMask,
                                 QueryTriggerInteraction.Ignore))
            {
                continue;
            }
            if (hit.normal.y < 0.6f)
            {
                continue;   // wall / steep surface, not a floor
            }
            // Never land on the drone itself (its home sits at the sample center).
            if (sim != null && (hit.transform == sim.transform
                                || hit.transform.IsChildOf(sim.transform)))
            {
                continue;
            }

            Vector3 candidate = hit.point;
            // Body must not be embedded in scenery.
            Vector3 p1 = candidate + Vector3.up * (radius + 0.05f);
            Vector3 p2 = candidate + Vector3.up * Mathf.Max(radius + 0.06f, height - radius);
            if (Physics.CheckCapsule(p1, p2, radius, environmentMask,
                                     QueryTriggerInteraction.Ignore))
            {
                continue;
            }
            // Keep people apart.
            bool tooClose = false;
            foreach (GameObject g in spawned)
            {
                if (g == null) continue;
                Vector3 d = g.transform.position - candidate;
                d.y = 0f;
                if (d.magnitude < minSeparation) { tooClose = true; break; }
            }
            if (tooClose)
            {
                continue;
            }

            feet = candidate;
            return true;
        }

        feet = Vector3.zero;
        return false;
    }

    // A simple standing figure: capsule body + sphere head, feet on the floor point.
    void BuildPerson(string name, Vector3 feet, float height, float radius, Color color)
    {
        GameObject person = new GameObject(name);
        person.transform.position = feet;
        person.transform.rotation = Quaternion.Euler(0f, Random.Range(0f, 360f), 0f);

        Material mat = MakeMaterial(color);

        float bodyHeight = height * 0.82f;
        GameObject body = GameObject.CreatePrimitive(PrimitiveType.Capsule);
        body.name = "Body";
        body.transform.SetParent(person.transform, false);
        body.transform.localPosition = new Vector3(0f, bodyHeight * 0.5f, 0f);
        // Unity capsule primitive is 2 units tall, 1 unit diameter.
        body.transform.localScale = new Vector3(radius * 2f, bodyHeight * 0.5f, radius * 2f);
        body.GetComponent<Renderer>().material = mat;
        MakeTrigger(body.GetComponent<Collider>());

        float headD = height * 0.16f;
        GameObject head = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        head.name = "Head";
        head.transform.SetParent(person.transform, false);
        head.transform.localPosition = new Vector3(0f, height - headD * 0.5f, 0f);
        head.transform.localScale = Vector3.one * headD;
        head.GetComponent<Renderer>().material = MakeMaterial(
            Color.Lerp(color, new Color(0.9f, 0.78f, 0.7f), 0.6f));
        MakeTrigger(head.GetComponent<Collider>());

        spawned.Add(person);
    }

    static void MakeTrigger(Collider c)
    {
        // Trigger so the drone (its probe ignores triggers) never counts a person as a
        // collision, and the person never physically blocks flight.
        if (c != null) c.isTrigger = true;
    }

    static Material MakeMaterial(Color color)
    {
        Shader shader = Shader.Find("Universal Render Pipeline/Lit");
        if (shader == null) shader = Shader.Find("Standard");
        if (shader == null) shader = Shader.Find("Sprites/Default");
        Material mat = new Material(shader) { color = color };
        return mat;
    }

    void Clear()
    {
        foreach (GameObject g in spawned)
        {
            if (g != null) Destroy(g);
        }
        spawned.Clear();
    }
}
