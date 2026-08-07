using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

// Places custom objects (packages, barrels, drop zones) into the scene at Play time
// and wires the drone with cargo interaction (grab / carry / drop). No scene changes
// are required: a RuntimeInitializeOnLoadMethod bootstraps everything next to the
// existing TelloSimulator.
//
// Extra UDP commands (same command port as TelloSimulator, default 9000):
//   grab                                     pick up the nearest carryable prop
//   drop                                     release the carried prop
//   objects                                  reply on the state port with all props/zones (JSON)
//   spawn <shape> <x> <y> <z> [scale] [rel]  place a new prop (box | sphere | barrel);
//                                            "rel" interprets x/y/z in the drone's local frame
//   clearobjects                             remove every spawned prop
//   ghost [count]                            import free-roaming ghost(s) that wander the scene
//   clearghosts                              remove every spawned ghost
//   avoid on | avoid off                     toggle the drone's dynamic ghost-avoidance assist
//
// The layout comes from Assets/StreamingAssets/cargo_layout.json when present,
// otherwise a built-in demo layout is used. Layout coordinates are relative to the
// drone spawn pose by default (+z = the direction the drone faces at start).
public class SimPropManager : MonoBehaviour
{
    [Serializable]
    public class PropDef
    {
        public string name;
        public string shape = "box";
        public float x, y, z;
        public float scale = 0.4f;
        public string color = "";
        public float mass = 0.5f;
    }

    [Serializable]
    public class ZoneDef
    {
        public string name = "dropzone";
        public float x, y, z;
        public float radius = 1.5f;
        public string color = "#26FF73";
    }

    [Serializable]
    public class Layout
    {
        public bool relativeToDrone = true;
        public PropDef[] props;
        public ZoneDef[] zones;
    }

    public const string LayoutFileName = "cargo_layout.json";

    private TelloSimulator simulator;
    private DroneCargo cargo;
    private DroneAvoidance avoidance;
    private Vector3 droneStartPos;
    private Quaternion droneStartRot;
    private int spawnCounter;

    private readonly List<CarryableProp> props = new List<CarryableProp>();
    private readonly List<DropZone> zones = new List<DropZone>();
    private readonly List<GhostRoamer> ghosts = new List<GhostRoamer>();
    private readonly List<Transform> billboards = new List<Transform>();
    private int ghostCounter;

    private static readonly Color[] SpawnPalette =
    {
        new Color(1f, 0.55f, 0.15f),   // orange
        new Color(0.25f, 0.55f, 1f),   // blue
        new Color(0.95f, 0.3f, 0.35f), // red
        new Color(0.6f, 0.4f, 0.9f),   // purple
        new Color(0.95f, 0.85f, 0.25f) // yellow
    };

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void Bootstrap()
    {
        TelloSimulator sim = FindFirstObjectByType<TelloSimulator>();
        if (sim == null)
        {
            return; // scene without the drone simulator — nothing to do
        }
        GameObject go = new GameObject("SimPropManager");
        SimPropManager manager = go.AddComponent<SimPropManager>();
        manager.simulator = sim;
    }

    void Start()
    {
        droneStartPos = simulator.transform.position;
        droneStartRot = Quaternion.Euler(0f, simulator.transform.eulerAngles.y, 0f);

        cargo = simulator.GetComponent<DroneCargo>();
        if (cargo == null)
        {
            cargo = simulator.gameObject.AddComponent<DroneCargo>();
        }

        avoidance = simulator.GetComponent<DroneAvoidance>();
        if (avoidance == null)
        {
            avoidance = simulator.gameObject.AddComponent<DroneAvoidance>();
        }

        simulator.commandHook = HandleCommand;
        simulator.stateExtraProvider = StateExtra;

        Layout layout = LoadLayout();
        BuildLayout(layout);

        Debug.Log($"<color=cyan>[Cargo] Simulation ready: {props.Count} prop(s), {zones.Count} zone(s). " +
                  "Commands: grab / drop / objects / spawn / clearobjects</color>");
    }

    // ------------------------------------------------------------------ layout

    Layout LoadLayout()
    {
        string path = Path.Combine(Application.streamingAssetsPath, LayoutFileName);
        try
        {
            if (File.Exists(path))
            {
                Layout parsed = JsonUtility.FromJson<Layout>(File.ReadAllText(path));
                if (parsed != null && ((parsed.props != null && parsed.props.Length > 0)
                                       || (parsed.zones != null && parsed.zones.Length > 0)))
                {
                    Debug.Log($"[Cargo] Loaded layout from {path}");
                    return parsed;
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[Cargo] Failed to read {path}: {e.Message} — using built-in layout.");
        }
        return DefaultLayout();
    }

    static Layout DefaultLayout()
    {
        return new Layout
        {
            relativeToDrone = true,
            props = new[]
            {
                new PropDef { name = "package_orange", shape = "box",    x =  1.5f, y = 0.4f, z = 3.0f, scale = 0.35f, color = "#FF8C26", mass = 0.4f },
                new PropDef { name = "package_blue",   shape = "box",    x = -1.2f, y = 0.4f, z = 4.0f, scale = 0.45f, color = "#4090FF", mass = 0.6f },
                new PropDef { name = "barrel_purple",  shape = "barrel", x =  0.4f, y = 0.4f, z = 5.5f, scale = 0.35f, color = "#9966E6", mass = 0.5f },
            },
            zones = new[]
            {
                new ZoneDef { name = "dropzone", x = 0f, y = 0f, z = 8.5f, radius = 1.5f, color = "#26FF73" },
            },
        };
    }

    void BuildLayout(Layout layout)
    {
        if (layout.zones != null)
        {
            foreach (ZoneDef def in layout.zones)
            {
                SpawnZone(def, layout.relativeToDrone);
            }
        }
        if (layout.props != null)
        {
            foreach (PropDef def in layout.props)
            {
                SpawnProp(def, layout.relativeToDrone);
            }
        }
    }

    Vector3 ResolvePosition(float x, float y, float z, bool relative)
    {
        Vector3 pos = new Vector3(x, y, z);
        return relative ? droneStartPos + droneStartRot * pos : pos;
    }

    // ------------------------------------------------------------------ placement

    // The environment is a scanned mesh, so a requested point can easily sit inside
    // furniture or a wall. Search outward from the desired point (expanding rings)
    // for ground where the object is not embedded in geometry and, for props, the
    // drone has room to hover above and grab.
    delegate bool SpotValidator(Vector3 groundPoint, out Vector3 placed);

    bool SearchSpot(Vector3 desired, SpotValidator validator, out Vector3 result)
    {
        for (int ring = 0; ring < 14; ring++)
        {
            float r = ring * 0.6f;
            int steps = Mathf.Max(1, Mathf.CeilToInt(2f * Mathf.PI * r / 0.7f));
            for (int i = 0; i < steps; i++)
            {
                float angle = (i + ring * 0.5f) / steps * 2f * Mathf.PI;
                Vector3 probe = new Vector3(
                    desired.x + Mathf.Sin(angle) * r,
                    desired.y + 2f,
                    desired.z + Mathf.Cos(angle) * r);

                if (!Physics.Raycast(probe, Vector3.down, out RaycastHit hit, 30f,
                                     ~0, QueryTriggerInteraction.Ignore))
                {
                    continue;
                }
                // Only accept floor-like surfaces, not glancing hits on walls.
                if (hit.normal.y < 0.7f)
                {
                    continue;
                }
                if (validator(hit.point, out result))
                {
                    return true;
                }
            }
        }
        result = desired;
        return false;
    }

    bool ValidPropSpot(float half, Vector3 ground, out Vector3 center)
    {
        center = ground + Vector3.up * (half + 0.05f);
        // Not inside a delivery zone — a prop that spawns on the pad would count
        // as instantly delivered and get bumped around during later deliveries.
        foreach (DropZone zone in zones)
        {
            if (zone == null)
            {
                continue;
            }
            Vector3 d = center - zone.transform.position;
            d.y = 0f;
            if (d.magnitude < zone.radius + 0.6f)
            {
                return false;
            }
        }
        // Embedded in scenery (or overlapping another prop / the drone)?
        if (Physics.CheckBox(center, Vector3.one * (half + 0.02f), Quaternion.identity,
                             ~0, QueryTriggerInteraction.Ignore))
        {
            return false;
        }
        // Clear column above so the drone can descend from cruise altitude and grab.
        // The scan environment is scaled 5x, so furniture is 3-5m tall — the drone
        // navigates above it (~9m) and needs a free vertical corridor down to the prop.
        if (Physics.CheckCapsule(center + Vector3.up * (half + 0.45f),
                                 center + Vector3.up * 6f, 0.4f,
                                 ~0, QueryTriggerInteraction.Ignore))
        {
            return false;
        }
        return true;
    }

    bool ValidZoneSpot(float radius, Vector3 ground, out Vector3 center)
    {
        center = ground + Vector3.up * 0.02f;
        // The pad disc plus ~6m of air above it must be free so the drone can
        // descend from cruise altitude with cargo hanging below.
        Vector3 boxCenter = center + Vector3.up * 3.05f;
        Vector3 halfExtents = new Vector3(radius * 0.85f, 2.95f, radius * 0.85f);
        return !Physics.CheckBox(boxCenter, halfExtents, Quaternion.identity,
                                 ~0, QueryTriggerInteraction.Ignore);
    }

    // ------------------------------------------------------------------ spawning

    CarryableProp SpawnProp(PropDef def, bool relative)
    {
        PrimitiveType primitive;
        switch ((def.shape ?? "box").ToLowerInvariant())
        {
            case "sphere": primitive = PrimitiveType.Sphere; break;
            case "barrel":
            case "cylinder": primitive = PrimitiveType.Cylinder; break;
            default: primitive = PrimitiveType.Cube; break;
        }

        spawnCounter++;
        string name = string.IsNullOrEmpty(def.name) ? $"{def.shape}_{spawnCounter}" : def.name;
        float scale = Mathf.Clamp(def.scale <= 0f ? 0.4f : def.scale, 0.1f, 2f);

        Vector3 desired = ResolvePosition(def.x, def.y, def.z, relative);
        float half = scale * 0.5f;
        bool found = SearchSpot(desired,
            (Vector3 ground, out Vector3 placed) => ValidPropSpot(half, ground, out placed),
            out Vector3 pos);
        if (!found)
        {
            // Last resort: keep the requested point but at least lift it above the
            // surface directly below so it is not buried under the floor.
            pos = desired;
            if (Physics.Raycast(pos + Vector3.up * 2f, Vector3.down, out RaycastHit hit, 60f,
                                ~0, QueryTriggerInteraction.Ignore))
            {
                pos.y = Mathf.Max(pos.y, hit.point.y + scale);
            }
            Debug.LogWarning($"[Cargo] '{name}' 주변에서 빈 공간을 찾지 못해 요청 위치에 그대로 배치합니다: {pos}");
        }
        else
        {
            float movedXZ = Vector2.Distance(new Vector2(desired.x, desired.z), new Vector2(pos.x, pos.z));
            if (movedXZ > 0.5f)
            {
                Debug.Log($"[Cargo] '{name}' 요청 위치가 막혀 있어 {movedXZ:F1}m 옆 빈 공간으로 재배치: {pos}");
            }
        }

        GameObject go = GameObject.CreatePrimitive(primitive);
        go.name = name;
        go.transform.position = pos;
        go.transform.localScale = Vector3.one * scale;

        Color color = SpawnPalette[(spawnCounter - 1) % SpawnPalette.Length];
        if (!string.IsNullOrEmpty(def.color) && ColorUtility.TryParseHtmlString(def.color, out Color parsed))
        {
            color = parsed;
        }
        go.GetComponent<Renderer>().material = MakeLitMaterial(color);

        Rigidbody rb = go.AddComponent<Rigidbody>();
        rb.mass = Mathf.Max(0.05f, def.mass);
        // No interpolation: while carried the body is a shapeless kinematic whose
        // physics pose goes stale; on release the interpolation buffer would snap
        // the prop back toward its pickup point (observed as a multi-meter "fling").
        rb.interpolation = RigidbodyInterpolation.None;
        rb.collisionDetectionMode = CollisionDetectionMode.Continuous;
        // If the prop ever overlaps another collider, eject it gently instead of
        // hurling it across the room.
        rb.maxDepenetrationVelocity = 1f;

        CarryableProp prop = go.AddComponent<CarryableProp>();
        prop.propName = name;
        prop.shape = def.shape;
        prop.label = MakeLabel(go.transform, name, 1.4f, color, scale);
        prop.CaptureBaseState();

        props.Add(prop);
        return prop;
    }

    void SpawnZone(ZoneDef def, bool relative)
    {
        Vector3 desired = ResolvePosition(def.x, def.y, def.z, relative);
        float zoneRadius = Mathf.Clamp(def.radius <= 0f ? 1.5f : def.radius, 0.5f, 10f);
        bool found = SearchSpot(desired,
            (Vector3 ground, out Vector3 placed) => ValidZoneSpot(zoneRadius, ground, out placed),
            out Vector3 pos);
        if (!found)
        {
            pos = desired;
            if (Physics.Raycast(pos + Vector3.up * 2f, Vector3.down, out RaycastHit hit, 60f,
                                ~0, QueryTriggerInteraction.Ignore))
            {
                pos.y = hit.point.y + 0.02f;
            }
            Debug.LogWarning($"[Cargo] 배달구역 '{def.name}' 주변에서 빈 공간을 찾지 못해 요청 위치에 배치합니다: {pos}");
        }
        else
        {
            float movedXZ = Vector2.Distance(new Vector2(desired.x, desired.z), new Vector2(pos.x, pos.z));
            if (movedXZ > 0.5f)
            {
                Debug.Log($"[Cargo] 배달구역 '{def.name}' 재배치: {movedXZ:F1}m 옆 빈 공간 {pos}");
            }
        }

        Color color = new Color(0.15f, 1f, 0.45f);
        if (!string.IsNullOrEmpty(def.color) && ColorUtility.TryParseHtmlString(def.color, out Color parsed))
        {
            color = parsed;
        }

        float radius = zoneRadius;

        GameObject zoneGO = new GameObject(string.IsNullOrEmpty(def.name) ? "dropzone" : def.name);
        zoneGO.transform.position = pos;

        DropZone zone = zoneGO.AddComponent<DropZone>();
        zone.zoneName = zoneGO.name;
        zone.radius = radius;
        zone.zoneColor = color;

        // Trigger volume above the pad that catches dropped props. Triggers are
        // invisible to the drone's collision probe and to the CharacterController.
        BoxCollider trigger = zoneGO.AddComponent<BoxCollider>();
        trigger.isTrigger = true;
        trigger.center = new Vector3(0f, 1f, 0f);
        trigger.size = new Vector3(radius * 2f, 2.2f, radius * 2f);

        // Visual pad (no collider so it cannot bump the drone or the cargo).
        GameObject pad = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        pad.name = "Pad";
        pad.transform.SetParent(zoneGO.transform, false);
        pad.transform.localScale = new Vector3(radius * 2f, 0.03f, radius * 2f);
        pad.transform.localPosition = new Vector3(0f, 0.03f, 0f);
        Destroy(pad.GetComponent<Collider>());
        Renderer padRend = pad.GetComponent<Renderer>();
        padRend.material = new Material(Shader.Find("Sprites/Default"));
        padRend.material.color = new Color(color.r, color.g, color.b, 0.45f);
        zone.padRenderer = padRend;

        zone.label = MakeLabel(zoneGO.transform, zoneGO.name.ToUpperInvariant(), 1.6f, color, 1f);

        zones.Add(zone);
    }

    void ClearProps()
    {
        if (cargo != null)
        {
            cargo.Drop();
        }
        foreach (CarryableProp prop in props)
        {
            if (prop != null)
            {
                billboards.Remove(prop.label != null ? prop.label.transform : null);
                Destroy(prop.gameObject);
            }
        }
        props.Clear();
        Debug.Log("[Cargo] Cleared all props.");
    }

    // ------------------------------------------------------------------ ghost (자유 배회)

    // Imports a free-roaming "ghost" that wanders the scanned scene on its own. It floats,
    // has no collider (so it neither blocks nor is grabbed by the drone), and steers around
    // walls with raycasts. Spawned near the drone at a floor-relative height.
    GhostRoamer SpawnGhost()
    {
        ghostCounter++;

        // Start near the drone but offset to the side so multiple ghosts fan out.
        float angle = ghostCounter * 2.399963f;                 // golden-angle spread
        Vector3 offset = new Vector3(Mathf.Cos(angle), 0f, Mathf.Sin(angle)) * 3f;
        Vector3 spawn = simulator.transform.position + offset;
        if (Physics.Raycast(spawn + Vector3.up * 4f, Vector3.down, out RaycastHit hit, 80f,
                            ~0, QueryTriggerInteraction.Ignore))
        {
            spawn.y = hit.point.y + 2.5f;
        }
        else
        {
            spawn.y = simulator.transform.position.y + 2.5f;
        }

        GameObject go = new GameObject($"ghost_{ghostCounter}");
        go.transform.position = spawn;

        Color ghostColor = new Color(0.65f, 0.85f, 1f);         // pale cyan-white

        // Wispy body: a faint translucent outer shell around a brighter core. No colliders,
        // so the ghost cannot bump the drone or register as an obstacle for it.
        GameObject shell = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        shell.name = "Body";
        shell.transform.SetParent(go.transform, false);
        shell.transform.localScale = Vector3.one * 1.6f;
        Destroy(shell.GetComponent<Collider>());
        shell.GetComponent<Renderer>().material = MakeGhostMaterial(ghostColor, 0.22f);

        GameObject core = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        core.name = "Core";
        core.transform.SetParent(go.transform, false);
        core.transform.localScale = Vector3.one * 0.9f;
        core.transform.localPosition = Vector3.up * 0.15f;
        Destroy(core.GetComponent<Collider>());
        core.GetComponent<Renderer>().material = MakeGhostMaterial(ghostColor, 0.5f);

        MakeLabel(go.transform, "유령", 1.4f, ghostColor, 1f);

        GhostRoamer ghost = go.AddComponent<GhostRoamer>();
        ghosts.Add(ghost);
        Debug.Log($"[Cargo] Ghost '{go.name}' imported at {spawn} — now free-roaming.");
        return ghost;
    }

    void ClearGhosts()
    {
        foreach (GhostRoamer ghost in ghosts)
        {
            if (ghost != null)
            {
                Transform label = ghost.transform.Find("Label");
                if (label != null)
                {
                    billboards.Remove(label);
                }
                Destroy(ghost.gameObject);
            }
        }
        ghosts.Clear();
        Debug.Log("[Cargo] Cleared all ghosts.");
    }

    // Best-effort transparent material. Uses URP/Lit's transparent surface when available
    // (URP is the project's pipeline) and falls back to the unlit Sprites shader otherwise.
    static Material MakeGhostMaterial(Color color, float alpha)
    {
        color.a = alpha;
        Shader urp = Shader.Find("Universal Render Pipeline/Lit");
        if (urp != null)
        {
            Material mat = new Material(urp);
            mat.SetFloat("_Surface", 1f);                        // 0 = opaque, 1 = transparent
            mat.SetFloat("_Blend", 0f);                          // alpha blend
            mat.SetFloat("_SrcBlend", (float)UnityEngine.Rendering.BlendMode.SrcAlpha);
            mat.SetFloat("_DstBlend", (float)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            mat.SetFloat("_ZWrite", 0f);
            mat.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
            mat.DisableKeyword("_ALPHATEST_ON");
            mat.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
            mat.color = color;
            if (mat.HasProperty("_EmissionColor"))
            {
                mat.EnableKeyword("_EMISSION");
                mat.SetColor("_EmissionColor", new Color(color.r, color.g, color.b) * 0.6f);
            }
            return mat;
        }

        Material fallback = new Material(Shader.Find("Sprites/Default"));
        fallback.color = color;
        return fallback;
    }

    // ------------------------------------------------------------------ visuals

    static Material MakeLitMaterial(Color color)
    {
        Shader shader = Shader.Find("Universal Render Pipeline/Lit");
        if (shader == null)
        {
            shader = Shader.Find("Standard");
        }
        if (shader == null)
        {
            shader = Shader.Find("Sprites/Default");
        }
        Material mat = new Material(shader);
        mat.color = color;
        return mat;
    }

    TextMesh MakeLabel(Transform parent, string text, float worldHeight, Color color, float parentScale)
    {
        GameObject go = new GameObject("Label");
        go.transform.SetParent(parent, false);
        float inv = 1f / Mathf.Max(0.01f, parentScale);
        go.transform.localScale = Vector3.one * inv;
        go.transform.localPosition = new Vector3(0f, worldHeight * inv, 0f);

        TextMesh tm = go.AddComponent<TextMesh>();
        tm.text = text;
        tm.fontSize = 48;
        tm.characterSize = 0.045f;
        tm.anchor = TextAnchor.LowerCenter;
        tm.alignment = TextAlignment.Center;
        tm.color = color;

        Font font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        if (font != null)
        {
            tm.font = font;
            go.GetComponent<MeshRenderer>().material = font.material;
        }

        billboards.Add(go.transform);
        return tm;
    }

    void LateUpdate()
    {
        Camera cam = Camera.main;
        if (cam != null)
        {
            foreach (Transform t in billboards)
            {
                if (t != null)
                {
                    t.rotation = Quaternion.LookRotation(t.position - cam.transform.position);
                }
            }
        }

        foreach (DropZone zone in zones)
        {
            if (zone != null && zone.label != null)
            {
                zone.label.text = $"{zone.zoneName.ToUpperInvariant()}  {DeliveredCount()}/{AliveProps()}";
            }
        }
    }

    // ------------------------------------------------------------------ commands / state

    int DeliveredCount()
    {
        int delivered = 0;
        foreach (CarryableProp prop in props)
        {
            if (prop != null && prop.isDelivered)
            {
                delivered++;
            }
        }
        return delivered;
    }

    int AliveProps()
    {
        int alive = 0;
        foreach (CarryableProp prop in props)
        {
            if (prop != null)
            {
                alive++;
            }
        }
        return alive;
    }

    bool HandleCommand(string cmd)
    {
        if (cmd == "grab")
        {
            cargo.TryGrab();
            return true;
        }
        if (cmd == "drop")
        {
            cargo.Drop();
            return true;
        }
        if (cmd == "objects")
        {
            simulator.SendJson(BuildObjectsJson());
            return true;
        }
        if (cmd == "clearobjects")
        {
            ClearProps();
            return true;
        }
        if (cmd.StartsWith("spawn "))
        {
            HandleSpawnCommand(cmd);
            return true;
        }
        if (cmd == "ghost" || cmd.StartsWith("ghost "))
        {
            int count = 1;
            string[] gp = cmd.Split(' ');
            if (gp.Length > 1)
            {
                int.TryParse(gp[1], out count);
            }
            for (int i = 0; i < Mathf.Clamp(count, 1, 20); i++)
            {
                SpawnGhost();
            }
            return true;
        }
        if (cmd == "clearghosts")
        {
            ClearGhosts();
            return true;
        }
        if (cmd == "avoid on" || cmd == "avoid off")
        {
            if (avoidance != null)
            {
                avoidance.avoidanceEnabled = cmd.EndsWith("on");
                Debug.Log($"[Cargo] Dynamic avoidance {(avoidance.avoidanceEnabled ? "ON" : "OFF")}.");
            }
            return true;
        }
        return false;
    }

    void HandleSpawnCommand(string cmd)
    {
        // spawn <shape> <x> <y> <z> [scale] [rel]
        string[] parts = cmd.Split(' ');
        if (parts.Length < 5
            || !float.TryParse(parts[2], NumberStyles.Float, CultureInfo.InvariantCulture, out float x)
            || !float.TryParse(parts[3], NumberStyles.Float, CultureInfo.InvariantCulture, out float y)
            || !float.TryParse(parts[4], NumberStyles.Float, CultureInfo.InvariantCulture, out float z))
        {
            Debug.LogWarning($"[Cargo] Failed to parse spawn command: '{cmd}' " +
                             "(expected: spawn <shape> <x> <y> <z> [scale] [rel])");
            return;
        }

        float scale = 0.4f;
        bool relative = false;
        for (int i = 5; i < parts.Length; i++)
        {
            if (parts[i] == "rel")
            {
                relative = true;
            }
            else if (float.TryParse(parts[i], NumberStyles.Float, CultureInfo.InvariantCulture, out float s))
            {
                scale = s;
            }
        }

        Vector3 pos = relative
            ? simulator.transform.TransformPoint(new Vector3(x, y, z))
            : new Vector3(x, y, z);

        PropDef def = new PropDef
        {
            name = "",
            shape = parts[1],
            x = pos.x,
            y = pos.y,
            z = pos.z,
            scale = scale,
        };
        CarryableProp prop = SpawnProp(def, false);
        Debug.Log($"[Cargo] Spawned '{prop.propName}' at {prop.transform.position}.");
    }

    string BuildObjectsJson()
    {
        StringBuilder sb = new StringBuilder(512);
        sb.Append("{\"type\":\"objects\",\"props\":[");
        bool first = true;
        foreach (CarryableProp prop in props)
        {
            if (prop == null)
            {
                continue;
            }
            if (!first)
            {
                sb.Append(',');
            }
            first = false;
            Vector3 p = prop.transform.position;
            sb.Append("{\"name\":\"").Append(prop.propName)
              .Append("\",\"shape\":\"").Append(prop.shape)
              .Append("\",\"x\":").Append(p.x.ToString("F3", CultureInfo.InvariantCulture))
              .Append(",\"y\":").Append(p.y.ToString("F3", CultureInfo.InvariantCulture))
              .Append(",\"z\":").Append(p.z.ToString("F3", CultureInfo.InvariantCulture))
              .Append(",\"carried\":").Append(prop.isCarried ? "true" : "false")
              .Append(",\"delivered\":").Append(prop.isDelivered ? "true" : "false")
              .Append('}');
        }
        sb.Append("],\"zones\":[");
        first = true;
        foreach (DropZone zone in zones)
        {
            if (zone == null)
            {
                continue;
            }
            if (!first)
            {
                sb.Append(',');
            }
            first = false;
            Vector3 p = zone.transform.position;
            sb.Append("{\"name\":\"").Append(zone.zoneName)
              .Append("\",\"x\":").Append(p.x.ToString("F3", CultureInfo.InvariantCulture))
              .Append(",\"y\":").Append(p.y.ToString("F3", CultureInfo.InvariantCulture))
              .Append(",\"z\":").Append(p.z.ToString("F3", CultureInfo.InvariantCulture))
              .Append(",\"radius\":").Append(zone.radius.ToString("F2", CultureInfo.InvariantCulture))
              .Append('}');
        }
        sb.Append("]}");
        return sb.ToString();
    }

    string StateExtra()
    {
        string carrying = (cargo != null && cargo.carried != null)
            ? "\"" + cargo.carried.propName + "\""
            : "null";
        // Ghost positions ride along on the 20Hz state stream so the autonomous mission
        // can steer around these moving obstacles in its guidance loop.
        return "\"carrying\":" + carrying
             + ",\"delivered\":" + DeliveredCount().ToString(CultureInfo.InvariantCulture)
             + ",\"props_total\":" + AliveProps().ToString(CultureInfo.InvariantCulture)
             + ",\"ghosts\":" + GhostsJson();
    }

    string GhostsJson()
    {
        StringBuilder sb = new StringBuilder(64);
        sb.Append('[');
        bool first = true;
        foreach (GhostRoamer g in ghosts)
        {
            if (g == null)
            {
                continue;
            }
            if (!first)
            {
                sb.Append(',');
            }
            first = false;
            Vector3 p = g.transform.position;
            sb.Append("{\"x\":").Append(p.x.ToString("F3", CultureInfo.InvariantCulture))
              .Append(",\"y\":").Append(p.y.ToString("F3", CultureInfo.InvariantCulture))
              .Append(",\"z\":").Append(p.z.ToString("F3", CultureInfo.InvariantCulture))
              .Append('}');
        }
        sb.Append(']');
        return sb.ToString();
    }

    void OnGUI()
    {
        GUIStyle style = new GUIStyle(GUI.skin.label);
        style.fontSize = 14;
        GUI.Label(new Rect(10, 185, 820, 25),
            $"[Cargo] Delivered: {DeliveredCount()}/{AliveProps()}   Ghosts: {ghosts.Count}   " +
            "(UDP: grab / drop / objects / spawn / clearobjects / ghost / clearghosts / avoid)", style);
    }
}
