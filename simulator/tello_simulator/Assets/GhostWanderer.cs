using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Animations;
using UnityEngine.Playables;

// Pale translucent ghosts drifting through the house. Atmosphere only — every
// collider is stripped off them, so they never touch the drone's collision probe,
// never block flight, and never reach the patrol pipeline (no ReportDetection
// call anywhere in here).
//
// The figures are the CC0 .glb models in Assets/Resources/Ghosts (see the README
// there), imported by the same UnityGLTF package the house glb uses. Each is
// rescaled at spawn to `ghostMeters`, so model files of any unit convention drop
// in without editing. With that folder missing it falls back to primitives.
//
// A RuntimeInitializeOnLoadMethod bootstraps one on Play, so the scene file stays
// untouched (README §8 — only TelloSimulator and CameraFollow live in the scene).
// Drop a GhostWanderer on any GameObject to tune it in the Inspector; the
// bootstrap then defers to it.
//
// Walls, floors and ceilings are all ignored — the ghosts have no physics at all,
// they are moved with transform writes, and they climb between storeys straight
// through the slabs. The one thing they cannot leave is the building: a step is
// only taken if the destination still has building geometry somewhere below it
// (scanned over the whole height, not just the current storey), with the building
// bounding box as a second net.
//
// Verifying without waiting for one to wander past (see ShowHud / summonKey):
//   G    summon the next ghost right in front of the camera, cycling 1..N so
//        every one of them can be confirmed alive in a few seconds
//   HUD  bottom-left list: per-ghost distance + how often containment fired
public class GhostWanderer : MonoBehaviour
{
    [Header("Count / Randomness")]
    public bool enableGhosts = true;
    public int ghostCount = 10;
    [Tooltip("0 = 매 실행마다 다른 무작위 배치. 다른 값이면 재현 가능한 시드.")]
    public int randomSeed = 0;

    [Header("Placement Area")]
    [Tooltip("배치 중심(월드). (0,0,0)이면 드론 홈(TelloSimulator.spawnPosition)을 자동 사용.")]
    public Vector3 areaCenter = Vector3.zero;
    [Tooltip("중심에서 이 반경(월드 단위) 안의 바닥 위에 배치.")]
    public float areaRadius = 60f;
    [Tooltip("바닥 탐색을 시작하는, 기준점 위쪽 오프셋.")]
    public float searchAbove = 10f;
    [Tooltip("그 지점에서 아래로 이만큼까지 바닥을 찾는다. 층고보다 커야 아래층 바닥을 잡는다.")]
    public float maxFloorDrop = 20f;
    [Tooltip("유령들 사이 최소 간격(월드 단위).")]
    public float minSeparation = 15f;
    public int maxTriesPerGhost = 60;
    [Tooltip("바닥으로 취급할 레이어.")]
    public LayerMask environmentMask = ~0;

    [Header("Size")]
    // 드론을 재서 스케일을 뽑으면 안 된다: 씬의 드론 파츠는 0.05 유닛짜리
    // 납작한 원기둥이라 유령이 10 cm 먼지가 된다. glb 가 scale 5 로 들어와
    // 1 m = 5 유닛이므로 (README 좌표계 항목) 키는 그 환산으로 직접 준다.
    [Tooltip("씬 1 미터가 몇 유닛인가. 00809 glb 는 scale 5 로 배치 = 5.")]
    public float unitsPerMeter = 5f;
    [Tooltip("유령 키(미터). 사람보다 살짝 크면 더 으스스하다.")]
    public float ghostMeters = 1.8f;

    [Header("Drift")]
    [Tooltip("이동 속도 = 유령 키 × 이 비율 (초당).")]
    public float speedPerHeight = 0.30f;
    [Tooltip("스폰할 때 바닥 위로 이만큼 띄운다 (유령 키 대비 비율). 이후 고도는 자유.")]
    public float hoverPerHeight = 0.25f;
    [Tooltip("방향을 새로 뽑을 때 층을 넘나들 확률. 0이면 한 층에만 머문다.")]
    [Range(0f, 1f)] public float climbChance = 0.35f;
    [Tooltip("오르내리는 속도 = 유령 키 × 이 비율 (초당).")]
    public float climbSpeedRatio = 0.22f;
    [Tooltip("새 방향을 뽑는 간격(초) 최소/최대.")]
    public Vector2 headingInterval = new Vector2(3f, 8f);
    [Tooltip("위아래 부유 진폭(유령 키 대비 비율) / 주기(초).")]
    public float bobPerHeight = 0.06f;
    public float bobPeriod = 4.5f;

    [Header("Model")]
    // Assets/Resources/Ghosts/*.glb — CC0 models, see that folder's README.
    // UnityGLTF (Packages/manifest.json) imports .glb, same as the house.
    [Tooltip("비우면 Resources/Ghosts 폴더의 모델을 전부 읽어 무작위로 섞어 쓴다. " +
             "채우면 그것만 쓴다.")]
    public GameObject[] ghostModels;
    [Tooltip("ghostModels 가 비었을 때 읽을 Resources 하위 폴더.")]
    public string modelsPath = "Ghosts";
    [Tooltip("모델이 옆이나 뒤를 보고 있으면 이 각도로 돌린다.")]
    public float modelYawOffset = 0f;
    [Tooltip("모델 재질을 반투명 + 발광으로 덮어 유령처럼 만든다. 끄면 원본 그대로.")]
    public bool makeTranslucent = true;
    [Tooltip("모델에 딸려온 애니메이션 클립을 재생한다. 끄면 바인드 포즈(T 자세)로 굳는다.")]
    public bool playAnimation = true;
    [Tooltip("클립 이름에 이게 들어가면 우선 고른다. 앞쪽이 더 높은 우선순위.")]
    public string[] clipPreference = { "flying_idle", "idle", "walk", "float", "run" };

    [Header("Appearance")]
    [Tooltip("몸통 알파. 낮을수록 흐릿. 모델에도 같이 적용된다.")]
    [Range(0.05f, 1f)] public float bodyAlpha = 0.38f;
    public Color ghostTint = new Color(0.78f, 0.86f, 0.92f);
    // HorrorAtmosphere runs the house near-black, so a purely lit ghost is a
    // silhouette against nothing. The emission is what makes it readable there.
    [Tooltip("은은한 발광 세기. 0이면 빛나지 않음.")]
    public float emission = 0.9f;

    [Header("Verification")]
    // Off by default: it sits on top of CamcorderHUD's timestamp (bottom-left) and
    // is a debug readout, not part of the in-fiction overlay.
    [Tooltip("좌하단에 유령 목록 HUD를 그린다. 디버그용.")]
    public bool showHud = false;
    [Tooltip("누르면 다음 유령을 카메라 정면으로 소환한다.")]
    public KeyCode summonKey = KeyCode.G;
    [Tooltip("소환 거리 = 유령 키 × 이 비율.")]
    public float summonDistance = 1.6f;

    // One drifting figure. baseY is its altitude before the bob is added, so the
    // bob stays an offset instead of accumulating into the climb.
    class Ghost
    {
        public Transform root;
        public Vector3 heading;     // horizontal only
        public float climbRate;     // units/sec, signed — storeys are just altitude
        public float nextHeadingTime;
        public float baseY;
        public float bobPhase;
        public int recoveries;      // containment kicks — see Step()
    }

    private readonly List<Ghost> ghosts = new List<Ghost>();
    private readonly List<GameObject> models = new List<GameObject>();
    private AnimationClip[] clips = new AnimationClip[0];
    private readonly List<PlayableGraph> graphs = new List<PlayableGraph>();
    private TelloSimulator sim;
    private Bounds houseBounds;
    private bool haveBounds;
    private float height = 1f;
    private int summonIndex;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void Bootstrap()
    {
        // A scene-placed wanderer wins (so its Inspector values are used).
        if (FindFirstObjectByType<GhostWanderer>() != null)
        {
            return;
        }
        if (FindFirstObjectByType<TelloSimulator>() == null)
        {
            return; // not the sim scene
        }
        GameObject go = new GameObject("GhostWanderer");
        go.AddComponent<GhostWanderer>();
    }

    void Start()
    {
        sim = FindFirstObjectByType<TelloSimulator>();
        Respawn();
    }

    [ContextMenu("Respawn Ghosts")]
    public void Respawn()
    {
        Clear();
        if (!enableGhosts || ghostCount <= 0)
        {
            return;
        }
        if (randomSeed != 0)
        {
            Random.InitState(randomSeed);
        }

        height = ResolveHeight();
        haveBounds = TryMeasureHouse(out houseBounds);
        LoadModels();

        // Center on the drone's home. Read spawnPosition (not the live transform) so we
        // do not depend on whether TelloSimulator.Start ran before this one.
        Vector3 center = areaCenter;
        if (center == Vector3.zero && sim != null)
        {
            center = sim.spawnAtHome ? sim.spawnPosition : sim.transform.position;
        }

        float hover = height * hoverPerHeight;
        for (int i = 0; i < ghostCount; i++)
        {
            if (!TryFindFloor(center, out Vector3 floor))
            {
                continue;
            }
            ghosts.Add(Build($"ghost_{ghosts.Count + 1:00}", floor, hover));
        }

        Debug.Log($"<color=#9fd8ff>[Ghosts] {ghosts.Count}/{ghostCount} drifting " +
                  $"({models.Count} model(s), height {height:F1}u = {ghostMeters:F1}m, " +
                  $"hover {hover:F1}u) around {center}. " +
                  $"Press {summonKey} to summon one in front of the camera.</color>");
        if (ghosts.Count < ghostCount)
        {
            Debug.LogWarning($"[Ghosts] Only {ghosts.Count} spot(s) found — raise areaRadius " +
                             "or lower minSeparation.");
        }
    }

    void Update()
    {
        float dt = Time.deltaTime;
        foreach (Ghost g in ghosts)
        {
            if (g.root != null) Step(g, dt);
        }
        if (SummonPressed())
        {
            Summon();
        }
    }

    // One ghost, one frame.
    //
    // Storeys are not a thing here: a ghost climbs and sinks straight through
    // floors and ceilings like it goes through walls, so `climbRate` is just
    // another velocity component. What still holds it in is the FOOTPRINT — the
    // check is "is there any part of the building below this spot", scanned over
    // the building's whole height rather than the slab it happens to be over. An
    // earlier version tracked the floor the ghost stood on and rode it, which is
    // exactly what pinned each ghost to the storey it spawned on.
    void Step(Ghost g, float dt)
    {
        if (Time.time >= g.nextHeadingTime)
        {
            PickHeading(g, Random.insideUnitCircle.normalized);
        }

        Vector3 pos = g.root.position;
        Vector3 next = new Vector3(pos.x, g.baseY, pos.z)
                       + g.heading * (height * speedPerHeight * dt)
                       + Vector3.up * (g.climbRate * dt);

        if (InsideHouse(next))
        {
            pos.x = next.x;
            pos.z = next.z;
            g.baseY = next.y;
        }
        else
        {
            // Edge of the house. Turn back, and stop climbing in case it was the
            // roof or the ground that rejected the step.
            g.recoveries++;
            PickHeading(g, -new Vector2(g.heading.x, g.heading.z)
                            + Random.insideUnitCircle * 0.6f);
            g.climbRate = -g.climbRate;
        }

        if (haveBounds)
        {
            float m = height * 0.5f;
            pos.x = Mathf.Clamp(pos.x, houseBounds.min.x + m, houseBounds.max.x - m);
            pos.z = Mathf.Clamp(pos.z, houseBounds.min.z + m, houseBounds.max.z - m);
            float loY = houseBounds.min.y + height * hoverPerHeight;
            float hiY = houseBounds.max.y - height;
            if (hiY > loY)
            {
                float clamped = Mathf.Clamp(g.baseY, loY, hiY);
                if (!Mathf.Approximately(clamped, g.baseY)) g.climbRate = -g.climbRate;
                g.baseY = clamped;
            }
        }

        float bob = Mathf.Sin((Time.time / Mathf.Max(0.1f, bobPeriod)) * Mathf.PI * 2f + g.bobPhase)
                    * (height * bobPerHeight);
        pos.y = g.baseY + bob;

        g.root.position = pos;
        g.root.rotation = Quaternion.Slerp(
            g.root.rotation, Quaternion.LookRotation(g.heading, Vector3.up),
            1f - Mathf.Exp(-2f * dt));
    }

    void PickHeading(Ghost g, Vector2 dir)
    {
        if (dir.sqrMagnitude < 0.0001f) dir = Vector2.up;
        dir.Normalize();
        g.heading = new Vector3(dir.x, 0f, dir.y);
        // Mostly level, sometimes rising or sinking through a floor. Signed, so a
        // ghost that goes up eventually comes back down on a later heading roll.
        g.climbRate = (Random.value < climbChance)
            ? Random.Range(-1f, 1f) * height * climbSpeedRatio
            : 0f;
        g.nextHeadingTime = Time.time + Random.Range(headingInterval.x, headingInterval.y);
    }

    // Containment. Inside = within the building box AND with building geometry
    // somewhere below, scanned over the full height so it holds at any altitude.
    // Without the ray part a ghost would drift out through an outside wall and
    // keep going to the corner of the bounding box, which is open air.
    bool InsideHouse(Vector3 p)
    {
        if (haveBounds)
        {
            float m = height * 0.5f;
            if (p.x < houseBounds.min.x + m || p.x > houseBounds.max.x - m) return false;
            if (p.z < houseBounds.min.z + m || p.z > houseBounds.max.z - m) return false;
        }
        float top = haveBounds ? houseBounds.max.y + 1f : p.y + searchAbove;
        float bottom = haveBounds ? houseBounds.min.y - 1f : p.y - maxFloorDrop;
        Vector3 origin = new Vector3(p.x, top, p.z);
        var hits = Physics.RaycastAll(origin, Vector3.down, top - bottom,
                                      environmentMask, QueryTriggerInteraction.Ignore);
        foreach (RaycastHit h in hits)
        {
            if (h.normal.y < 0.5f) continue;                        // wall, not a slab
            if (sim != null && (h.transform == sim.transform
                                || h.transform.IsChildOf(sim.transform))) continue;
            return true;
        }
        return false;
    }

    // Floor height under `p`, searched around the ghost's current floor so a ghost
    // on the 2nd storey does not snap down to the 1st. False = no floor = outside.
    bool FloorUnder(Vector3 p, float fromY, out float floorY)
    {
        floorY = fromY;
        Vector3 origin = new Vector3(p.x, fromY + searchAbove, p.z);
        var hits = Physics.RaycastAll(origin, Vector3.down, searchAbove + maxFloorDrop,
                                      environmentMask, QueryTriggerInteraction.Ignore);
        float best = float.NegativeInfinity;
        foreach (RaycastHit h in hits)
        {
            if (h.normal.y < 0.5f) continue;                       // wall, not floor
            if (sim != null && (h.transform == sim.transform
                                || h.transform.IsChildOf(sim.transform))) continue;
            if (h.point.y > best) best = h.point.y;                 // highest slab below
        }
        if (float.IsNegativeInfinity(best)) return false;
        floorY = best;
        return true;
    }

    bool TryFindFloor(Vector3 center, out Vector3 floor)
    {
        for (int t = 0; t < maxTriesPerGhost; t++)
        {
            float ang = Random.Range(0f, Mathf.PI * 2f);
            float r = Mathf.Sqrt(Random.value) * areaRadius;
            Vector3 p = center + new Vector3(Mathf.Cos(ang), 0f, Mathf.Sin(ang)) * r;
            if (!FloorUnder(p, center.y, out float y))
            {
                continue;
            }
            Vector3 candidate = new Vector3(p.x, y, p.z);
            bool tooClose = false;
            foreach (Ghost g in ghosts)
            {
                Vector3 d = g.root.position - candidate;
                d.y = 0f;
                if (d.magnitude < minSeparation) { tooClose = true; break; }
            }
            if (tooClose) continue;

            floor = candidate;
            return true;
        }
        floor = Vector3.zero;
        return false;
    }

    // One ghost. A downloaded model is used when there is one (see LoadModels);
    // the primitive build below is the fallback so the component still works in a
    // project without the Resources/Ghosts folder.
    Ghost Build(string name, Vector3 floor, float hover)
    {
        GameObject root = new GameObject(name);
        root.transform.SetParent(transform, false);
        root.transform.position = floor + Vector3.up * hover;

        if (models.Count > 0)
        {
            AttachModel(root.transform, models[Random.Range(0, models.Count)]);
        }
        else
        {
            BuildPrimitive(root.transform);
        }

        Ghost built = new Ghost
        {
            root = root.transform,
            baseY = floor.y + hover,
            bobPhase = Random.Range(0f, Mathf.PI * 2f),
        };
        PickHeading(built, Random.insideUnitCircle.normalized);
        return built;
    }

    // Drop the model in, scaled so it is exactly `height` tall and standing on the
    // root, whatever the source file's units and pivot were. Every collider goes:
    // no trigger, no physics, nothing for the drone's probe to hit.
    void AttachModel(Transform root, GameObject prefab)
    {
        // The prefab's own root rotation/scale is kept and multiplied into, never
        // replaced: the glTF importer bakes the right-handed→left-handed conversion
        // there, and overwriting it mirrors or lays the model on its side.
        GameObject go = Instantiate(prefab, root, false);
        go.transform.localPosition = Vector3.zero;
        go.transform.localRotation = Quaternion.Euler(0f, modelYawOffset, 0f)
                                     * go.transform.localRotation;

        foreach (Collider c in go.GetComponentsInChildren<Collider>()) Destroy(c);

        var rends = go.GetComponentsInChildren<Renderer>();
        if (rends.Length == 0)
        {
            Destroy(go);
            BuildPrimitive(root);
            return;
        }

        // Measure what it actually renders as — a model file's units are anyone's
        // guess (metres, centimetres, "1 blender unit"), and the importer may have
        // baked a scale of its own on top.
        Bounds b = rends[0].bounds;
        foreach (Renderer r in rends) b.Encapsulate(r.bounds);
        if (b.size.y > 0.0001f)
        {
            go.transform.localScale *= height / b.size.y;
        }
        // Re-measure after scaling and sit the feet on the root's origin.
        b = rends[0].bounds;
        foreach (Renderer r in rends) b.Encapsulate(r.bounds);
        go.transform.localPosition = new Vector3(0f, root.position.y - b.min.y, 0f);

        if (makeTranslucent)
        {
            foreach (Renderer r in rends) Spectralize(r);
        }
        if (playAnimation)
        {
            PlayIdle(go);
        }
    }

    // Push the imported material toward "apparition": alpha down, faint glow up.
    // Everything is HasProperty-guarded because the glTF importer's shader is not
    // URP Lit and does not carry the same property names.
    void Spectralize(Renderer r)
    {
        foreach (Material m in r.materials)
        {
            foreach (string prop in new[] { "_BaseColor", "_Color" })
            {
                if (!m.HasProperty(prop)) continue;
                Color c = m.GetColor(prop);
                m.SetColor(prop, new Color(
                    Mathf.Lerp(c.r, ghostTint.r, 0.5f),
                    Mathf.Lerp(c.g, ghostTint.g, 0.5f),
                    Mathf.Lerp(c.b, ghostTint.b, 0.5f), bodyAlpha));
            }
            if (m.HasProperty("_Surface")) m.SetFloat("_Surface", 1f);
            if (m.HasProperty("_Blend")) m.SetFloat("_Blend", 0f);
            if (m.HasProperty("_Mode")) m.SetFloat("_Mode", 2f);
            if (m.HasProperty("_SrcBlend"))
                m.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            if (m.HasProperty("_DstBlend"))
                m.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            if (m.HasProperty("_ZWrite")) m.SetInt("_ZWrite", 0);
            m.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
            m.EnableKeyword("_ALPHABLEND_ON");
            m.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;

            if (emission > 0f && m.HasProperty("_EmissionColor"))
            {
                m.EnableKeyword("_EMISSION");
                m.SetColor("_EmissionColor", ghostTint * emission);
            }
        }
    }

    // Fallback figure: capsule shroud + head + two dark eye hollows.
    void BuildPrimitive(Transform root)
    {
        float radius = height * 0.17f;
        Material shroud = MakeGhostMaterial(ghostTint, bodyAlpha, emission);

        GameObject body = Primitive(PrimitiveType.Capsule, root.transform, shroud);
        float bodyH = height * 0.72f;
        body.transform.localPosition = new Vector3(0f, bodyH * 0.5f, 0f);
        // Unity's capsule primitive is 2 units tall, 1 unit across.
        body.transform.localScale = new Vector3(radius * 2f, bodyH * 0.5f, radius * 2f);

        GameObject head = Primitive(PrimitiveType.Sphere, root.transform, shroud);
        float headD = height * 0.30f;
        head.transform.localPosition = new Vector3(0f, bodyH * 0.95f, 0f);
        head.transform.localScale = Vector3.one * headD;

        // Trailing wisp instead of legs — squashed sphere fading out at the bottom.
        GameObject tail = Primitive(PrimitiveType.Sphere, root.transform,
                                    MakeGhostMaterial(ghostTint, bodyAlpha * 0.5f, emission * 0.5f));
        tail.transform.localPosition = new Vector3(0f, height * 0.05f, 0f);
        tail.transform.localScale = new Vector3(radius * 2.1f, height * 0.22f, radius * 2.1f);

        Material eyeMat = MakeGhostMaterial(new Color(0.03f, 0.03f, 0.05f), 0.85f, 0f);
        for (int s = -1; s <= 1; s += 2)
        {
            GameObject eye = Primitive(PrimitiveType.Sphere, root.transform, eyeMat);
            eye.transform.localPosition = new Vector3(s * headD * 0.20f,
                                                      bodyH * 0.95f + headD * 0.06f,
                                                      headD * 0.42f);
            eye.transform.localScale = Vector3.one * (headD * 0.20f);
        }
    }

    static GameObject Primitive(PrimitiveType type, Transform parent, Material mat)
    {
        GameObject go = GameObject.CreatePrimitive(type);
        go.transform.SetParent(parent, false);
        go.GetComponent<Renderer>().material = mat;
        Collider c = go.GetComponent<Collider>();
        if (c != null) Destroy(c);
        return go;
    }

    // Transparent material for whichever pipeline the project is on. URP Lit needs
    // the surface-type switch flipped by hand from script (setting only the color's
    // alpha leaves it opaque); Standard needs its Fade mode set the same way.
    static Material MakeGhostMaterial(Color tint, float alpha, float emission)
    {
        Shader shader = Shader.Find("Universal Render Pipeline/Lit");
        bool urp = shader != null;
        if (shader == null) shader = Shader.Find("Standard");
        if (shader == null) shader = Shader.Find("Sprites/Default");

        Color c = new Color(tint.r, tint.g, tint.b, alpha);
        Material m = new Material(shader) { color = c };
        if (m.HasProperty("_BaseColor")) m.SetColor("_BaseColor", c);

        if (urp)
        {
            m.SetFloat("_Surface", 1f);          // 0 opaque, 1 transparent
            m.SetFloat("_Blend", 0f);            // alpha blend
            m.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
        }
        else
        {
            m.SetFloat("_Mode", 2f);             // Fade
            m.EnableKeyword("_ALPHABLEND_ON");
            m.DisableKeyword("_ALPHATEST_ON");
            m.DisableKeyword("_ALPHAPREMULTIPLY_ON");
        }
        m.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
        m.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
        m.SetInt("_ZWrite", 0);
        m.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;

        if (emission > 0f)
        {
            m.EnableKeyword("_EMISSION");
            m.SetColor("_EmissionColor", tint * emission);
        }
        return m;
    }

    float ResolveHeight()
    {
        return Mathf.Max(0.5f, ghostMeters * unitsPerMeter);
    }

    // Inspector list wins; otherwise everything under Resources/Ghosts. Loading by
    // folder means dropping another .glb in there is the whole job of adding a new
    // ghost — no scene wiring, no code change.
    void LoadModels()
    {
        models.Clear();
        if (ghostModels != null && ghostModels.Length > 0)
        {
            foreach (GameObject g in ghostModels)
            {
                if (g != null) models.Add(g);
            }
        }
        if (models.Count == 0)
        {
            models.AddRange(Resources.LoadAll<GameObject>(modelsPath));
        }
        if (models.Count == 0)
        {
            Debug.LogWarning($"[Ghosts] no models under Resources/{modelsPath} — " +
                             "falling back to primitive figures.");
        }
        // All clips from all files in the folder, sorted out per model in PlayIdle.
        clips = playAnimation ? Resources.LoadAll<AnimationClip>(modelsPath)
                              : new AnimationClip[0];
    }

    // The models are rigged (Quaternius/Polygonal Mind ship Idle/Walk/Death clips)
    // but the glTF importer sets _addAnimatorComponent: 0, so out of the box they
    // render in the bind pose — a T-posed zombie standing to attention. There is no
    // AnimatorController to give them, and none is needed: a Playable graph plays a
    // bare clip on a bare Animator.
    //
    // Clips arrive as one flat pile for the whole folder, so they have to be paired
    // back to their own model or they silently bind to nothing. The pairing is the
    // rig name the exporter prefixes them with ("CharacterArmature|Idle"), matched
    // against the model's own transform names.
    void PlayIdle(GameObject go)
    {
        if (clips.Length == 0) return;

        var bones = new HashSet<string>();
        foreach (Transform t in go.GetComponentsInChildren<Transform>()) bones.Add(t.name);

        AnimationClip best = null;
        int bestRank = int.MaxValue;
        foreach (AnimationClip c in clips)
        {
            int bar = c.name.IndexOf('|');
            if (bar > 0 && !bones.Contains(c.name.Substring(0, bar))) continue;  // other rig
            string n = (bar >= 0 ? c.name.Substring(bar + 1) : c.name).ToLowerInvariant();
            int rank = clipPreference.Length;                                     // unranked
            for (int i = 0; i < clipPreference.Length; i++)
            {
                if (n.Contains(clipPreference[i].ToLowerInvariant())) { rank = i; break; }
            }
            if (rank < bestRank) { bestRank = rank; best = c; }
        }
        if (best == null) return;

        Animator anim = go.GetComponent<Animator>();
        if (anim == null) anim = go.AddComponent<Animator>();
        anim.applyRootMotion = false;   // the drift is ours; keep the clip in place
        AnimationPlayableUtilities.PlayClip(anim, best, out PlayableGraph graph);
        graphs.Add(graph);              // destroyed in Clear() — graphs are not GC'd
    }

    // Everything rendered that is not the drone and not a ghost — i.e. the house.
    bool TryMeasureHouse(out Bounds bounds)
    {
        bounds = new Bounds();
        bool found = false;
        foreach (Renderer r in FindObjectsByType<Renderer>(FindObjectsSortMode.None))
        {
            if (r is TrailRenderer || r is LineRenderer) continue;
            if (sim != null && r.transform.IsChildOf(sim.transform)) continue;
            if (r.transform.IsChildOf(transform)) continue;
            if (!found) { bounds = r.bounds; found = true; }
            else { bounds.Encapsulate(r.bounds); }
        }
        return found;
    }

    // --- verification ------------------------------------------------------

    // Teleport the next ghost into the camera's view, cycling 1..N so a few presses
    // confirm every ghost is alive. Placed relative to the CAMERA, at the camera's
    // own height — not relative to the drone. The drone only moves on UDP commands
    // from the Python side, so a drone-relative summon is unusable when the sim is
    // running standalone, and the ghost ends up below the chase cam's view anyway.
    // baseY is set to the camera's eye level and the climb zeroed, so it arrives
    // level with the view; it then drifts at the camera and through it.
    [ContextMenu("Summon Next Ghost")]
    public void Summon()
    {
        if (ghosts.Count == 0) return;
        Camera cam = Camera.main;
        Transform eye = (cam != null) ? cam.transform
                                      : (sim != null ? sim.transform : transform);

        summonIndex = (summonIndex + 1) % ghosts.Count;
        Ghost g = ghosts[summonIndex];
        Vector3 fwd = eye.forward;
        fwd.y = 0f;
        if (fwd.sqrMagnitude < 0.0001f) fwd = Vector3.forward;
        fwd.Normalize();

        Vector3 p = eye.position + fwd * (height * summonDistance);
        g.baseY = eye.position.y - height * 0.35f;            // roughly eye-level torso
        g.climbRate = 0f;
        g.root.position = new Vector3(p.x, g.baseY, p.z);
        g.root.rotation = Quaternion.LookRotation(-fwd, Vector3.up);   // face the camera
        PickHeading(g, new Vector2(-fwd.x, -fwd.z));                   // drift toward it
        Debug.Log($"<color=#9fd8ff>[Ghosts] summoned {g.root.name} at {g.root.position} " +
                  $"(height {height:F1}u, {height / Mathf.Max(0.01f, unitsPerMeter):F2}m)</color>");
    }

    bool SummonPressed()
    {
#if ENABLE_INPUT_SYSTEM
        // Same dual-backend guard as CameraFollow/HorrorAtmosphere — the project
        // ships the Input System package, so UnityEngine.Input alone throws.
        var kb = UnityEngine.InputSystem.Keyboard.current;
        if (kb != null && kb.gKey.wasPressedThisFrame) return true;
#endif
#if ENABLE_LEGACY_INPUT_MANAGER
        if (Input.GetKeyDown(summonKey)) return true;
#endif
        return false;
    }

    // Distances are measured from the camera, not the drone — the drone can sit
    // parked at home for a whole session (it only moves on UDP commands), so a
    // drone-relative readout says nothing about what is on screen.
    void OnGUI()
    {
        if (!showHud) return;
        Camera cam = Camera.main;
        Transform eye = (cam != null) ? cam.transform : transform;

        // Never draw nothing: 0 ghosts is itself the finding, and a silent HUD
        // looks identical to a broken script.
        int lines = Mathf.Max(1, ghosts.Count) + 2;
        float w = 250f;
        float h = 16f * lines + 12f;
        GUI.color = new Color(0f, 0f, 0f, 0.55f);
        GUI.DrawTexture(new Rect(8f, Screen.height - h - 8f, w, h), Texture2D.whiteTexture);
        GUI.color = new Color(0.72f, 0.86f, 0.95f);

        float y = Screen.height - h - 4f;
        GUI.Label(new Rect(14f, y, w - 12f, 18f),
                  $"GHOSTS {ghosts.Count}  h={height:F1}u  [{summonKey}] summon");
        y += 18f;
        if (ghosts.Count == 0)
        {
            GUI.color = new Color(1f, 0.6f, 0.5f);
            GUI.Label(new Rect(14f, y, w - 12f, 18f), "none placed — see Console");
            GUI.color = Color.white;
            return;
        }
        for (int i = 0; i < ghosts.Count; i++)
        {
            Ghost g = ghosts[i];
            float d = Vector3.Distance(eye.position, g.root.position);
            string mark = (i == summonIndex) ? ">" : " ";
            GUI.Label(new Rect(14f, y, w - 12f, 18f),
                      $"{mark}{i + 1:00}  {d,7:F1}u  y {g.root.position.y,6:F1}  turn {g.recoveries}");
            y += 16f;
        }
        GUI.color = Color.white;
    }

    // Scene view: ghosts show through walls as wire spheres, and the containment
    // box is drawn, so placement can be checked without hunting the Game view.
    void OnDrawGizmos()
    {
        Gizmos.color = new Color(0.6f, 0.9f, 1f, 0.9f);
        foreach (Ghost g in ghosts)
        {
            if (g.root == null) continue;
            Gizmos.DrawWireSphere(g.root.position + Vector3.up * height * 0.4f, height * 0.25f);
            Gizmos.DrawRay(g.root.position, g.heading * height * 0.6f);
        }
        if (haveBounds)
        {
            Gizmos.color = new Color(0.4f, 0.7f, 1f, 0.35f);
            Gizmos.DrawWireCube(houseBounds.center, houseBounds.size);
        }
    }

    void Clear()
    {
        // PlayableGraphs are unmanaged — dropping the GameObject does not free them,
        // and Respawn is on a ContextMenu, so they would pile up per press.
        foreach (PlayableGraph graph in graphs)
        {
            if (graph.IsValid()) graph.Destroy();
        }
        graphs.Clear();

        foreach (Ghost g in ghosts)
        {
            if (g.root != null) Destroy(g.root.gameObject);
        }
        ghosts.Clear();
        summonIndex = 0;
    }

    void OnDestroy()
    {
        Clear();
    }
}
