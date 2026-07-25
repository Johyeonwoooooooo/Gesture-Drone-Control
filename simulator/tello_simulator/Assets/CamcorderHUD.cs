using System;
using UnityEngine;

// Found-footage camcorder overlay (Paranormal Activity style): blinking REC dot,
// tape counter, wall clock, draining battery, viewfinder brackets and the odd VHS
// tracking glitch. Purely cosmetic — nothing here feeds the flight or the UDP
// protocol, and the battery never grounds the drone.
//
// Bootstraps itself like HorrorAtmosphere, so no scene editing is needed. Draws
// on top of TelloSimulator's own OnGUI (status banner top-center, confirm buttons
// bottom-center); those areas are deliberately left empty here.
//
// Keys:  H = hide/show the HUD     N = night shot (green IR look)
[DisallowMultipleComponent]
public class CamcorderHUD : MonoBehaviour
{
    [Header("Display")]
    public bool visible = true;
    public string cameraId = "TELLO CAM-01";
    [Tooltip("Tape mode label next to the counter. SP/LP is the VHS speed setting.")]
    public string tapeMode = "SP";
    public Color hudColor = new Color(1f, 1f, 1f, 0.85f);

    [Header("Battery (cosmetic)")]
    [Tooltip("Real minutes for a full charge to reach empty while idle. Flying drains " +
             "at twice this rate. Purely a prop — it never affects the drone.")]
    public float batteryMinutes = 25f;
    public float batteryStart = 92f;

    [Header("Tracking Glitch")]
    public bool trackingGlitch = true;
    public float glitchMinDelay = 9f;
    public float glitchMaxDelay = 26f;
    public float glitchDuration = 0.45f;

    [Header("Night Shot")]
    public bool nightVision = false;

    private TelloSimulator sim;
    private HorrorAtmosphere atmosphere;
    private Texture2D px;
    private GUIStyle text;
    private GUIStyle textBold;
    private float battery;
    private float tapeSeconds;
    private float nextGlitchTime;
    private float glitchUntil;
    private float glitchSeed;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void Bootstrap()
    {
        if (FindFirstObjectByType<CamcorderHUD>() != null) return;
        new GameObject("CamcorderHUD").AddComponent<CamcorderHUD>();
    }

    void Start()
    {
        sim = FindFirstObjectByType<TelloSimulator>();
        atmosphere = FindFirstObjectByType<HorrorAtmosphere>();
        battery = batteryStart;
        px = new Texture2D(1, 1);
        px.SetPixel(0, 0, Color.white);
        px.Apply();
        ScheduleGlitch();
        if (nightVision) PushNightVision();
    }

    void Update()
    {
        tapeSeconds += Time.unscaledDeltaTime;

        bool flying = sim != null && sim.IsFlying;
        float drainPerSecond = 100f / Mathf.Max(batteryMinutes, 0.1f) / 60f;
        battery = Mathf.Max(0f, battery - drainPerSecond * (flying ? 2f : 1f) * Time.unscaledDeltaTime);

        if (HidePressed()) visible = !visible;
        if (NightPressed())
        {
            nightVision = !nightVision;
            PushNightVision();
        }

        if (trackingGlitch && Time.unscaledTime >= nextGlitchTime)
        {
            glitchUntil = Time.unscaledTime + glitchDuration;
            glitchSeed = Time.unscaledTime;
            ScheduleGlitch();
        }
    }

    void ScheduleGlitch()
    {
        nextGlitchTime = Time.unscaledTime + UnityEngine.Random.Range(glitchMinDelay, glitchMaxDelay);
    }

    void PushNightVision()
    {
        if (atmosphere != null) atmosphere.SetNightVision(nightVision);
    }

    void EnsureStyles()
    {
        if (text != null) return;
        Font mono = null;
        foreach (string candidate in new[] { "Consolas", "Menlo", "DejaVu Sans Mono", "Courier New" })
        {
            mono = Font.CreateDynamicFontFromOSFont(candidate, 15);
            if (mono != null) break;
        }
        text = new GUIStyle(GUI.skin.label) { fontSize = 15, richText = false };
        if (mono != null) text.font = mono;
        text.normal.textColor = Color.white;
        textBold = new GUIStyle(text) { fontSize = 17, fontStyle = FontStyle.Bold };
    }

    void OnGUI()
    {
        if (!visible) return;
        EnsureStyles();

        float w = Screen.width;
        float h = Screen.height;
        float m = 22f;                       // margin from the screen edge
        Color tint = nightVision ? new Color(0.72f, 1f, 0.78f, hudColor.a) : hudColor;

        DrawViewfinderCorners(w, h, m, tint);
        DrawFocusBrackets(w, h, tint);

        // --- top-left: REC dot + tape counter -------------------------------
        bool blink = Mathf.Repeat(Time.unscaledTime, 1.4f) < 0.75f;
        if (blink)
        {
            GUI.color = new Color(1f, 0.15f, 0.12f, 0.95f);
            GUI.DrawTexture(new Rect(m, m + 3f, 13f, 13f), px);
        }
        GUI.color = tint;
        GUI.Label(new Rect(m + 22f, m - 3f, 200f, 24f), "REC", textBold);
        GUI.Label(new Rect(m, m + 22f, 240f, 22f), $"{tapeMode}  {Timecode(tapeSeconds)}", text);

        // --- top-right: battery ---------------------------------------------
        DrawBattery(w - m - 74f, m + 1f, tint);

        // --- bottom-left: date + wall clock ----------------------------------
        DateTime now = DateTime.Now;
        GUI.color = tint;
        GUI.Label(new Rect(m, h - m - 46f, 320f, 24f), now.ToString("yyyy. MM. dd."), text);
        GUI.Label(new Rect(m, h - m - 24f, 320f, 24f),
                  now.ToString("tt hh:mm:ss").ToUpperInvariant(), textBold);

        // --- bottom-right: camera id + night shot -----------------------------
        GUIStyle right = new GUIStyle(text) { alignment = TextAnchor.MiddleRight };
        GUI.Label(new Rect(w - m - 320f, h - m - 46f, 320f, 24f), cameraId, right);
        if (nightVision)
        {
            GUI.color = new Color(0.55f, 1f, 0.6f, 0.95f);
            GUIStyle nv = new GUIStyle(textBold) { alignment = TextAnchor.MiddleRight };
            GUI.Label(new Rect(w - m - 320f, h - m - 24f, 320f, 24f), "◉ NIGHT SHOT", nv);
        }

        // --- key hints, dim, above the confirm buttons ------------------------
        GUI.color = new Color(tint.r, tint.g, tint.b, 0.4f);
        GUIStyle hint = new GUIStyle(text) { fontSize = 12, alignment = TextAnchor.MiddleCenter };
        GUI.Label(new Rect(0f, h - 20f, w, 18f),
                  "C 시점   L 호러   F 손전등   N 나이트샷   [ ] 밝기   H HUD", hint);

        if (Time.unscaledTime < glitchUntil) DrawTracking(w, h);

        GUI.color = Color.white;
    }

    static string Timecode(float seconds)
    {
        int s = Mathf.FloorToInt(seconds);
        return $"{s / 3600:00}:{(s / 60) % 60:00}:{s % 60:00}";
    }

    void DrawBattery(float x, float y, Color tint)
    {
        float bw = 44f, bh = 17f;
        bool low = battery <= 20f;
        bool blink = Mathf.Repeat(Time.unscaledTime, 0.9f) < 0.5f;

        GUI.color = tint;
        DrawOutline(new Rect(x, y, bw, bh), 1.5f);
        GUI.DrawTexture(new Rect(x + bw, y + 5f, 3f, 7f), px);   // the nub

        float fill = Mathf.Clamp01(battery / 100f) * (bw - 6f);
        GUI.color = low
            ? new Color(1f, 0.25f, 0.2f, blink ? 0.95f : 0.35f)
            : new Color(tint.r, tint.g, tint.b, 0.8f);
        GUI.DrawTexture(new Rect(x + 3f, y + 3f, fill, bh - 6f), px);

        GUI.color = low && blink ? new Color(1f, 0.4f, 0.35f) : tint;
        GUIStyle right = new GUIStyle(text) { alignment = TextAnchor.MiddleRight };
        GUI.Label(new Rect(x - 66f, y - 2f, 60f, 22f), $"{Mathf.CeilToInt(battery)}%", right);
    }

    void DrawOutline(Rect r, float t)
    {
        GUI.DrawTexture(new Rect(r.x, r.y, r.width, t), px);
        GUI.DrawTexture(new Rect(r.x, r.yMax - t, r.width, t), px);
        GUI.DrawTexture(new Rect(r.x, r.y, t, r.height), px);
        GUI.DrawTexture(new Rect(r.xMax - t, r.y, t, r.height), px);
    }

    // Four L-shaped brackets framing the whole image, like a viewfinder mask.
    void DrawViewfinderCorners(float w, float h, float m, Color tint)
    {
        GUI.color = new Color(tint.r, tint.g, tint.b, tint.a * 0.75f);
        float len = 26f, t = 2f;
        float l = m - 8f, r = w - m + 8f, top = m - 8f, bot = h - m + 8f;

        GUI.DrawTexture(new Rect(l, top, len, t), px);
        GUI.DrawTexture(new Rect(l, top, t, len), px);
        GUI.DrawTexture(new Rect(r - len, top, len, t), px);
        GUI.DrawTexture(new Rect(r - t, top, t, len), px);
        GUI.DrawTexture(new Rect(l, bot - t, len, t), px);
        GUI.DrawTexture(new Rect(l, bot - len, t, len), px);
        GUI.DrawTexture(new Rect(r - len, bot - t, len, t), px);
        GUI.DrawTexture(new Rect(r - t, bot - len, t, len), px);
    }

    // Small centre autofocus box — the detail that reads as "camcorder" more than
    // any label does.
    void DrawFocusBrackets(float w, float h, Color tint)
    {
        GUI.color = new Color(tint.r, tint.g, tint.b, tint.a * 0.35f);
        float bw = Mathf.Min(w, h) * 0.16f;
        float x = (w - bw) / 2f, y = (h - bw) / 2f;
        float len = bw * 0.28f, t = 1.5f;

        GUI.DrawTexture(new Rect(x, y, len, t), px);
        GUI.DrawTexture(new Rect(x, y, t, len), px);
        GUI.DrawTexture(new Rect(x + bw - len, y, len, t), px);
        GUI.DrawTexture(new Rect(x + bw - t, y, t, len), px);
        GUI.DrawTexture(new Rect(x, y + bw - t, len, t), px);
        GUI.DrawTexture(new Rect(x, y + bw - len, t, len), px);
        GUI.DrawTexture(new Rect(x + bw - len, y + bw - t, len, t), px);
        GUI.DrawTexture(new Rect(x + bw - t, y + bw - len, t, len), px);
    }

    // VHS tracking noise: a band of torn scanlines sweeping down the frame.
    void DrawTracking(float w, float h)
    {
        float t = (Time.unscaledTime - (glitchUntil - glitchDuration)) / glitchDuration;
        float bandY = Mathf.Lerp(-60f, h, t);
        float bandH = 46f;

        UnityEngine.Random.State state = UnityEngine.Random.state;
        UnityEngine.Random.InitState((int)(glitchSeed * 1000f));
        for (int i = 0; i < 9; i++)
        {
            float sliceY = bandY + UnityEngine.Random.Range(0f, bandH);
            float sliceH = UnityEngine.Random.Range(1f, 4f);
            float offset = UnityEngine.Random.Range(-24f, 24f);
            GUI.color = new Color(1f, 1f, 1f, UnityEngine.Random.Range(0.05f, 0.22f));
            GUI.DrawTexture(new Rect(offset, sliceY, w, sliceH), px);
        }
        UnityEngine.Random.state = state;

        GUI.color = new Color(1f, 1f, 1f, 0.55f);
        GUIStyle c = new GUIStyle(text) { alignment = TextAnchor.MiddleCenter };
        GUI.Label(new Rect(0f, h * 0.5f + 60f, w, 22f), "-- TRACKING --", c);
    }

    // Same dual-backend guard as CameraFollow.TogglePressed.
    bool HidePressed()
    {
#if ENABLE_INPUT_SYSTEM
        var kb = UnityEngine.InputSystem.Keyboard.current;
        if (kb != null && kb.hKey.wasPressedThisFrame) return true;
#endif
#if ENABLE_LEGACY_INPUT_MANAGER
        if (Input.GetKeyDown(KeyCode.H)) return true;
#endif
        return false;
    }

    bool NightPressed()
    {
#if ENABLE_INPUT_SYSTEM
        var kb = UnityEngine.InputSystem.Keyboard.current;
        if (kb != null && kb.nKey.wasPressedThisFrame) return true;
#endif
#if ENABLE_LEGACY_INPUT_MANAGER
        if (Input.GetKeyDown(KeyCode.N)) return true;
#endif
        return false;
    }

    void OnDestroy()
    {
        if (px != null) Destroy(px);
    }
}
