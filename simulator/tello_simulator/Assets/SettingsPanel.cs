using UnityEngine;

// In-game settings panel (Tab). Two groups:
//   경로 표시 — flight trail, planned path, flown trajectory, collision markers
//   사운드     — master + per-layer volumes
//
// Everything is applied live and stored in PlayerPrefs, so a preference survives
// the next Play. Bootstraps itself like the other overlays; no scene editing.
//
// The visuals it toggles are debug aids that break the horror mood (bright unlit
// lines through a dark house), which is why they are worth switching off without
// leaving Play or hunting through the Inspector.
[DisallowMultipleComponent]
public class SettingsPanel : MonoBehaviour
{
    const string PrefPrefix = "tello.settings.";

    [Header("경로 표시")]
    public bool showTrail = true;
    public bool showPlannedPath = true;
    public bool showFlightReport = true;
    public bool showCollisionMarkers = true;

    [Header("사운드")]
    [Range(0f, 1f)] public float masterVolume = 1f;
    public bool muted = false;

    [Header("Panel")]
    public bool open = false;
    public KeyCode toggleKey = KeyCode.Tab;

    private TelloSimulator sim;
    private PlannedPathRenderer plannedPath;
    private FlightReportRenderer flightReport;
    private HorrorAudio audioRig;
    private Rect window = new Rect(20f, 92f, 330f, 430f);
    private GUIStyle header;
    private bool applied;
    private bool audioPrefsLoaded;
    private bool prefsDirty;
    private float bindDeadline;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void Bootstrap()
    {
        if (FindFirstObjectByType<SettingsPanel>() != null) return;
        new GameObject("SettingsPanel").AddComponent<SettingsPanel>();
    }

    void Start()
    {
        sim = FindFirstObjectByType<TelloSimulator>();
        plannedPath = FindFirstObjectByType<PlannedPathRenderer>();
        flightReport = FindFirstObjectByType<FlightReportRenderer>();
        audioRig = FindFirstObjectByType<HorrorAudio>();
        bindDeadline = Time.unscaledTime + 5f;
        Load();
        Apply();
    }

    void Update()
    {
        if (TogglePressed())
        {
            open = !open;
            if (!open) Flush();
        }

        // The renderers create their line/markers lazily, and HorrorAudio is added
        // during another component's Start, so the first Apply can run too early.
        // Retry briefly, then give up rather than searching the scene forever.
        if (!applied && Time.unscaledTime < bindDeadline) Apply();
    }

    void Load()
    {
        showTrail = GetBool("trail", showTrail);
        showPlannedPath = GetBool("plannedPath", showPlannedPath);
        showFlightReport = GetBool("flightReport", showFlightReport);
        showCollisionMarkers = GetBool("collisionMarkers", showCollisionMarkers);
        masterVolume = PlayerPrefs.GetFloat(PrefPrefix + "master", masterVolume);
        muted = GetBool("muted", muted);
        LoadAudioPrefs();
    }

    // Separate from Load(): HorrorAudio is created during HorrorAtmosphere.Start,
    // which may run after this component's Start, so its prefs are read whenever
    // the rig first turns up.
    void LoadAudioPrefs()
    {
        if (audioRig == null || audioPrefsLoaded) return;
        audioPrefsLoaded = true;
        audioRig.ambientVolume = PlayerPrefs.GetFloat(PrefPrefix + "ambient", audioRig.ambientVolume);
        audioRig.stingerVolume = PlayerPrefs.GetFloat(PrefPrefix + "stinger", audioRig.stingerVolume);
        audioRig.heartbeatMaxVolume = PlayerPrefs.GetFloat(PrefPrefix + "heartbeat", audioRig.heartbeatMaxVolume);
        audioRig.droneVolumeIdle = PlayerPrefs.GetFloat(PrefPrefix + "droneIdle", audioRig.droneVolumeIdle);
        audioRig.droneVolumeMax = PlayerPrefs.GetFloat(PrefPrefix + "droneMax", audioRig.droneVolumeMax);
    }

    void Save()
    {
        SetBool("trail", showTrail);
        SetBool("plannedPath", showPlannedPath);
        SetBool("flightReport", showFlightReport);
        SetBool("collisionMarkers", showCollisionMarkers);
        PlayerPrefs.SetFloat(PrefPrefix + "master", masterVolume);
        SetBool("muted", muted);
        if (audioRig != null)
        {
            PlayerPrefs.SetFloat(PrefPrefix + "ambient", audioRig.ambientVolume);
            PlayerPrefs.SetFloat(PrefPrefix + "stinger", audioRig.stingerVolume);
            PlayerPrefs.SetFloat(PrefPrefix + "heartbeat", audioRig.heartbeatMaxVolume);
            PlayerPrefs.SetFloat(PrefPrefix + "droneIdle", audioRig.droneVolumeIdle);
            PlayerPrefs.SetFloat(PrefPrefix + "droneMax", audioRig.droneVolumeMax);
        }
        // No PlayerPrefs.Save() here: dragging a slider calls this every frame and
        // Save() writes to disk. Flushed on close / disable instead.
        prefsDirty = true;
    }

    void Flush()
    {
        if (!prefsDirty) return;
        prefsDirty = false;
        PlayerPrefs.Save();
    }

    void OnDisable()
    {
        Flush();
    }

    void OnApplicationQuit()
    {
        Flush();
    }

    static bool GetBool(string key, bool fallback)
    {
        return PlayerPrefs.GetInt(PrefPrefix + key, fallback ? 1 : 0) != 0;
    }

    static void SetBool(string key, bool value)
    {
        PlayerPrefs.SetInt(PrefPrefix + key, value ? 1 : 0);
    }

    void Apply()
    {
        if (audioRig == null) audioRig = FindFirstObjectByType<HorrorAudio>();
        if (plannedPath == null) plannedPath = FindFirstObjectByType<PlannedPathRenderer>();
        if (flightReport == null) flightReport = FindFirstObjectByType<FlightReportRenderer>();
        LoadAudioPrefs();

        if (sim != null)
        {
            sim.SetTrailVisible(showTrail);
            sim.SetCollisionMarkersVisible(showCollisionMarkers);
        }
        if (plannedPath != null) plannedPath.SetVisible(showPlannedPath);
        if (flightReport != null) flightReport.SetVisible(showFlightReport);

        AudioListener.volume = muted ? 0f : masterVolume;
        applied = sim != null && audioRig != null;
    }

    void OnGUI()
    {
        if (!open) return;
        if (header == null)
        {
            header = new GUIStyle(GUI.skin.label) { fontStyle = FontStyle.Bold };
        }
        window = GUILayout.Window(GetInstanceID(), window, DrawWindow, "설정  (Tab)");
    }

    void DrawWindow(int id)
    {
        bool changed = false;

        GUILayout.Space(4f);
        GUILayout.Label("경로 표시", header);
        changed |= Toggle(ref showTrail, "비행 트레일 (지나온 경로)");
        changed |= Toggle(ref showPlannedPath, "계획 경로 (A* 결과)");
        changed |= Toggle(ref showFlightReport, "비행 리포트 (실제 궤적)");
        changed |= Toggle(ref showCollisionMarkers, "충돌 마커");

        GUILayout.Space(10f);
        GUILayout.Label("사운드", header);
        changed |= Slider(ref masterVolume, "마스터");
        bool wasMuted = muted;
        muted = GUILayout.Toggle(muted, " 전체 음소거");
        changed |= muted != wasMuted;

        if (audioRig != null)
        {
            changed |= Slider(ref audioRig.ambientVolume, "앰비언트");
            changed |= Slider(ref audioRig.droneVolumeIdle, "로터 (호버링)");
            changed |= Slider(ref audioRig.droneVolumeMax, "로터 (전속)");
            changed |= Slider(ref audioRig.stingerVolume, "스팅어");
            changed |= Slider(ref audioRig.heartbeatMaxVolume, "심박");
        }
        else
        {
            GUILayout.Label("HorrorAudio 없음 — 볼륨 슬라이더 비활성");
        }

        GUILayout.Space(8f);
        if (GUILayout.Button("닫기"))
        {
            open = false;
            Flush();
        }

        if (changed)
        {
            Apply();
            Save();
        }

        GUI.DragWindow(new Rect(0f, 0f, 10000f, 20f));
    }

    static bool Toggle(ref bool value, string label)
    {
        bool now = GUILayout.Toggle(value, " " + label);
        bool changed = now != value;
        value = now;
        return changed;
    }

    static bool Slider(ref float value, string label)
    {
        GUILayout.BeginHorizontal();
        GUILayout.Label(label, GUILayout.Width(110f));
        float now = GUILayout.HorizontalSlider(value, 0f, 1f, GUILayout.Width(130f));
        GUILayout.Label($"{Mathf.RoundToInt(now * 100f)}%", GUILayout.Width(42f));
        GUILayout.EndHorizontal();

        bool changed = !Mathf.Approximately(now, value);
        value = now;
        return changed;
    }

    // Same dual-backend guard as CameraFollow.TogglePressed.
    bool TogglePressed()
    {
#if ENABLE_INPUT_SYSTEM
        var kb = UnityEngine.InputSystem.Keyboard.current;
        if (kb != null && kb.tabKey.wasPressedThisFrame) return true;
#endif
#if ENABLE_LEGACY_INPUT_MANAGER
        if (Input.GetKeyDown(toggleKey)) return true;
#endif
        return false;
    }
}
