using UnityEngine;

// Sound bed for the horror sim, driven by HorrorAtmosphere (added automatically
// alongside it). Three layers:
//   ambient   — 2D room tone, looping
//   stingers  — one-shots fired at random 3D positions around the drone, so the
//               noise seems to come from somewhere in the building
//   heartbeat — 2D loop that swells as the drone closes on the preview candidate
//
// Every clip is optional. With no clips assigned the component is silent and
// harmless, so the scene runs before any audio asset lands in the project.
// Assign clips in the Inspector, or drop files into Assets/Resources/Audio/
// (ambient.wav, heartbeat.wav, Stingers/*.wav) to have them auto-loaded.
[DisallowMultipleComponent]
public class HorrorAudio : MonoBehaviour
{
    [Header("Clips (optional — leave empty to disable that layer)")]
    public AudioClip ambientLoop;
    public AudioClip[] stingers;
    public AudioClip heartbeat;

    [Header("Ambient")]
    public float ambientVolume = 0.35f;

    [Header("Stingers")]
    public float stingerMinDelay = 12f;
    public float stingerMaxDelay = 35f;
    public float stingerMinDistance = 8f;
    public float stingerMaxDistance = 18f;
    public float stingerVolume = 0.7f;

    [Header("Heartbeat")]
    [Tooltip("Distance to the preview target at which the heartbeat starts to be heard.")]
    public float heartbeatFarDistance = 40f;
    [Tooltip("Distance at which it reaches full volume.")]
    public float heartbeatNearDistance = 6f;
    public float heartbeatMaxVolume = 0.6f;

    private TelloSimulator sim;
    private AudioSource ambientSource;
    private AudioSource heartbeatSource;
    private AudioSource stingerSource;
    private float nextStingerTime;
    private bool active = true;
    private bool initialized;

    void Start()
    {
        Init();
    }

    void Init()
    {
        if (initialized) return;
        initialized = true;

        sim = FindFirstObjectByType<TelloSimulator>();
        LoadFallbackClips();

        ambientSource = MakeSource("AmbientSource", spatial: false);
        ambientSource.loop = true;
        ambientSource.volume = ambientVolume;
        ambientSource.clip = ambientLoop;

        heartbeatSource = MakeSource("HeartbeatSource", spatial: false);
        heartbeatSource.loop = true;
        heartbeatSource.volume = 0f;
        heartbeatSource.clip = heartbeat;

        stingerSource = MakeSource("StingerSource", spatial: true);
        stingerSource.loop = false;
        stingerSource.volume = 1f;   // PlayOneShot applies stingerVolume as the scale

        ScheduleNextStinger();
        SetActive(active);
    }

    // Inspector assignment wins; this only fills the gaps so clips dropped into
    // Resources/Audio/ work even when the rig is spawned at runtime (no scene
    // object to drag them onto).
    void LoadFallbackClips()
    {
        if (ambientLoop == null) ambientLoop = Resources.Load<AudioClip>("Audio/ambient");
        if (heartbeat == null) heartbeat = Resources.Load<AudioClip>("Audio/heartbeat");
        if (stingers == null || stingers.Length == 0)
        {
            stingers = Resources.LoadAll<AudioClip>("Audio/Stingers");
        }
    }

    AudioSource MakeSource(string sourceName, bool spatial)
    {
        GameObject go = new GameObject(sourceName);
        go.transform.SetParent(transform, false);
        AudioSource src = go.AddComponent<AudioSource>();
        src.playOnAwake = false;
        src.spatialBlend = spatial ? 1f : 0f;
        if (spatial)
        {
            src.rolloffMode = AudioRolloffMode.Logarithmic;
            src.minDistance = 3f;
            src.maxDistance = stingerMaxDistance * 2f;
        }
        return src;
    }

    // Called by HorrorAtmosphere when the L key flips horror mode.
    public void SetActive(bool on)
    {
        active = on;
        if (!initialized) return;   // Init() re-applies once the sources exist

        if (on)
        {
            if (ambientSource.clip != null && !ambientSource.isPlaying) ambientSource.Play();
            if (heartbeatSource.clip != null && !heartbeatSource.isPlaying) heartbeatSource.Play();
            ScheduleNextStinger();
        }
        else
        {
            ambientSource.Stop();
            heartbeatSource.Stop();
            stingerSource.Stop();
        }
    }

    void Update()
    {
        if (!active || !initialized) return;
        UpdateStingers();
        UpdateHeartbeat();
    }

    void UpdateStingers()
    {
        if (stingers == null || stingers.Length == 0) return;
        if (Time.time < nextStingerTime) return;
        ScheduleNextStinger();

        // Random point on a horizontal ring around the drone — the listener hears
        // it off to one side without knowing what made it.
        Vector3 origin = sim != null ? sim.transform.position : transform.position;
        float angle = Random.Range(0f, Mathf.PI * 2f);
        float dist = Random.Range(stingerMinDistance, stingerMaxDistance);
        Vector3 at = origin + new Vector3(Mathf.Cos(angle) * dist,
                                          Random.Range(-2f, 4f),
                                          Mathf.Sin(angle) * dist);

        stingerSource.transform.position = at;
        stingerSource.pitch = Random.Range(0.9f, 1.1f);
        stingerSource.PlayOneShot(stingers[Random.Range(0, stingers.Length)], stingerVolume);
    }

    void ScheduleNextStinger()
    {
        nextStingerTime = Time.time + Random.Range(stingerMinDelay, stingerMaxDelay);
    }

    void UpdateHeartbeat()
    {
        if (heartbeatSource.clip == null) return;

        float t = 0f;
        if (sim != null && sim.previewActive)
        {
            float d = Vector3.Distance(sim.transform.position, sim.previewTarget);
            t = Mathf.InverseLerp(heartbeatFarDistance, heartbeatNearDistance, d);
        }

        // Ease toward the target so the swell is not a step when preview toggles.
        heartbeatSource.volume = Mathf.Lerp(heartbeatSource.volume,
                                            t * heartbeatMaxVolume,
                                            Time.deltaTime * 2f);
        heartbeatSource.pitch = Mathf.Lerp(1f, 1.35f, t);
    }
}
