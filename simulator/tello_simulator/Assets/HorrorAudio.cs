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
    public AudioClip droneLoop;
    public AudioClip droneTakeoff;
    public AudioClip droneLand;

    [Header("Ambient")]
    public float ambientVolume = 0.35f;
    [Tooltip("Seconds of overlap between one pass of the ambient clip and the next. " +
             "Two sources play the clip staggered and cross-fade, so a clip whose " +
             "start and end do not match still loops without an audible seam. " +
             "Clamped to 40% of the clip length.")]
    public float ambientCrossfade = 3f;

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

    [Header("Drone")]
    [Tooltip("Rotor loop volume while hovering. Rises toward droneVolumeMax at full speed.")]
    public float droneVolumeIdle = 0.25f;
    public float droneVolumeMax = 0.55f;
    public float dronePitchIdle = 0.85f;
    public float dronePitchMax = 1.3f;
    [Tooltip("Speed (u/s) treated as full throttle for the pitch/volume ramp. " +
             "TelloSimulator.moveSpeed is 15 by default.")]
    public float droneFullSpeed = 15f;
    [Tooltip("Distance at which the rotors stop getting louder as you approach. " +
             "The chase camera sits ~4 u behind the drone, FPV sits on it.")]
    public float droneMinDistance = 3f;
    public float droneMaxDistance = 60f;

    private TelloSimulator sim;
    private readonly AudioSource[] ambientSources = new AudioSource[2];
    private readonly double[] ambientStart = new double[2];
    private double ambientNextTime;
    private int ambientNext;
    private AudioSource heartbeatSource;
    private AudioSource stingerSource;
    private AudioSource droneSource;
    private Vector3 dronePrevPos;
    private float droneSpeed;
    private bool droneWasFlying;
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

        for (int i = 0; i < ambientSources.Length; i++)
        {
            ambientSources[i] = MakeSource($"AmbientSource{i}", spatial: false);
            ambientSources[i].loop = false;   // the cross-fade schedules each pass
            ambientSources[i].volume = 0f;
            ambientSources[i].clip = ambientLoop;
        }

        heartbeatSource = MakeSource("HeartbeatSource", spatial: false);
        heartbeatSource.loop = true;
        heartbeatSource.volume = 0f;
        heartbeatSource.clip = heartbeat;

        stingerSource = MakeSource("StingerSource", spatial: true);
        stingerSource.loop = false;
        stingerSource.volume = 1f;   // PlayOneShot applies stingerVolume as the scale

        if (sim != null)
        {
            // Parented to the drone so the rotors pan and fall off with distance —
            // audible behind you in chase view, right on top of you in FPV.
            droneSource = MakeSource("DroneSource", spatial: true, parent: sim.transform);
            droneSource.loop = true;
            droneSource.volume = 0f;
            droneSource.pitch = dronePitchIdle;
            droneSource.minDistance = droneMinDistance;
            droneSource.maxDistance = droneMaxDistance;
            droneSource.clip = droneLoop;
            dronePrevPos = sim.transform.position;
            droneWasFlying = sim.IsFlying;
        }

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
        if (droneLoop == null) droneLoop = Resources.Load<AudioClip>("Audio/drone");
        if (droneTakeoff == null) droneTakeoff = Resources.Load<AudioClip>("Audio/drone_takeoff");
        if (droneLand == null) droneLand = Resources.Load<AudioClip>("Audio/drone_land");
        if (stingers == null || stingers.Length == 0)
        {
            stingers = Resources.LoadAll<AudioClip>("Audio/Stingers");
        }
    }

    AudioSource MakeSource(string sourceName, bool spatial, Transform parent = null)
    {
        GameObject go = new GameObject(sourceName);
        go.transform.SetParent(parent != null ? parent : transform, false);
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
            StartAmbient();
            if (heartbeatSource.clip != null && !heartbeatSource.isPlaying) heartbeatSource.Play();
            if (droneSource != null && droneSource.clip != null && !droneSource.isPlaying)
            {
                droneSource.Play();
            }
            ScheduleNextStinger();
        }
        else
        {
            foreach (AudioSource src in ambientSources)
            {
                if (src != null) src.Stop();
            }
            heartbeatSource.Stop();
            stingerSource.Stop();
            if (droneSource != null) droneSource.Stop();
        }
    }

    void Update()
    {
        if (!active || !initialized) return;
        UpdateAmbient();
        UpdateStingers();
        UpdateHeartbeat();
        UpdateDrone();
    }

    // Rotor loop: pitch and volume ride the drone's speed, so an rc burst is
    // audible as a spin-up. On the ground the rotors wind down to silence.
    void UpdateDrone()
    {
        if (droneSource == null || sim == null) return;


        float dt = Mathf.Max(Time.deltaTime, 1e-4f);
        Vector3 pos = sim.transform.position;
        float instant = (pos - dronePrevPos).magnitude / dt;
        dronePrevPos = pos;
        // A `setpos` teleport would read as an enormous speed; ignore those frames.
        if (instant < droneFullSpeed * 4f)
        {
            droneSpeed = Mathf.Lerp(droneSpeed, instant, dt * 6f);
        }

        // isFlying only goes true on the server's `takeoff`. But spawnAtHome drops
        // the drone in mid-air, so with Unity alone it visibly hovers with the
        // rotors silent. Anything off the floor counts as powered.
        bool airborne = pos.y > sim.minHeight + 0.25f;
        bool flying = sim.IsFlying;
        bool powered = flying || airborne;

        if (flying != droneWasFlying)
        {
            AudioClip accent = flying ? droneTakeoff : droneLand;
            // Not droneSource.PlayOneShot: that scales by the source volume, which
            // is still ramping up from zero at the moment of takeoff.
            if (accent != null) AudioSource.PlayClipAtPoint(accent, pos, 0.9f);
            droneWasFlying = flying;
        }

        float t = Mathf.Clamp01(droneSpeed / Mathf.Max(droneFullSpeed, 0.01f));
        float targetVolume = powered ? Mathf.Lerp(droneVolumeIdle, droneVolumeMax, t) : 0f;
        droneSource.volume = Mathf.Lerp(droneSource.volume, targetVolume, dt * 3f);
        droneSource.pitch = Mathf.Lerp(droneSource.pitch,
                                       Mathf.Lerp(dronePitchIdle, dronePitchMax, t),
                                       dt * 3f);
    }

    // --------------------------------------------------------------- ambient
    // A room tone downloaded from a sound library rarely has matching start and
    // end samples, so plain loop=true clicks once per pass. Instead two sources
    // play the same clip staggered by (length - crossfade) and equal-power
    // cross-fade into each other, which hides the seam whatever the clip does.
    //
    // Timing runs on AudioSettings.dspTime with PlayScheduled, not on Update:
    // the audio thread honours the schedule exactly even if a frame hitches,
    // where a frame-timed Play() would leave an audible gap.
    float AmbientCrossfade()
    {
        if (ambientLoop == null) return 0f;
        return Mathf.Clamp(ambientCrossfade, 0.05f, ambientLoop.length * 0.4f);
    }

    void StartAmbient()
    {
        if (ambientLoop == null || ambientSources[0] == null) return;
        if (ambientSources[0].isPlaying || ambientSources[1].isPlaying) return;

        ambientNext = 0;
        // Small lead-in so the first schedule is comfortably in the future.
        ambientNextTime = AudioSettings.dspTime + 0.15;
        ScheduleAmbient();   // this pass
        ScheduleAmbient();   // and the one it will fade into
    }

    void ScheduleAmbient()
    {
        if (ambientLoop == null) return;
        AudioSource src = ambientSources[ambientNext];
        src.clip = ambientLoop;
        src.volume = 0f;
        src.PlayScheduled(ambientNextTime);
        ambientStart[ambientNext] = ambientNextTime;

        ambientNextTime += Mathf.Max(0.1f, ambientLoop.length - AmbientCrossfade());
        ambientNext ^= 1;
    }

    void UpdateAmbient()
    {
        if (ambientLoop == null || ambientSources[0] == null) return;

        double now = AudioSettings.dspTime;
        float len = ambientLoop.length;
        float fade = AmbientCrossfade();

        for (int i = 0; i < ambientSources.Length; i++)
        {
            double elapsed = now - ambientStart[i];
            if (elapsed < 0.0 || elapsed > len)
            {
                ambientSources[i].volume = 0f;
                continue;
            }
            ambientSources[i].volume = Envelope((float)elapsed, len, fade) * ambientVolume;
        }

        // Queue the next pass once the source it will reuse has finished.
        if (now >= ambientStart[ambientNext] + len - 0.05) ScheduleAmbient();
    }

    // Equal-power ramp (sin/cos): a linear cross-fade dips in the middle because
    // two uncorrelated signals sum in power, not amplitude.
    static float Envelope(float elapsed, float length, float fade)
    {
        if (elapsed < fade) return Mathf.Sin(elapsed / fade * Mathf.PI * 0.5f);
        float remaining = length - elapsed;
        if (remaining < fade) return Mathf.Sin(remaining / fade * Mathf.PI * 0.5f);
        return 1f;
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

        // The heartbeat used to swell as the drone closed on a preview
        // candidate. There are no candidates any more (the operator picks areas
        // on the web floor plan before launch), so it is idle until something
        // drives it — the scan is the obvious hook.
        float t = 0f;

        // Ease toward the target so the swell is not a step.
        heartbeatSource.volume = Mathf.Lerp(heartbeatSource.volume,
                                            t * heartbeatMaxVolume,
                                            Time.deltaTime * 2f);
        heartbeatSource.pitch = Mathf.Lerp(1f, 1.35f, t);
    }
}
