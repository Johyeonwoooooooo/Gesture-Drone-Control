using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

// Horror lighting for the Tello sim: kills the daylight rig, drops thick fog,
// hangs a flashlight on the camera and stacks a URP post-processing volume.
//
// Nothing here is authored in test.unity — every light, volume and render
// setting is built at runtime, so the 8600-line scene YAML stays untouched.
// The component bootstraps itself on scene load; adding it to a GameObject by
// hand also works (the bootstrap skips when an instance already exists) and is
// the way to tune the values in the Inspector persistently.
//
// Keys:  L = horror mode on/off (back to the original bright scene)
//        F = flashlight on/off
[DisallowMultipleComponent]
public class HorrorAtmosphere : MonoBehaviour
{
    [Header("Mode")]
    [Tooltip("Applied on Start. Toggle at runtime with the L key.")]
    public bool horrorEnabled = true;
    public KeyCode toggleKey = KeyCode.L;
    public KeyCode flashlightKey = KeyCode.F;

    [Header("Brightness")]
    [Tooltip("Master multiplier over ambient, moonlight, flashlight, fill light and " +
             "exposure. Adjust live with the [ and ] keys — the current value is " +
             "printed to the Console so it can be pasted back here as the new default.")]
    [Range(0.25f, 6f)]
    public float brightness = 1f;
    public float brightnessStep = 0.25f;
    [Tooltip("Extra light multiplier while CamcorderHUD's night shot (N) is on.")]
    public float nightVisionBoost = 1.9f;

    [Header("Fog")]
    [Tooltip("The building GLB is placed at scale 5, so world distances are large — " +
             "small density values already swallow a corridor. Tune in 0.003 ~ 0.02.")]
    public float fogDensity = 0.006f;
    public Color fogColor = new Color(0.03f, 0.035f, 0.05f, 1f);

    [Header("Ambient / Sun")]
    [Tooltip("Indoors the directional light barely reaches, so this is what makes the " +
             "rooms readable at all. Raise it first when the scene is too dark.")]
    public Color ambientColor = new Color(0.11f, 0.12f, 0.16f, 1f);
    [Tooltip("Intensity the existing Directional Light is dimmed to (was 1.0).")]
    public float moonIntensity = 0.25f;
    public Color moonColor = new Color(0.55f, 0.65f, 0.9f, 1f);

    [Header("Flashlight")]
    public bool flashlightOn = true;
    [Tooltip("Mount on the camera (follows where you look, works in chase AND FPV). " +
             "Off = mount on the drone, which does not yaw, so it always points +Z.")]
    public bool attachToCamera = true;
    public float flashlightIntensity = 28f;
    public float flashlightRange = 70f;
    public float flashlightAngle = 68f;
    public Color flashlightColor = new Color(1f, 0.94f, 0.82f, 1f);

    [Header("Drone Fill Light")]
    [Tooltip("Dim point light on the drone. The spot cone alone leaves everything " +
             "outside it pitch black; this reveals the near geometry so you can tell " +
             "where the drone is without lifting the darkness.")]
    public bool fillLight = true;
    public float fillIntensity = 1.6f;
    public float fillRange = 30f;
    public Color fillColor = new Color(0.7f, 0.78f, 1f, 1f);

    [Header("Preview Highlight")]
    [Tooltip("Light up the candidate during a `preview` command — without it the " +
             "target is too dark to judge the [이동]/[다음 후보] buttons.")]
    public bool lightPreviewTarget = true;
    public float previewLightRange = 14f;
    public float previewLightIntensity = 6f;
    public Color previewLightColor = new Color(1f, 0.85f, 0.55f, 1f);

    [Header("Post Processing (URP Volume)")]
    public bool postProcessing = true;
    [Tooltip("Negative values darken the image on top of the lighting. Keep it near 0 " +
             "while the scene is still too dark to read.")]
    public float postExposure = -0.1f;
    public float saturation = -35f;
    public float contrast = 15f;
    public float vignetteIntensity = 0.42f;
    public float filmGrainIntensity = 0.35f;
    public float bloomIntensity = 0.35f;
    public float chromaticAberration = 0.12f;

    private TelloSimulator sim;
    private Camera cam;
    private UniversalAdditionalCameraData camData;
    private Light sun;
    private Light flashlight;
    private Light fill;
    private Light previewLight;
    private Volume volume;
    private VolumeProfile profile;
    private ColorAdjustments colorAdjust;
    private HorrorAudio audioRig;
    private bool nightVision;

    // Original scene state, captured before the first Apply(true) so L can
    // restore the daylight look for calibration / voxel-map work.
    private bool origFog;
    private FogMode origFogMode;
    private Color origFogColor;
    private float origFogDensity;
    private AmbientMode origAmbientMode;
    private Color origAmbientLight;
    private Material origSkybox;
    private float origSunIntensity;
    private Color origSunColor;
    private CameraClearFlags origClearFlags;
    private Color origBackgroundColor;
    private bool origPostProcessing;
    private AntialiasingMode origAntialiasing;
    private bool captured;

    // Spawns the rig when the scene does not carry one, so the horror look is
    // live with zero Editor setup.
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void Bootstrap()
    {
        if (FindFirstObjectByType<HorrorAtmosphere>() != null) return;
        GameObject go = new GameObject("HorrorAtmosphere");
        go.layer = 0;   // Default — the Main Camera's volume layer mask only includes it
        go.AddComponent<HorrorAtmosphere>();
    }

    void Start()
    {
        sim = FindFirstObjectByType<TelloSimulator>();
        cam = Camera.main;
        if (cam != null)
        {
            camData = cam.GetUniversalAdditionalCameraData();
        }
        else
        {
            Debug.LogWarning("[Horror] No camera tagged MainCamera — flashlight and " +
                             "post-processing are skipped, fog and lights still apply.");
        }
        sun = FindSun();

        CaptureOriginal();
        BuildLights();
        BuildVolume();

        audioRig = GetComponent<HorrorAudio>();
        if (audioRig == null)
        {
            audioRig = gameObject.AddComponent<HorrorAudio>();
        }

        Apply(horrorEnabled);
        // CamcorderHUD may have flipped night shot on before this Start ran, in
        // which case the volume did not exist yet to receive the green cast.
        SetNightVision(nightVision);
    }

    void Update()
    {
        if (KeyPressed(toggleKey))
        {
            horrorEnabled = !horrorEnabled;
            Apply(horrorEnabled);
            Debug.Log($"[Horror] mode {(horrorEnabled ? "ON" : "OFF")}");
        }

        if (KeyPressed(flashlightKey))
        {
            flashlightOn = !flashlightOn;
            if (flashlight != null) flashlight.enabled = horrorEnabled && flashlightOn;
        }

        int step = BrightnessStepPressed();
        if (step != 0 && horrorEnabled)
        {
            brightness = Mathf.Clamp(brightness + step * brightnessStep, 0.25f, 6f);
            ApplyBrightness();
            Debug.Log($"[Horror] brightness = {brightness:F2}");
        }

        UpdatePreviewLight();
    }

    // The Directional Light already in test.unity — dimmed rather than deleted so
    // the scene still has a faint key light and Apply(false) can restore it.
    Light FindSun()
    {
        if (RenderSettings.sun != null) return RenderSettings.sun;
        Light[] lights = FindObjectsByType<Light>(FindObjectsSortMode.None);
        foreach (Light l in lights)
        {
            if (l.type == LightType.Directional) return l;
        }
        return null;
    }

    void CaptureOriginal()
    {
        if (captured) return;
        origFog = RenderSettings.fog;
        origFogMode = RenderSettings.fogMode;
        origFogColor = RenderSettings.fogColor;
        origFogDensity = RenderSettings.fogDensity;
        origAmbientMode = RenderSettings.ambientMode;
        origAmbientLight = RenderSettings.ambientLight;
        origSkybox = RenderSettings.skybox;
        if (sun != null)
        {
            origSunIntensity = sun.intensity;
            origSunColor = sun.color;
        }
        if (cam != null)
        {
            origClearFlags = cam.clearFlags;
            origBackgroundColor = cam.backgroundColor;
        }
        if (camData != null)
        {
            origPostProcessing = camData.renderPostProcessing;
            origAntialiasing = camData.antialiasing;
        }
        captured = true;
    }

    void BuildLights()
    {
        Transform mount = attachToCamera && cam != null ? cam.transform
                        : (sim != null ? sim.transform : null);
        if (mount != null)
        {
            GameObject go = new GameObject("Flashlight");
            go.transform.SetParent(mount, false);
            go.transform.localPosition = Vector3.zero;
            go.transform.localRotation = Quaternion.identity;
            flashlight = go.AddComponent<Light>();
            flashlight.type = LightType.Spot;
            flashlight.range = flashlightRange;
            flashlight.spotAngle = flashlightAngle;
            flashlight.innerSpotAngle = flashlightAngle * 0.4f;
            flashlight.intensity = flashlightIntensity;
            flashlight.color = flashlightColor;
            flashlight.shadows = LightShadows.Soft;
            flashlight.enabled = false;
        }

        if (sim != null)
        {
            GameObject fgo = new GameObject("FillLight");
            fgo.transform.SetParent(sim.transform, false);
            fill = fgo.AddComponent<Light>();
            fill.type = LightType.Point;
            fill.range = fillRange;
            fill.intensity = fillIntensity;
            fill.color = fillColor;
            fill.shadows = LightShadows.None;
            fill.enabled = false;
        }

        GameObject pgo = new GameObject("PreviewLight");
        pgo.transform.SetParent(transform, false);
        previewLight = pgo.AddComponent<Light>();
        previewLight.type = LightType.Point;
        previewLight.range = previewLightRange;
        previewLight.intensity = previewLightIntensity;
        previewLight.color = previewLightColor;
        previewLight.shadows = LightShadows.None;
        previewLight.enabled = false;
    }

    // Built in code instead of a .asset so the settings live in a readable diff
    // and there is no profile GUID to keep in sync with the scene.
    void BuildVolume()
    {
        profile = ScriptableObject.CreateInstance<VolumeProfile>();
        profile.name = "HorrorProfile";

        Tonemapping tm = profile.Add<Tonemapping>(true);
        tm.mode.Override(TonemappingMode.Neutral);

        colorAdjust = profile.Add<ColorAdjustments>(true);
        colorAdjust.postExposure.Override(postExposure);
        colorAdjust.saturation.Override(saturation);
        colorAdjust.contrast.Override(contrast);
        colorAdjust.colorFilter.Override(new Color(0.85f, 0.9f, 1f, 1f));   // cold cast

        ShadowsMidtonesHighlights smh = profile.Add<ShadowsMidtonesHighlights>(true);
        smh.shadows.Override(new Vector4(0.85f, 0.92f, 1.15f, 0f)); // blue shadows
        smh.highlights.Override(new Vector4(1f, 0.97f, 0.92f, 0f));

        Vignette vg = profile.Add<Vignette>(true);
        vg.color.Override(Color.black);
        vg.intensity.Override(vignetteIntensity);
        vg.smoothness.Override(0.4f);

        FilmGrain fg = profile.Add<FilmGrain>(true);
        fg.type.Override(FilmGrainLookup.Medium1);
        fg.intensity.Override(filmGrainIntensity);
        fg.response.Override(0.8f);

        Bloom bl = profile.Add<Bloom>(true);
        bl.threshold.Override(0.9f);
        bl.intensity.Override(bloomIntensity);
        bl.scatter.Override(0.7f);

        ChromaticAberration cab = profile.Add<ChromaticAberration>(true);
        cab.intensity.Override(chromaticAberration);

        // Own child on the Default layer: the Main Camera's volume layer mask only
        // includes Default, so hanging this on a user-placed object of another
        // layer would silently render the whole profile inert.
        GameObject vgo = new GameObject("HorrorVolume");
        vgo.layer = 0;
        vgo.transform.SetParent(transform, false);

        volume = vgo.AddComponent<Volume>();
        volume.isGlobal = true;
        volume.priority = 10f;
        volume.weight = 1f;
        volume.sharedProfile = profile;
        volume.enabled = false;
    }

    void Apply(bool on)
    {
        if (on)
        {
            RenderSettings.fog = true;
            RenderSettings.fogMode = FogMode.ExponentialSquared;
            RenderSettings.fogColor = fogColor;
            RenderSettings.fogDensity = fogDensity;
            RenderSettings.ambientMode = AmbientMode.Flat;
            RenderSettings.skybox = null;
            if (sun != null)
            {
                sun.color = moonColor;
            }
            ApplyBrightness();
            if (cam != null)
            {
                cam.clearFlags = CameraClearFlags.SolidColor;
                cam.backgroundColor = fogColor;
            }
            if (camData != null && postProcessing)
            {
                camData.renderPostProcessing = true;
                camData.antialiasing = AntialiasingMode.FastApproximateAntialiasing;
            }
            if (volume != null) volume.enabled = postProcessing;
            if (flashlight != null) flashlight.enabled = flashlightOn;
            if (fill != null) fill.enabled = fillLight;
        }
        else
        {
            RenderSettings.fog = origFog;
            RenderSettings.fogMode = origFogMode;
            RenderSettings.fogColor = origFogColor;
            RenderSettings.fogDensity = origFogDensity;
            RenderSettings.ambientMode = origAmbientMode;
            RenderSettings.ambientLight = origAmbientLight;
            RenderSettings.skybox = origSkybox;
            if (sun != null)
            {
                sun.intensity = origSunIntensity;
                sun.color = origSunColor;
            }
            if (cam != null)
            {
                cam.clearFlags = origClearFlags;
                cam.backgroundColor = origBackgroundColor;
            }
            if (camData != null)
            {
                camData.renderPostProcessing = origPostProcessing;
                camData.antialiasing = origAntialiasing;
            }
            if (volume != null) volume.enabled = false;
            if (flashlight != null) flashlight.enabled = false;
            if (fill != null) fill.enabled = false;
            if (previewLight != null) previewLight.enabled = false;
        }

        if (audioRig != null) audioRig.SetActive(on);
    }

    // Everything the `brightness` multiplier touches, in one place so the [ and ]
    // keys can re-apply it live. It scales the lights only — postExposure is left
    // alone so the two knobs stay independent instead of compounding.
    void ApplyBrightness()
    {
        float b = brightness * (nightVision ? nightVisionBoost : 1f);
        RenderSettings.ambientLight = ambientColor * b;
        if (sun != null) sun.intensity = moonIntensity * b;
        if (flashlight != null) flashlight.intensity = flashlightIntensity * b;
        if (fill != null) fill.intensity = fillIntensity * b;
        if (colorAdjust != null) colorAdjust.postExposure.Override(postExposure);
    }

    void UpdatePreviewLight()
    {
        if (previewLight == null) return;
        bool want = horrorEnabled && lightPreviewTarget && sim != null && sim.previewActive;
        if (want)
        {
            previewLight.transform.position = sim.previewTarget;
        }
        if (previewLight.enabled != want)
        {
            previewLight.enabled = want;
        }
    }

    // Same dual-backend guard as CameraFollow.TogglePressed — the project ships
    // the Input System package, so UnityEngine.Input alone throws at runtime.
    bool KeyPressed(KeyCode key)
    {
#if ENABLE_INPUT_SYSTEM
        var kb = UnityEngine.InputSystem.Keyboard.current;
        if (kb != null)
        {
            if (key == KeyCode.L && kb.lKey.wasPressedThisFrame) return true;
            if (key == KeyCode.F && kb.fKey.wasPressedThisFrame) return true;
        }
#endif
#if ENABLE_LEGACY_INPUT_MANAGER
        if (Input.GetKeyDown(key)) return true;
#endif
        return false;
    }

    // -1 = dimmer ([), +1 = brighter (]), 0 = no input this frame.
    int BrightnessStepPressed()
    {
#if ENABLE_INPUT_SYSTEM
        var kb = UnityEngine.InputSystem.Keyboard.current;
        if (kb != null)
        {
            if (kb.leftBracketKey.wasPressedThisFrame) return -1;
            if (kb.rightBracketKey.wasPressedThisFrame) return 1;
        }
#endif
#if ENABLE_LEGACY_INPUT_MANAGER
        if (Input.GetKeyDown(KeyCode.LeftBracket)) return -1;
        if (Input.GetKeyDown(KeyCode.RightBracket)) return 1;
#endif
        return 0;
    }

    // Night-shot look, driven by CamcorderHUD's N key: green IR cast plus a light
    // boost, because a camcorder's IR mode sees further than the naked eye.
    public void SetNightVision(bool on)
    {
        nightVision = on;
        if (colorAdjust != null)
        {
            colorAdjust.colorFilter.Override(on ? new Color(0.55f, 1f, 0.62f, 1f)
                                                : new Color(0.85f, 0.9f, 1f, 1f));
            colorAdjust.saturation.Override(on ? -75f : saturation);
        }
        if (flashlight != null)
        {
            flashlight.color = on ? new Color(0.75f, 1f, 0.8f, 1f) : flashlightColor;
        }
        ApplyBrightness();
    }

    void OnDestroy()
    {
        if (profile != null) Destroy(profile);
    }
}
