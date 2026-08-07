using UnityEngine;
using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;

// Person detection for the patrol scan: capture the drone's own camera, ask
// person_detector_tcp.py (YOLO, TCP 9100) about the frame, and report hits back
// to the pipeline through TelloSimulator.ReportDetection.
//
// TelloSimulator owns the sweep — it answers the `scan` verb, spins, and sends
// scan_started/scan_done. This component only rides along: it captures while
// `scanActive` is set and never rotates the drone itself. Two rotation drivers
// on one transform would fight, and the pipeline counts degrees off the
// simulator's own scanTurnedDeg.
//
// The detector process is optional. With nothing listening on 9100 every
// request times out, the sweep still completes, and the room simply reports no
// detections — the same as an empty room.
public class PatrolPersonDetection : MonoBehaviour
{
    private const string SavedLogPrefix = "__PATROL_SAVED_LOG__\n";

    [Header("Detector TCP")]
    [Tooltip("Where person_detector_tcp.py listens. Same PC as Unity by default.")]
    public string detectorHost = "127.0.0.1";
    public int detectorPort = 9100;
    public int connectTimeoutMs = 500;
    public int responseTimeoutMs = 1500;

    [Header("Capture")]
    [Tooltip("Use the 3rd-person main view camera (Camera.main) for YOLO detection by default.")]
    public bool useMainCameraForDetection = true;
    public Camera captureCamera;
    public bool createOnboardCameraIfMissing = false;
    public Vector3 onboardCameraLocalPosition = new Vector3(0f, 0.05f, 0.18f);
    public Vector3 onboardCameraLocalEuler = Vector3.zero;
    public float onboardCameraFov = 82f;
    public int imageWidth = 640;
    public int imageHeight = 360;
    [Range(1, 100)] public int jpegQuality = 70;

    [Header("Fallback Detection Box Size")]
    [Tooltip("Used when no PersonTarget could be resolved: box this tall/wide at " +
             "the point the ray hit instead.")]
    public float fallbackPersonHeight = 1.8f;
    public float fallbackPersonWidth = 0.7f;

    [Header("Onboard Flashlight")]
    [Tooltip("The house is lit for horror, which is far too dark for YOLO. This " +
             "spot light rides the capture camera so the frames we send are " +
             "readable; HorrorAtmosphere's flashlight is a separate, keyboard-only one.")]
    public bool createOnboardFlashlight = true;
    public bool onboardFlashlightOn = true;
    public float onboardFlashlightIntensity = 45f;
    public float onboardFlashlightRange = 110f;
    public float onboardFlashlightAngle = 90f;
    public Color onboardFlashlightColor = new Color(1f, 0.94f, 0.82f, 1f);

    [Header("Detected Frame Save")]
    [Tooltip("Photos for the patrol report. The path travels in the `detect` " +
             "event as image_path and patrol_mission copies the file into the " +
             "report folder, so it has to be readable by the Python side.")]
    public bool saveNewPersonFrames = true;
    [Tooltip("Relative to Application.dataPath (…/simulator/tello_simulator/Assets).")]
    public string saveDirectory = "../../bridge/detection_frames";
    [Tooltip("Fallback dedup when no PersonTarget can be raycast behind the box: " +
             "two hits closer than this in drone yaw count as the same person.")]
    public float minSavedFrameAngularSeparationDeg = 60f;

    [Header("Detection")]
    [Tooltip("Seconds between capture attempts. Only one request is in flight " +
             "at a time, so a slow detector just thins the frames out.")]
    public float frameInterval = 0.35f;
    public float detectionConfidenceThreshold = 0.45f;
    [Tooltip("How long the drone freezes when it finds someone. The sweep " +
             "resumes from the same angle — paused seconds are not counted.")]
    public float pauseOnDetectionSeconds = 2.0f;
    public float detectionAlertCooldownSeconds = 2.0f;
    [Tooltip("Person-free responses needed before the same person can alert " +
             "again. Without it one person fills the room's event budget.")]
    public int noPersonResponsesToRearm = 2;
    [Tooltip("Stop capturing this close to the end of the sweep. Inside the " +
             "guard an outstanding request holds the drone instead, so a hit " +
             "on the last frame still reports before scan_done goes out.")]
    public float endScanGuardDeg = 8f;

    private TelloSimulator tello;
    private Light onboardFlashlight;
    private RenderTexture renderTexture;
    private Texture2D captureTexture;
    private bool requestInFlight;
    private bool wasScanning;
    private Task<DetectionResponse> pendingDetectionTask;
    private byte[] pendingDetectionJpg;
    private float pendingDetectionYaw;
    private float nextFrameTime;
    private float nextDetectionAlertTime;
    private bool personDetectionLatched;
    private int consecutiveNoPersonResponses;
    private readonly List<float> savedDetectionAngles = new List<float>();
    private readonly HashSet<string> savedPersonIds = new HashSet<string>();

    [Serializable]
    private class DetectionResponse
    {
        public bool ok;
        public bool person_detected;
        public float best_confidence;
        public DetectionBox[] detections;
        public string error;
    }

    [Serializable]
    private class DetectionBox
    {
        public string label;
        public float confidence;
        public float x1;
        public float y1;
        public float x2;
        public float y2;
        public float cx;
        public float cy;
    }

    // Attach ourselves to the drone when the scene does not carry the component,
    // the same way HorrorAtmosphere bootstraps its rig — the scan path then works
    // with zero Editor setup, and test.unity needs no GUID for us.
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void Bootstrap()
    {
        if (FindFirstObjectByType<PatrolPersonDetection>() != null) return;
        TelloSimulator sim = FindFirstObjectByType<TelloSimulator>();
        if (sim == null) return;
        sim.gameObject.AddComponent<PatrolPersonDetection>();
    }

    private void Awake()
    {
        tello = GetComponent<TelloSimulator>();
        if (tello == null)
        {
            Debug.LogWarning("[PatrolDetection] no TelloSimulator on this object — "
                             + "detection is off (it needs the scan state and ReportDetection).");
            enabled = false;
            return;
        }

        if (captureCamera == null && createOnboardCameraIfMissing)
        {
            captureCamera = FindOrCreateOnboardCamera();
        }
        if (captureCamera != null && createOnboardFlashlight)
        {
            onboardFlashlight = FindOrCreateOnboardFlashlight(captureCamera.transform);
        }
        if (captureCamera != null)
        {
            EnsureCaptureTargets();
        }
    }

    // Only a camera parented to the drone qualifies — a frame from CameraFollow's
    // chase cam would show the drone, not what the drone sees. The display
    // camera is skipped explicitly on top of that: we point the capture camera
    // at a RenderTexture, which would blank the game view if we took that one.
    private Camera FindOrCreateOnboardCamera()
    {
        foreach (Camera child in GetComponentsInChildren<Camera>(true))
        {
            if (child == Camera.main || child.CompareTag("MainCamera"))
            {
                continue;
            }
            return child;
        }

        GameObject cameraObject = new GameObject("DroneDetectionCamera");
        cameraObject.transform.SetParent(transform, false);
        cameraObject.transform.localPosition = onboardCameraLocalPosition;
        cameraObject.transform.localRotation = Quaternion.Euler(onboardCameraLocalEuler);

        Camera cameraComponent = cameraObject.AddComponent<Camera>();
        cameraComponent.fieldOfView = onboardCameraFov;
        cameraComponent.nearClipPlane = 0.03f;
        cameraComponent.farClipPlane = 80f;
        return cameraComponent;
    }

    private Light FindOrCreateOnboardFlashlight(Transform mount)
    {
        Transform existing = mount.Find("DroneDetectionFlashlight");
        Light lightComponent = existing != null ? existing.GetComponent<Light>() : null;
        if (lightComponent == null)
        {
            GameObject lightObject = new GameObject("DroneDetectionFlashlight");
            lightObject.transform.SetParent(mount, false);
            lightObject.transform.localPosition = Vector3.zero;
            lightObject.transform.localRotation = Quaternion.identity;
            lightComponent = lightObject.AddComponent<Light>();
            lightComponent.type = LightType.Spot;
            lightComponent.shadows = LightShadows.Soft;
        }

        ApplyOnboardFlashlightSettings(lightComponent);
        return lightComponent;
    }

    private void ApplyOnboardFlashlightSettings(Light lightComponent)
    {
        lightComponent.enabled = onboardFlashlightOn;
        lightComponent.type = LightType.Spot;
        lightComponent.intensity = onboardFlashlightIntensity;
        lightComponent.range = onboardFlashlightRange;
        lightComponent.spotAngle = onboardFlashlightAngle;
        lightComponent.innerSpotAngle = onboardFlashlightAngle * 0.4f;
        lightComponent.color = onboardFlashlightColor;
    }

    private void OnDestroy()
    {
        if (captureCamera != null && captureCamera.targetTexture == renderTexture)
        {
            captureCamera.targetTexture = null;
        }
        if (renderTexture != null)
        {
            renderTexture.Release();
            Destroy(renderTexture);
        }
        if (captureTexture != null)
        {
            Destroy(captureTexture);
        }
    }

    private void Update()
    {
        if (tello == null)
        {
            return;
        }
        if (onboardFlashlight != null)
        {
            ApplyOnboardFlashlightSettings(onboardFlashlight);
        }

        bool scanning = tello.scanActive;
        if (scanning && !wasScanning)
        {
            BeginRoomScan();
        }
        wasScanning = scanning;

        ProcessCompletedDetection();

        if (!scanning)
        {
            return;
        }

        // Near the end of the sweep, hold the drone until the outstanding
        // request answers. StopScan sends scan_done and the pipeline leaves the
        // room on it, so a detect that lands after it is thrown away.
        bool inEndGuard = tello.ScanRemainingDeg <= endScanGuardDeg;
        if (inEndGuard)
        {
            if (requestInFlight)
            {
                tello.PauseForDetection(0.2f);
            }
            return;
        }

        if (!requestInFlight && Time.time >= nextFrameTime)
        {
            nextFrameTime = Time.time + frameInterval;
            byte[] jpg = CaptureJpeg();
            if (jpg != null)
            {
                requestInFlight = true;
                pendingDetectionJpg = jpg;
                pendingDetectionYaw = transform.eulerAngles.y;
                pendingDetectionTask = SendFrameAsync(jpg);
            }
        }
    }

    // Dedup state is per room: the same person seen again in the next room's
    // sweep is a new sighting worth reporting.
    private void BeginRoomScan()
    {
        nextFrameTime = 0f;
        nextDetectionAlertTime = 0f;
        personDetectionLatched = false;
        consecutiveNoPersonResponses = 0;
        savedDetectionAngles.Clear();
        savedPersonIds.Clear();
    }

    private void ProcessCompletedDetection()
    {
        if (!requestInFlight || pendingDetectionTask == null || !pendingDetectionTask.IsCompleted)
        {
            return;
        }

        requestInFlight = false;
        DetectionResponse response = null;
        if (pendingDetectionTask.Status == TaskStatus.RanToCompletion)
        {
            response = pendingDetectionTask.Result;
        }

        byte[] jpg = pendingDetectionJpg;
        float yaw = pendingDetectionYaw;
        pendingDetectionTask = null;
        pendingDetectionJpg = null;

        if (response != null && !response.ok && !string.IsNullOrEmpty(response.error))
        {
            Debug.LogWarning($"[PatrolDetection] detector error: {response.error}");
            return;
        }

        bool personAboveThreshold = response != null
            && response.ok
            && response.person_detected
            && response.best_confidence >= detectionConfidenceThreshold;

        if (!personAboveThreshold)
        {
            consecutiveNoPersonResponses += 1;
            if (consecutiveNoPersonResponses >= Mathf.Max(1, noPersonResponsesToRearm))
            {
                personDetectionLatched = false;
            }
            return;
        }

        consecutiveNoPersonResponses = 0;
        if (personDetectionLatched || Time.time < nextDetectionAlertTime)
        {
            return;
        }

        DetectionBox best = BestBox(response);
        if (best == null)
        {
            return;
        }

        personDetectionLatched = true;
        nextDetectionAlertTime = Time.time + detectionAlertCooldownSeconds;

        // One raycast pass for duplicate filter and saved-frame bookkeeping.
        List<PersonTarget> detectedTargets = FindDetectedPersonTargets(response);

        string imagePath = saveNewPersonFrames && jpg != null
            ? SaveDetectedFrame(jpg, yaw, response, detectedTargets)
            : "";

        Debug.Log($"[PatrolDetection] PERSON detected confidence={response.best_confidence:F2} "
                  + $"- pause {pauseOnDetectionSeconds:F1}s");

        // Percent of the frame, not pixels — that is what the web console draws
        // in, so nothing between here and the browser rescales.
        float w = Mathf.Max(1f, imageWidth);
        float h = Mathf.Max(1f, imageHeight);
        tello.ReportDetection(
            string.IsNullOrEmpty(best.label) ? "person" : best.label,
            best.confidence,
            best.x1 / w * 100f,
            best.y1 / h * 100f,
            (best.x2 - best.x1) / w * 100f,
            (best.y2 - best.y1) / h * 100f,
            imagePath);

        tello.PauseForDetection(pauseOnDetectionSeconds);
    }

    private DetectionBox BestBox(DetectionResponse response)
    {
        if (response == null || response.detections == null)
        {
            return null;
        }
        DetectionBox best = null;
        foreach (DetectionBox detection in response.detections)
        {
            if (detection == null || detection.confidence < detectionConfidenceThreshold)
            {
                continue;
            }
            if (best == null || detection.confidence > best.confidence)
            {
                best = detection;
            }
        }
        return best;
    }

    private Camera GetActiveCamera()
    {
        if (useMainCameraForDetection && Camera.main != null)
        {
            return Camera.main;
        }
        return captureCamera != null ? captureCamera : Camera.main;
    }

    // Returns the path to hand the pipeline, or "" when nothing was written
    // (duplicate sighting, or the folder is not writable).
    private string SaveDetectedFrame(byte[] jpg, float yaw, DetectionResponse response,
                                     List<PersonTarget> detectedTargets)
    {
        bool hasResolvedTarget = detectedTargets.Count > 0;
        bool hasNewResolvedTarget = false;
        foreach (PersonTarget target in detectedTargets)
        {
            if (target != null && !savedPersonIds.Contains(target.PersonId))
            {
                hasNewResolvedTarget = true;
                break;
            }
        }

        // A raycast through the box gives us the actual person, which beats any
        // angle heuristic. The yaw fallback is for people the ray misses (no
        // collider, occluded, detector false positive).
        bool duplicateResolvedTarget = hasResolvedTarget && !hasNewResolvedTarget;
        bool duplicateUnresolvedTarget = !hasResolvedTarget && IsDuplicateSavedAngle(yaw);
        if (duplicateResolvedTarget || duplicateUnresolvedTarget)
        {
            Debug.Log($"[PatrolDetection] skipped duplicate person frame yaw={yaw:F1} "
                      + (hasResolvedTarget ? "(same PersonTarget)" : "(angle fallback)"));
            return "";
        }

        try
        {
            string root = Path.GetFullPath(Path.Combine(Application.dataPath, saveDirectory));
            Directory.CreateDirectory(root);
            string stamp = DateTime.Now.ToString("yyyyMMdd_HHmmss_fff");
            string stem = $"patrol_person_yaw_{Mathf.RoundToInt(NormalizeAngle(yaw)):000}_{stamp}";
            string path = Path.Combine(root, stem + ".jpg");

            savedDetectionAngles.Add(NormalizeAngle(yaw));
            foreach (PersonTarget target in detectedTargets)
            {
                if (target != null)
                {
                    savedPersonIds.Add(target.PersonId);
                }
            }

            string savedLog = BuildSavedDetectionLog(response, yaw);
            Debug.Log(savedLog.TrimEnd());
            _ = SendSavedDetectionLogAsync(savedLog);

            // Save ONLY ONE photo file with the red web-style bounding box annotated on it.
            if (SaveAnnotatedDetectedFrame(jpg, response, path))
            {
                return path;
            }
            File.WriteAllBytes(path, jpg);
            return path;
        }
        catch (Exception e)
        {
            // A photo is a nice-to-have; the detect event still goes out without one.
            Debug.LogWarning($"[PatrolDetection] could not save frame: {e.Message}");
            return "";
        }
    }

    private List<PersonTarget> FindDetectedPersonTargets(DetectionResponse response)
    {
        List<PersonTarget> targets = new List<PersonTarget>();
        if (response == null || response.detections == null)
        {
            return targets;
        }

        foreach (DetectionBox detection in response.detections)
        {
            if (detection == null || detection.confidence < detectionConfidenceThreshold)
            {
                continue;
            }

            PersonTarget target = FindPersonTargetForDetection(detection);
            if (target != null && !targets.Contains(target))
            {
                targets.Add(target);
            }
        }
        return targets;
    }

    private PersonTarget FindPersonTargetForDetection(DetectionBox detection)
    {
        Camera cam = GetActiveCamera();
        if (cam == null || detection == null)
        {
            return null;
        }

        float cx = detection.cx;
        float cy = detection.cy;
        if (Mathf.Approximately(cx, 0f) && Mathf.Approximately(cy, 0f))
        {
            cx = (detection.x1 + detection.x2) * 0.5f;
            cy = (detection.y1 + detection.y2) * 0.5f;
        }

        // The box centre often lands between the legs or on a held object, so
        // probe a small cross around it instead of a single ray.
        float boxWidth = Mathf.Max(1f, detection.x2 - detection.x1);
        float boxHeight = Mathf.Max(1f, detection.y2 - detection.y1);
        Vector2[] samples =
        {
            new Vector2(cx, cy),
            new Vector2(cx - boxWidth * 0.18f, cy),
            new Vector2(cx + boxWidth * 0.18f, cy),
            new Vector2(cx, cy - boxHeight * 0.18f),
            new Vector2(cx, cy + boxHeight * 0.18f)
        };

        PersonTarget closestTarget = null;
        float closestDistance = float.MaxValue;
        foreach (Vector2 sample in samples)
        {
            float viewportX = Mathf.Clamp01(sample.x / Mathf.Max(1f, imageWidth));
            float viewportY = 1f - Mathf.Clamp01(sample.y / Mathf.Max(1f, imageHeight));
            Ray ray = cam.ViewportPointToRay(new Vector3(viewportX, viewportY, 0f));
            RaycastHit[] hits = Physics.RaycastAll(
                ray,
                Mathf.Max(cam.farClipPlane, 100f),
                Physics.AllLayers,
                QueryTriggerInteraction.Collide);

            foreach (RaycastHit hit in hits)
            {
                PersonTarget target = hit.collider != null
                    ? hit.collider.GetComponentInParent<PersonTarget>()
                    : null;
                if (target != null && hit.distance < closestDistance)
                {
                    closestTarget = target;
                    closestDistance = hit.distance;
                }
            }
        }
        return closestTarget;
    }

    private bool IsDuplicateSavedAngle(float yaw)
    {
        if (minSavedFrameAngularSeparationDeg <= 0f)
        {
            return false;
        }

        float normalized = NormalizeAngle(yaw);
        foreach (float savedAngle in savedDetectionAngles)
        {
            if (CircularAngleDistance(normalized, savedAngle) < minSavedFrameAngularSeparationDeg)
            {
                return true;
            }
        }
        return false;
    }

    private static float NormalizeAngle(float angle)
    {
        angle %= 360f;
        if (angle < 0f)
        {
            angle += 360f;
        }
        return angle;
    }

    private static float CircularAngleDistance(float a, float b)
    {
        float diff = Mathf.Abs(NormalizeAngle(a) - NormalizeAngle(b));
        return Mathf.Min(diff, 360f - diff);
    }

    private string BuildSavedDetectionLog(DetectionResponse response, float yaw)
    {
        StringBuilder builder = new StringBuilder();
        if (response == null || response.detections == null || response.detections.Length == 0)
        {
            builder.AppendLine($"[PatrolDetection] saved frame yaw={yaw:F1}, but detector returned no boxes");
            return builder.ToString();
        }

        builder.AppendLine($"[PatrolDetection] saved frame yaw={yaw:F1} confidence={response.best_confidence:F2}");
        foreach (DetectionBox detection in response.detections)
        {
            if (detection == null || detection.confidence < detectionConfidenceThreshold)
            {
                continue;
            }

            float cx = detection.cx;
            float cy = detection.cy;
            if (Mathf.Approximately(cx, 0f) && Mathf.Approximately(cy, 0f))
            {
                cx = (detection.x1 + detection.x2) * 0.5f;
                cy = (detection.y1 + detection.y2) * 0.5f;
            }

            builder.AppendLine(
                $"[PatrolDetection] bbox=({detection.x1:F1}, {detection.y1:F1}, {detection.x2:F1}, {detection.y2:F1}) "
                + $"center=({cx:F1}, {cy:F1}) confidence={detection.confidence:F2}");
        }

        return builder.ToString();
    }

    private bool SaveAnnotatedDetectedFrame(byte[] jpg, DetectionResponse response, string path)
    {
        if (response == null || response.detections == null || response.detections.Length == 0)
        {
            return false;
        }

        Texture2D texture = new Texture2D(2, 2, TextureFormat.RGB24, false);
        try
        {
            if (!texture.LoadImage(jpg))
            {
                return false;
            }

            foreach (DetectionBox detection in response.detections)
            {
                if (detection == null || detection.confidence < detectionConfidenceThreshold)
                {
                    continue;
                }

                DrawDetectionOverlay(texture, detection);
            }

            texture.Apply();
            File.WriteAllBytes(path, texture.EncodeToJPG(jpegQuality));
            return true;
        }
        finally
        {
            Destroy(texture);
        }
    }

    private void DrawDetectionOverlay(Texture2D texture, DetectionBox detection)
    {
        int x1 = Mathf.RoundToInt(detection.x1);
        int y1 = Mathf.RoundToInt(detection.y1);
        int x2 = Mathf.RoundToInt(detection.x2);
        int y2 = Mathf.RoundToInt(detection.y2);

        Color redBorder = new Color(0.937f, 0.267f, 0.267f, 1f);   // #ef4444
        Color lightCorner = new Color(0.996f, 0.792f, 0.792f, 1f); // #fecaca
        Color badgeBg = new Color(0.271f, 0.039f, 0.039f, 1f);     // #450a0a

        int borderWidth = 3;
        int cornerLength = Mathf.Clamp(Mathf.Min(x2 - x1, y2 - y1) / 4, 12, 32);
        int cornerWidth = 4;

        // 1. Red Border (#ef4444)
        for (int offset = 0; offset < borderWidth; offset++)
        {
            DrawHorizontalLineTopLeft(texture, x1, x2, y1 + offset, redBorder);
            DrawHorizontalLineTopLeft(texture, x1, x2, y2 - offset, redBorder);
            DrawVerticalLineTopLeft(texture, x1 + offset, y1, y2, redBorder);
            DrawVerticalLineTopLeft(texture, x2 - offset, y1, y2, redBorder);
        }

        // 2. 4 Corner Bracket accents (#fecaca)
        for (int offset = -1; offset < cornerWidth - 1; offset++)
        {
            // Top-Left corner bracket
            DrawHorizontalLineTopLeft(texture, x1 - 1, x1 + cornerLength, y1 + offset, lightCorner);
            DrawVerticalLineTopLeft(texture, x1 + offset, y1 - 1, y1 + cornerLength, lightCorner);
            // Top-Right corner bracket
            DrawHorizontalLineTopLeft(texture, x2 - cornerLength, x2 + 1, y1 + offset, lightCorner);
            DrawVerticalLineTopLeft(texture, x2 - offset, y1 - 1, y1 + cornerLength, lightCorner);
            // Bottom-Left corner bracket
            DrawHorizontalLineTopLeft(texture, x1 - 1, x1 + cornerLength, y2 - offset, lightCorner);
            DrawVerticalLineTopLeft(texture, x1 + offset, y2 - cornerLength, y2 + 1, lightCorner);
            // Bottom-Right corner bracket
            DrawHorizontalLineTopLeft(texture, x2 - cornerLength, x2 + 1, y2 - offset, lightCorner);
            DrawVerticalLineTopLeft(texture, x2 - offset, y2 - cornerLength, y2 + 1, lightCorner);
        }

        // 3. Dark Red Badge header block (#450a0a with #ef4444 border)
        int badgeH = 20;
        int badgeW = 120;
        int badgeX1 = x1;
        int badgeY1 = Mathf.Max(0, y1 - badgeH - 4);
        int badgeX2 = Mathf.Min(texture.width - 1, badgeX1 + badgeW);
        int badgeY2 = badgeY1 + badgeH;

        // Badge background fill (#450a0a)
        FillRectTopLeft(texture, badgeX1, badgeY1, badgeX2, badgeY2, badgeBg);
        // Badge border (#ef4444)
        for (int offset = 0; offset < 2; offset++)
        {
            DrawHorizontalLineTopLeft(texture, badgeX1, badgeX2, badgeY1 + offset, redBorder);
            DrawHorizontalLineTopLeft(texture, badgeX1, badgeX2, badgeY2 - offset, redBorder);
            DrawVerticalLineTopLeft(texture, badgeX1 + offset, badgeY1, badgeY2, redBorder);
            DrawVerticalLineTopLeft(texture, badgeX2 - offset, badgeY1, badgeY2, redBorder);
        }
    }

    private void FillRectTopLeft(Texture2D texture, int xStart, int yStart, int xEnd, int yEnd, Color color)
    {
        int minX = Mathf.Clamp(Mathf.Min(xStart, xEnd), 0, texture.width - 1);
        int maxX = Mathf.Clamp(Mathf.Max(xStart, xEnd), 0, texture.width - 1);
        int minY = Mathf.Clamp(Mathf.Min(yStart, yEnd), 0, texture.height - 1);
        int maxY = Mathf.Clamp(Mathf.Max(yStart, yEnd), 0, texture.height - 1);

        for (int y = minY; y <= maxY; y++)
        {
            int texY = texture.height - 1 - y;
            for (int x = minX; x <= maxX; x++)
            {
                texture.SetPixel(x, texY, color);
            }
        }
    }

    // YOLO boxes are top-left origin, Texture2D pixels are bottom-left — hence
    // the flip in both helpers.
    private void DrawHorizontalLineTopLeft(Texture2D texture, int xStart, int xEnd, int yTopLeft, Color color)
    {
        if (yTopLeft < 0 || yTopLeft >= texture.height)
        {
            return;
        }

        int minX = Mathf.Clamp(Mathf.Min(xStart, xEnd), 0, texture.width - 1);
        int maxX = Mathf.Clamp(Mathf.Max(xStart, xEnd), 0, texture.width - 1);
        int textureY = texture.height - 1 - yTopLeft;
        for (int x = minX; x <= maxX; x++)
        {
            texture.SetPixel(x, textureY, color);
        }
    }

    private void DrawVerticalLineTopLeft(Texture2D texture, int xTopLeft, int yStart, int yEnd, Color color)
    {
        if (xTopLeft < 0 || xTopLeft >= texture.width)
        {
            return;
        }

        int minY = Mathf.Clamp(Mathf.Min(yStart, yEnd), 0, texture.height - 1);
        int maxY = Mathf.Clamp(Mathf.Max(yStart, yEnd), 0, texture.height - 1);
        for (int y = minY; y <= maxY; y++)
        {
            texture.SetPixel(xTopLeft, texture.height - 1 - y, color);
        }
    }

    private byte[] CaptureJpeg()
    {
        if (GetActiveCamera() == null)
        {
            return null;
        }

        ReadCurrentCameraTexture();
        if (captureTexture == null)
        {
            return null;
        }

        return captureTexture.EncodeToJPG(jpegQuality);
    }

    private void EnsureCaptureTargets()
    {
        if (renderTexture == null || renderTexture.width != imageWidth || renderTexture.height != imageHeight)
        {
            if (renderTexture != null)
            {
                renderTexture.Release();
                Destroy(renderTexture);
            }
            if (captureTexture != null)
            {
                Destroy(captureTexture);
            }
            renderTexture = new RenderTexture(imageWidth, imageHeight, 24, RenderTextureFormat.ARGB32);
            captureTexture = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
        }
    }

    private void ReadCurrentCameraTexture()
    {
        EnsureCaptureTargets();
        Camera cam = GetActiveCamera();
        if (cam == null || renderTexture == null || captureTexture == null)
        {
            return;
        }

        RenderTexture prevTarget = cam.targetTexture;
        RenderTexture prevActive = RenderTexture.active;

        cam.targetTexture = renderTexture;
        cam.Render();
        RenderTexture.active = renderTexture;
        captureTexture.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        captureTexture.Apply(false);

        cam.targetTexture = prevTarget;
        RenderTexture.active = prevActive;
    }

    // 4-byte big-endian length + JPEG out, the same framing back with JSON.
    // One connection per frame keeps a stalled detector from wedging the next
    // request; the cost is a TCP handshake we can afford at ~3 frames/s.
    private async Task<DetectionResponse> SendFrameAsync(byte[] jpg)
    {
        try
        {
            using (TcpClient client = new TcpClient())
            {
                Task connect = client.ConnectAsync(detectorHost, detectorPort);
                if (await Task.WhenAny(connect, Task.Delay(connectTimeoutMs)) != connect)
                {
                    return Error("connect timeout");
                }

                client.ReceiveTimeout = responseTimeoutMs;
                client.SendTimeout = responseTimeoutMs;
                using (NetworkStream stream = client.GetStream())
                {
                    await WriteFrame(stream, jpg);
                    byte[] responseBytes = await ReadFrame(stream, responseTimeoutMs);
                    if (responseBytes == null)
                    {
                        return Error("response timeout");
                    }
                    string json = Encoding.UTF8.GetString(responseBytes);
                    return JsonUtility.FromJson<DetectionResponse>(json);
                }
            }
        }
        catch (Exception e)
        {
            return Error(e.Message);
        }
    }

    // Mirror the saved-frame log into the detector's terminal, which is where
    // whoever is running the YOLO service is looking.
    private async Task SendSavedDetectionLogAsync(string savedLog)
    {
        if (string.IsNullOrEmpty(savedLog))
        {
            return;
        }

        try
        {
            using (TcpClient client = new TcpClient())
            {
                Task connect = client.ConnectAsync(detectorHost, detectorPort);
                if (await Task.WhenAny(connect, Task.Delay(connectTimeoutMs)) != connect)
                {
                    return;
                }

                client.ReceiveTimeout = responseTimeoutMs;
                client.SendTimeout = responseTimeoutMs;
                using (NetworkStream stream = client.GetStream())
                {
                    byte[] payload = Encoding.UTF8.GetBytes(SavedLogPrefix + savedLog);
                    await WriteFrame(stream, payload);
                    await ReadFrame(stream, responseTimeoutMs);
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[PatrolDetection] saved log notify failed: {e.Message}");
        }
    }

    private static async Task WriteFrame(NetworkStream stream, byte[] payload)
    {
        int len = System.Net.IPAddress.HostToNetworkOrder(payload.Length);
        byte[] header = BitConverter.GetBytes(len);
        await stream.WriteAsync(header, 0, header.Length);
        await stream.WriteAsync(payload, 0, payload.Length);
    }

    private static async Task<byte[]> ReadFrame(NetworkStream stream, int timeoutMs)
    {
        byte[] header = await ReadExact(stream, 4, timeoutMs);
        if (header == null)
        {
            return null;
        }
        int len = System.Net.IPAddress.NetworkToHostOrder(BitConverter.ToInt32(header, 0));
        if (len <= 0 || len > 1024 * 1024)
        {
            return null;
        }
        return await ReadExact(stream, len, timeoutMs);
    }

    private static async Task<byte[]> ReadExact(NetworkStream stream, int count, int timeoutMs)
    {
        byte[] buffer = new byte[count];
        int offset = 0;
        while (offset < count)
        {
            Task<int> readTask = stream.ReadAsync(buffer, offset, count - offset);
            if (await Task.WhenAny(readTask, Task.Delay(timeoutMs)) != readTask)
            {
                return null;
            }
            int n = readTask.Result;
            if (n <= 0)
            {
                return null;
            }
            offset += n;
        }
        return buffer;
    }

    private static DetectionResponse Error(string message)
    {
        return new DetectionResponse
        {
            ok = false,
            person_detected = false,
            best_confidence = 0f,
            detections = new DetectionBox[0],
            error = message
        };
    }
}
