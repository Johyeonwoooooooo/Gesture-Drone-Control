using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;

public class PatrolPersonDetection : MonoBehaviour
{
    private const float EndScanCaptureGuardDegrees = 2.0f;
    private const string SavedLogPrefix = "__PATROL_SAVED_LOG__\n";

    [Header("Detector TCP")]
    public string detectorHost = "127.0.0.1";
    public int detectorPort = 9100;
    public int connectTimeoutMs = 500;
    public int responseTimeoutMs = 1500;

    [Header("Capture")]
    public Camera captureCamera;
    public bool createOnboardCameraIfMissing = true;
    public Vector3 onboardCameraLocalPosition = new Vector3(0f, 0.05f, 0.18f);
    public Vector3 onboardCameraLocalEuler = Vector3.zero;
    public float onboardCameraFov = 82f;
    public int imageWidth = 640;
    public int imageHeight = 360;
    [Range(1, 100)] public int jpegQuality = 70;

    [Header("On-screen Preview")]
    public bool showOnboardPreview = true;
    public Rect previewRect = new Rect(16f, 16f, 320f, 180f);
    public bool previewOnlyDuringScan = false;

    [Header("Onboard Flashlight")]
    public bool createOnboardFlashlight = true;
    public bool onboardFlashlightOn = true;
    public float onboardFlashlightIntensity = 45f;
    public float onboardFlashlightRange = 110f;
    public float onboardFlashlightAngle = 90f;
    public Color onboardFlashlightColor = new Color(1f, 0.94f, 0.82f, 1f);

    [Header("Detected Frame Save")]
    public bool saveNewPersonFrames = true;
    public string saveDirectory = "../../bridge/detection_frames";
    public float minSavedFrameAngularSeparationDeg = 60f;

    [Header("360 Scan")]
    public bool scanOnStart = false;
    public KeyCode scanKey = KeyCode.P;
    public bool holdDroneWhileScanning = true;
    public float scanYawSpeedDegPerSec = 60f;
    public float frameInterval = 0.35f;
    public float detectionConfidenceThreshold = 0.45f;
    public float pauseOnDetectionSeconds = 2.0f;
    public float detectionAlertCooldownSeconds = 2.0f;
    public int noPersonResponsesToRearm = 2;
    public float scanCooldownSeconds = 5.0f;
    public float waitForLastDetectionSeconds = 3.0f;

    private TelloSimulator tello;
    private Light onboardFlashlight;
    private RenderTexture renderTexture;
    private Texture2D captureTexture;
    private bool scanRunning;
    private bool requestInFlight;
    private Task<DetectionResponse> pendingDetectionTask;
    private byte[] pendingDetectionJpg;
    private float pendingDetectionAngle;
    private float nextScanAllowedTime;
    private float pauseRotationUntil;
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

    private void Awake()
    {
        tello = GetComponent<TelloSimulator>();
        if (captureCamera == null)
        {
            captureCamera = FindOnboardCamera();
        }
        if (captureCamera == null && createOnboardCameraIfMissing)
        {
            captureCamera = CreateOnboardCamera();
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

    private Camera FindOnboardCamera()
    {
        Camera[] childCameras = GetComponentsInChildren<Camera>(true);
        if (childCameras.Length > 0)
        {
            return childCameras[0];
        }
        return null;
    }

    private Camera CreateOnboardCamera()
    {
        GameObject cameraObject = new GameObject("DroneDetectionCamera");
        cameraObject.transform.SetParent(transform, false);
        cameraObject.transform.localPosition = onboardCameraLocalPosition;
        cameraObject.transform.localRotation = Quaternion.Euler(onboardCameraLocalEuler);

        Camera cameraComponent = cameraObject.AddComponent<Camera>();
        cameraComponent.fieldOfView = onboardCameraFov;
        cameraComponent.nearClipPlane = 0.03f;
        cameraComponent.farClipPlane = 80f;
        cameraComponent.enabled = true;
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

    private void OnGUI()
    {
        if (!showOnboardPreview || captureCamera == null)
        {
            return;
        }
        if (previewOnlyDuringScan && !scanRunning)
        {
            return;
        }

        EnsureCaptureTargets();
        if (renderTexture != null)
        {
            GUI.DrawTexture(previewRect, renderTexture, ScaleMode.ScaleToFit, false);
            GUI.Box(previewRect, GUIContent.none);
        }
    }

    private void Update()
    {
        if (onboardFlashlight != null)
        {
            ApplyOnboardFlashlightSettings(onboardFlashlight);
        }

        if (ScanKeyPressed())
        {
            StartScan();
        }

        if (scanOnStart && !scanRunning && Time.time >= nextScanAllowedTime)
        {
            StartScan();
        }
    }

    private bool ScanKeyPressed()
    {
#if ENABLE_INPUT_SYSTEM
        var kb = UnityEngine.InputSystem.Keyboard.current;
        if (kb != null)
        {
            switch (scanKey)
            {
                case KeyCode.P:
                    return kb.pKey.wasPressedThisFrame;
                case KeyCode.Space:
                    return kb.spaceKey.wasPressedThisFrame;
                case KeyCode.Return:
                    return kb.enterKey.wasPressedThisFrame;
                case KeyCode.F:
                    return kb.fKey.wasPressedThisFrame;
                case KeyCode.C:
                    return kb.cKey.wasPressedThisFrame;
                case KeyCode.R:
                    return kb.rKey.wasPressedThisFrame;
                default:
                    break;
            }
        }
#endif
#if ENABLE_LEGACY_INPUT_MANAGER
        if (Input.GetKeyDown(scanKey))
        {
            return true;
        }
#endif
        return false;
    }

    public void StartScan()
    {
        if (scanRunning || Time.time < nextScanAllowedTime)
        {
            return;
        }
        StartCoroutine(ScanRoom360());
    }

    private IEnumerator ScanRoom360()
    {
        scanRunning = true;
        float scanned = 0f;
        float nextFrameTime = 0f;
        float nextDetectionAlertTime = 0f;
        pauseRotationUntil = 0f;
        personDetectionLatched = false;
        consecutiveNoPersonResponses = 0;
        savedDetectionAngles.Clear();
        savedPersonIds.Clear();

        while (scanned < 360f)
        {
            if (holdDroneWhileScanning && tello != null)
            {
                tello.PauseForDetection(0.2f);
            }

            ProcessCompletedDetection(ref nextDetectionAlertTime);

            bool canCaptureFrame = scanned < 360f - EndScanCaptureGuardDegrees;
            if (canCaptureFrame && !requestInFlight && Time.time >= nextFrameTime)
            {
                nextFrameTime = Time.time + frameInterval;
                byte[] jpg = CaptureJpeg();
                if (jpg != null)
                {
                    requestInFlight = true;
                    pendingDetectionJpg = jpg;
                    pendingDetectionAngle = scanned;
                    pendingDetectionTask = SendFrameAsync(jpg);
                }
            }

            if (Time.time >= pauseRotationUntil)
            {
                float deltaYaw = Mathf.Min(scanYawSpeedDegPerSec * Time.deltaTime, 360f - scanned);
                transform.Rotate(Vector3.up, deltaYaw, Space.World);
                scanned += Mathf.Abs(deltaYaw);
            }
            else if (tello != null)
            {
                tello.PauseForDetection(Mathf.Max(0.2f, pauseRotationUntil - Time.time));
            }

            yield return null;
        }

        float waitDeadline = Time.time + waitForLastDetectionSeconds;
        while (requestInFlight && Time.time < waitDeadline)
        {
            ProcessCompletedDetection(ref nextDetectionAlertTime);
            if (Time.time < pauseRotationUntil && tello != null)
            {
                tello.PauseForDetection(Mathf.Max(0.2f, pauseRotationUntil - Time.time));
            }
            yield return null;
        }

        while (Time.time < pauseRotationUntil)
        {
            if (tello != null)
            {
                tello.PauseForDetection(Mathf.Max(0.2f, pauseRotationUntil - Time.time));
            }
            yield return null;
        }

        pendingDetectionTask = null;
        pendingDetectionJpg = null;
        requestInFlight = false;
        nextScanAllowedTime = Time.time + scanCooldownSeconds;
        scanRunning = false;
    }

    private bool ProcessCompletedDetection(ref float nextDetectionAlertTime)
    {
        if (!requestInFlight || pendingDetectionTask == null || !pendingDetectionTask.IsCompleted)
        {
            return false;
        }

        requestInFlight = false;
        DetectionResponse response = null;
        if (pendingDetectionTask.Status == TaskStatus.RanToCompletion)
        {
            response = pendingDetectionTask.Result;
        }

        byte[] jpg = pendingDetectionJpg;
        float scanAngle = pendingDetectionAngle;
        pendingDetectionTask = null;
        pendingDetectionJpg = null;

        if (response != null && !response.ok && !string.IsNullOrEmpty(response.error))
        {
            Debug.LogWarning($"[PatrolDetection] detector error: {response.error}");
            return false;
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
            return false;
        }

        consecutiveNoPersonResponses = 0;
        if (personDetectionLatched)
        {
            return false;
        }

        if (Time.time >= nextDetectionAlertTime)
        {
            personDetectionLatched = true;
            if (saveNewPersonFrames && jpg != null)
            {
                string path = SaveDetectedFrame(jpg, scanAngle, response);
                if (!string.IsNullOrEmpty(path))
                {
                    Debug.Log($"[PatrolDetection] saved detected person frame: {path}");
                }
            }

            nextDetectionAlertTime = Time.time + detectionAlertCooldownSeconds;
            Debug.Log(
                $"[PatrolDetection] PERSON detected confidence={response.best_confidence:F2} " +
                $"- pause {pauseOnDetectionSeconds:F1}s");
            if (tello != null)
            {
                tello.PauseForDetection(pauseOnDetectionSeconds);
            }
            pauseRotationUntil = Mathf.Max(pauseRotationUntil, Time.time + pauseOnDetectionSeconds);
            return true;
        }

        return false;
    }

    private string SaveDetectedFrame(byte[] jpg, float scanAngle, DetectionResponse response)
    {
        List<PersonTarget> detectedTargets = FindDetectedPersonTargets(response);
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

        bool duplicateResolvedTarget = hasResolvedTarget && !hasNewResolvedTarget;
        bool duplicateUnresolvedTarget = !hasResolvedTarget && IsDuplicateSavedAngle(scanAngle);
        if (duplicateResolvedTarget || duplicateUnresolvedTarget)
        {
            Debug.LogWarning(
                $"[PatrolDetection] skipped duplicate person frame angle={scanAngle:F1} " +
                (hasResolvedTarget ? "(same PersonTarget)" : "(angle fallback)"));
            return "";
        }

        string root = Path.GetFullPath(Path.Combine(Application.dataPath, saveDirectory));
        Directory.CreateDirectory(root);
        string stamp = DateTime.Now.ToString("yyyyMMdd_HHmmss_fff");
        string path = Path.Combine(root, $"patrol_person_detected_angle_{scanAngle:000}_{stamp}.jpg");
        File.WriteAllBytes(path, jpg);
        savedDetectionAngles.Add(NormalizeAngle(scanAngle));
        foreach (PersonTarget target in detectedTargets)
        {
            if (target != null)
            {
                savedPersonIds.Add(target.PersonId);
            }
        }
        string savedLog = BuildSavedDetectionLog(response, scanAngle);
        LogSavedDetectionBoxes(savedLog);
        _ = SendSavedDetectionLogAsync(savedLog);

        string annotatedPath = Path.Combine(root, $"patrol_person_detected_angle_{scanAngle:000}_{stamp}_annotated.jpg");
        if (SaveAnnotatedDetectedFrame(jpg, response, annotatedPath))
        {
            Debug.Log($"[PatrolDetection] saved annotated detected person frame: {annotatedPath}");
        }
        return path;
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
        if (captureCamera == null || detection == null)
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
            Ray ray = captureCamera.ViewportPointToRay(new Vector3(viewportX, viewportY, 0f));
            RaycastHit[] hits = Physics.RaycastAll(
                ray,
                Mathf.Max(captureCamera.farClipPlane, 100f),
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

    private bool IsDuplicateSavedAngle(float scanAngle)
    {
        if (minSavedFrameAngularSeparationDeg <= 0f)
        {
            return false;
        }

        float normalized = NormalizeAngle(scanAngle);
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

    private string BuildSavedDetectionLog(DetectionResponse response, float scanAngle)
    {
        StringBuilder builder = new StringBuilder();
        if (response == null)
        {
            builder.AppendLine($"[PatrolDetection] saved frame angle={scanAngle:F1}, but detector response is null");
            return builder.ToString();
        }
        if (response.detections == null || response.detections.Length == 0)
        {
            builder.AppendLine(
                $"[PatrolDetection] saved frame angle={scanAngle:F1} confidence={response.best_confidence:F2}, " +
                "but detector returned no boxes");
            return builder.ToString();
        }

        builder.AppendLine($"[PatrolDetection] saved frame angle={scanAngle:F1} confidence={response.best_confidence:F2}");
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
                $"[PatrolDetection] bbox=({detection.x1:F1}, {detection.y1:F1}, {detection.x2:F1}, {detection.y2:F1}) " +
                $"center=({cx:F1}, {cy:F1}) confidence={detection.confidence:F2}");
        }

        return builder.ToString();
    }

    private void LogSavedDetectionBoxes(string savedLog)
    {
        if (string.IsNullOrEmpty(savedLog))
        {
            return;
        }

        foreach (string line in savedLog.Split(new[] { '\n' }, StringSplitOptions.RemoveEmptyEntries))
        {
            Debug.LogWarning(line);
        }
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
        int cx = Mathf.RoundToInt(detection.cx);
        int cy = Mathf.RoundToInt(detection.cy);

        if (cx == 0 && cy == 0)
        {
            cx = Mathf.RoundToInt((detection.x1 + detection.x2) * 0.5f);
            cy = Mathf.RoundToInt((detection.y1 + detection.y2) * 0.5f);
        }

        Color boxColor = Color.green;
        Color centerColor = Color.red;
        int lineWidth = 4;
        int centerSize = 12;

        for (int offset = 0; offset < lineWidth; offset++)
        {
            DrawHorizontalLineTopLeft(texture, x1, x2, y1 + offset, boxColor);
            DrawHorizontalLineTopLeft(texture, x1, x2, y2 - offset, boxColor);
            DrawVerticalLineTopLeft(texture, x1 + offset, y1, y2, boxColor);
            DrawVerticalLineTopLeft(texture, x2 - offset, y1, y2, boxColor);
        }

        for (int offset = -lineWidth; offset <= lineWidth; offset++)
        {
            DrawHorizontalLineTopLeft(texture, cx - centerSize, cx + centerSize, cy + offset, centerColor);
            DrawVerticalLineTopLeft(texture, cx + offset, cy - centerSize, cy + centerSize, centerColor);
        }
    }

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
        if (captureCamera == null)
        {
            Debug.LogWarning("[PatrolDetection] no capture camera assigned");
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

        if (captureCamera != null)
        {
            captureCamera.enabled = true;
            captureCamera.targetTexture = renderTexture;
        }
    }

    private void ReadCurrentCameraTexture()
    {
        EnsureCaptureTargets();
        if (renderTexture == null || captureTexture == null || captureCamera == null)
        {
            return;
        }

        RenderTexture prevActive = RenderTexture.active;
        RenderTexture.active = renderTexture;
        captureTexture.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        captureTexture.Apply(false);
        RenderTexture.active = prevActive;
    }

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
                    Debug.LogWarning("[PatrolDetection] saved log notify timeout");
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
