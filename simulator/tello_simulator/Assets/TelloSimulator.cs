using UnityEngine;
using System;
using System.Collections.Concurrent;
using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;

public class TelloSimulator : MonoBehaviour
{
    [Header("Network Settings")]
    public int port = 9000;
    public int statePort = 9002;
    public float stateSendHz = 20f;
    public bool autoSendState = true;

    [Header("Movement Settings")]
    public float moveSpeed = 15f;
    public float rotationSpeed = 100f;
    public float smoothTime = 0.05f;

    [Header("Floor Settings")]
    [Tooltip("Clamp the drone so it never drops below this world-space Y value.")]
    public float minHeight = 0.5f;

    [Header("Flight Visualization")]
    [Tooltip("Draw a colored trail behind the drone while it flies.")]
    public bool showFlightTrail = true;
    public Color trailColor = new Color(0.2f, 0.85f, 1f, 0.9f);
    public float trailWidth = 0.3f;
    [Tooltip("Drop a red sphere marker at every recorded collision position.")]
    public bool markCollisions = true;
    public float collisionMarkerSize = 1.2f;

    [Header("Collision Settings")]
    [Tooltip("Radius of the sphere used to detect environment collisions every frame, " +
             "independent of the CharacterController callback. Keep it below minHeight so " +
             "the floor is not constantly counted as a collision. Set to 0 to disable.")]
    public float collisionProbeRadius = 0.3f;
    [Tooltip("Which layers count as environment obstacles for collision detection.")]
    public LayerMask collisionMask = ~0;
    [Tooltip("Minimum seconds between two recorded collision events.")]
    public float collisionDebounce = 0.2f;

    private readonly ConcurrentQueue<string> commandQueue = new ConcurrentQueue<string>();

    private UdpClient udpServer;
    private Thread receiveThread;
    private IPEndPoint lastRemoteEndPoint;

    private float targetLR;
    private float targetFB;
    private float targetUD;
    private float targetYaw;
    private float currentLR;
    private float currentFB;
    private float currentUD;
    private float currentYaw;
    private float velLR;
    private float velFB;
    private float velUD;
    private float velYaw;
    private float nextStateSendTime;
    private float lastCollisionRecordTime = -10f;

    private bool isFlying = false;
    private bool shouldQuit = false;
    private bool hadCollision = false;
    private string lastCommand = "";
    private int collisionCount = 0;
    private string statusMessage = "";
    private float statusMessageTime = -1f;

    // Candidate-preview state (set by the "preview" UDP command). CameraFollow
    // switches to its Preview mode while previewActive; OnGUI shows the
    // [이동]/[다음 후보] buttons whose answers go back over the state channel.
    [NonSerialized] public bool previewActive = false;
    [NonSerialized] public Vector3 previewTarget;
    [NonSerialized] public string previewLabel = "";
    private GameObject previewMarker;

    private CharacterController cc;
    private TrailRenderer trail;
    private readonly System.Collections.Generic.List<GameObject> collisionMarkers =
        new System.Collections.Generic.List<GameObject>();

    // Reused buffer for the per-frame collision probe so it does not allocate.
    private readonly Collider[] overlapResults = new Collider[16];

    void Start()
    {
        Application.runInBackground = true;
        cc = GetComponent<CharacterController>();
        DisableLeftoverPhysics();
        if (showFlightTrail)
        {
            SetupFlightTrail();
        }
        StartUDPServer();
    }

    void SetupFlightTrail()
    {
        GameObject trailGO = new GameObject("FlightTrail");
        trailGO.transform.SetParent(transform, false);
        trail = trailGO.AddComponent<TrailRenderer>();
        trail.time = Mathf.Infinity;              // keep the whole flight visible
        trail.minVertexDistance = 0.15f;
        trail.startWidth = trailWidth;
        trail.endWidth = trailWidth;
        trail.material = new Material(Shader.Find("Sprites/Default"));
        trail.startColor = trailColor;
        trail.endColor = trailColor;
        trail.emitting = false;                   // only while flying
    }

    // The tello model comes from a URDF import that leaves gravity-driven
    // ArticulationBody components on the visual hierarchy. On Play they make the
    // mesh fall out of the CharacterController-driven root, so neutralize them.
    void DisableLeftoverPhysics()
    {
        ArticulationBody[] articulations = GetComponentsInChildren<ArticulationBody>(true);
        // Children before parents, and immediate destruction so the order is respected:
        // deferred Destroy() runs in a batch where the dependency checks can still fail.
        for (int i = articulations.Length - 1; i >= 0; i--)
        {
            ArticulationBody ab = articulations[i];

            // URDF helper scripts (UrdfInertial, UrdfJointFixed, ...) declare
            // RequireComponent(ArticulationBody); Unity refuses to remove the body
            // while they exist, so they have to go first.
            foreach (MonoBehaviour script in ab.GetComponents<MonoBehaviour>())
            {
                if (script != null && script.GetType().Name.StartsWith("Urdf"))
                {
                    DestroyImmediate(script);
                }
            }

            DestroyImmediate(ab);
        }

        foreach (Rigidbody rb in GetComponentsInChildren<Rigidbody>(true))
        {
            rb.isKinematic = true;
            rb.useGravity = false;
        }

        if (articulations.Length > 0)
        {
            Debug.Log($"[Tello] Removed {articulations.Length} leftover ArticulationBody component(s) from the URDF import.");
        }
    }

    void StartUDPServer()
    {
        try
        {
            udpServer = new UdpClient(port);
            udpServer.Client.ReceiveTimeout = 100;

            receiveThread = new Thread(ReceiveLoop);
            receiveThread.IsBackground = true;
            receiveThread.Start();

            Debug.Log($"<color=green>[Tello] UDP server listening on {port}</color>");
        }
        catch (Exception e)
        {
            Debug.LogError($"[Tello] Failed to start UDP server: {e.Message}");
        }
    }

    void ReceiveLoop()
    {
        IPEndPoint remote = new IPEndPoint(IPAddress.Any, 0);

        while (!shouldQuit)
        {
            try
            {
                byte[] data = udpServer.Receive(ref remote);
                lastRemoteEndPoint = remote;

                // UTF-8 and original case are preserved so the "msg" command can
                // carry human-readable (e.g. Korean) status text; keyword matching
                // lowercases a copy in ProcessCommand.
                string msg = Encoding.UTF8.GetString(data).Trim();
                byte[] ok = Encoding.ASCII.GetBytes("ok");
                udpServer.Send(ok, ok.Length, remote);

                commandQueue.Enqueue(msg);
            }
            catch (SocketException)
            {
            }
            catch (Exception e)
            {
                if (!shouldQuit)
                {
                    Debug.LogWarning($"[Tello] Receive error: {e.Message}");
                }
            }
        }
    }

    void Update()
    {
        while (commandQueue.TryDequeue(out string cmd))
        {
            ProcessCommand(cmd);
        }

        if (isFlying)
        {
            currentLR = Mathf.SmoothDamp(currentLR, targetLR, ref velLR, smoothTime);
            currentFB = Mathf.SmoothDamp(currentFB, targetFB, ref velFB, smoothTime);
            currentUD = Mathf.SmoothDamp(currentUD, targetUD, ref velUD, smoothTime);
            currentYaw = Mathf.SmoothDamp(currentYaw, targetYaw, ref velYaw, smoothTime);

            Vector3 localMove = new Vector3(currentLR, currentUD, currentFB) * moveSpeed * Time.deltaTime;
            Vector3 worldMove = transform.TransformDirection(localMove);
            cc.Move(worldMove);

            // Floor guard: lift back to minHeight through the CharacterController so the
            // correction still respects colliders (a direct transform.position write would
            // tunnel and skip OnControllerColliderHit).
            float belowFloor = minHeight - transform.position.y;
            if (belowFloor > 0f)
            {
                cc.Move(Vector3.up * belowFloor);
                if (targetUD < 0f)
                {
                    targetUD = 0f;
                    currentUD = 0f;
                }
            }

            transform.Rotate(Vector3.up, currentYaw * rotationSpeed * Time.deltaTime, Space.World);

            // Detect collisions independently of OnControllerColliderHit so a hit is recorded
            // even when the controller slides along a wall or is nudged by a direct move.
            DetectEnvironmentCollision();
        }

        if (autoSendState && Time.unscaledTime >= nextStateSendTime)
        {
            SendState();
            nextStateSendTime = Time.unscaledTime + (1f / Mathf.Max(1f, stateSendHz));
        }
    }

    void ProcessCommand(string raw)
    {
        lastCommand = raw;
        string cmd = raw.ToLowerInvariant();

        if (cmd.StartsWith("msg "))
        {
            // On-screen status text from the pipeline (keep original casing).
            statusMessage = raw.Substring(4);
            statusMessageTime = Time.unscaledTime;
            return;
        }

        if (cmd.StartsWith("preview "))
        {
            // "preview x y z [label…]" — point the camera at a candidate target.
            // Parsed from `raw` so the optional label keeps its original
            // (e.g. Korean) casing. The drone does NOT move.
            string[] parts = raw.Split(' ');
            if (parts.Length >= 4
                && float.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out float vx)
                && float.TryParse(parts[2], NumberStyles.Float, CultureInfo.InvariantCulture, out float vy)
                && float.TryParse(parts[3], NumberStyles.Float, CultureInfo.InvariantCulture, out float vz))
            {
                previewTarget = new Vector3(vx, vy, vz);
                previewLabel = parts.Length > 4 ? string.Join(" ", parts, 4, parts.Length - 4) : "";
                previewActive = true;
                ShowPreviewMarker(previewTarget);
            }
            else
            {
                Debug.LogWarning($"[Tello] Failed to parse preview command: '{raw}'");
            }
            return;
        }

        if (cmd == "preview_off")
        {
            EndPreview();
            return;
        }

        if (cmd == "command")
        {
            SendState();
            return;
        }

        if (cmd == "state")
        {
            SendState();
            return;
        }

        if (cmd == "takeoff")
        {
            isFlying = true;
            hadCollision = false;
            collisionCount = 0;
            lastCollisionRecordTime = -10f;
            ClearFlightVisuals();
            if (trail != null)
            {
                trail.emitting = true;
            }
            // Takeoff is a deliberate ground-escape: lift the drone to at least minHeight.
            // We set the position directly (not cc.Move) on purpose, because a CharacterController
            // that starts embedded in the floor cannot climb out with cc.Move and would stay stuck.
            Vector3 takeoffPos = transform.position;
            takeoffPos.y = Mathf.Max(takeoffPos.y, minHeight) + 1.0f;
            transform.position = takeoffPos;
            SendState();
            return;
        }

        if (cmd == "land")
        {
            isFlying = false;
            targetLR = targetFB = targetUD = targetYaw = 0f;
            currentLR = currentFB = currentUD = currentYaw = 0f;
            if (trail != null)
            {
                trail.emitting = false;   // freeze the trail so the flight stays reviewable
            }
            SendState();
            return;
        }

        if (cmd.StartsWith("setpos "))
        {
            // "setpos x y z [yaw]" — teleport the drone to a start position (world space).
            string[] parts = cmd.Split(' ');
            if (parts.Length >= 4
                && float.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out float px)
                && float.TryParse(parts[2], NumberStyles.Float, CultureInfo.InvariantCulture, out float py)
                && float.TryParse(parts[3], NumberStyles.Float, CultureInfo.InvariantCulture, out float pz))
            {
                // A CharacterController caches its position; disable it around the teleport
                // so the new transform actually takes effect.
                bool wasEnabled = cc != null && cc.enabled;
                if (wasEnabled) cc.enabled = false;
                transform.position = new Vector3(px, py, pz);
                if (parts.Length >= 5
                    && float.TryParse(parts[4], NumberStyles.Float, CultureInfo.InvariantCulture, out float pyaw))
                {
                    transform.rotation = Quaternion.Euler(0f, pyaw, 0f);
                }
                if (wasEnabled) cc.enabled = true;
                SendState();
            }
            else
            {
                Debug.LogWarning($"[Tello] Failed to parse setpos command: '{cmd}'");
            }
            return;
        }

        if (cmd.StartsWith("rc "))
        {
            string[] parts = cmd.Split(' ');
            if (parts.Length == 5
                && float.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out float lr)
                && float.TryParse(parts[2], NumberStyles.Float, CultureInfo.InvariantCulture, out float fb)
                && float.TryParse(parts[3], NumberStyles.Float, CultureInfo.InvariantCulture, out float ud)
                && float.TryParse(parts[4], NumberStyles.Float, CultureInfo.InvariantCulture, out float yaw))
            {
                targetLR = lr / 100f;
                targetFB = fb / 100f;
                targetUD = ud / 100f;
                targetYaw = yaw / 100f;
            }
            else
            {
                Debug.LogWarning($"[Tello] Failed to parse rc command: '{cmd}'");
            }
            return;
        }

        Debug.Log($"[Tello] Unknown command: '{cmd}'");
    }

    void SendState()
    {
        if (udpServer == null || lastRemoteEndPoint == null)
        {
            return;
        }

        try
        {
            Vector3 pos = transform.position;
            float yaw = transform.eulerAngles.y;
            string payload =
                "{"
                + "\"x\":" + pos.x.ToString("F4", CultureInfo.InvariantCulture) + ","
                + "\"y\":" + pos.y.ToString("F4", CultureInfo.InvariantCulture) + ","
                + "\"z\":" + pos.z.ToString("F4", CultureInfo.InvariantCulture) + ","
                + "\"yaw\":" + yaw.ToString("F4", CultureInfo.InvariantCulture) + ","
                + "\"flying\":" + (isFlying ? "true" : "false") + ","
                + "\"had_collision\":" + (hadCollision ? "true" : "false") + ","
                + "\"collision_count\":" + collisionCount.ToString(CultureInfo.InvariantCulture) + ","
                + "\"time\":" + Time.time.ToString("F4", CultureInfo.InvariantCulture)
                + "}";

            byte[] bytes = Encoding.ASCII.GetBytes(payload);
            IPEndPoint stateEndpoint = new IPEndPoint(lastRemoteEndPoint.Address, statePort);
            udpServer.Send(bytes, bytes.Length, stateEndpoint);
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[Tello] Failed to send state: {e.Message}");
        }
    }

    // User-interaction event (confirm / next) back to the pipeline. Shares the
    // state channel (statePort); the bridge tells the two apart by the "event" key.
    void SendEvent(string name)
    {
        if (udpServer == null || lastRemoteEndPoint == null)
        {
            return;
        }
        try
        {
            string payload = "{\"event\":\"" + name + "\"}";
            byte[] bytes = Encoding.ASCII.GetBytes(payload);
            IPEndPoint stateEndpoint = new IPEndPoint(lastRemoteEndPoint.Address, statePort);
            udpServer.Send(bytes, bytes.Length, stateEndpoint);
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[Tello] Failed to send event: {e.Message}");
        }
    }

    void ShowPreviewMarker(Vector3 position)
    {
        if (previewMarker == null)
        {
            previewMarker = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            previewMarker.name = "PreviewMarker";
            previewMarker.transform.localScale = Vector3.one * 1.5f;
            Destroy(previewMarker.GetComponent<Collider>());
            Renderer rend = previewMarker.GetComponent<Renderer>();
            rend.material = new Material(Shader.Find("Sprites/Default"));
            rend.material.color = new Color(1f, 0.85f, 0.1f, 0.95f);
        }
        previewMarker.transform.position = position;
    }

    void EndPreview()
    {
        previewActive = false;
        previewLabel = "";
        if (previewMarker != null)
        {
            Destroy(previewMarker);
            previewMarker = null;
        }
    }

    void OnControllerColliderHit(ControllerColliderHit hit)
    {
        if (!isFlying || hit.collider == null)
        {
            return;
        }

        RecordCollision();
    }

    // Per-frame proximity probe: catches collisions that OnControllerColliderHit misses
    // (sliding contact, direct moves, or a CharacterController that fails to block).
    void DetectEnvironmentCollision()
    {
        if (collisionProbeRadius <= 0f)
        {
            return;
        }

        int count = Physics.OverlapSphereNonAlloc(
            transform.position,
            collisionProbeRadius,
            overlapResults,
            collisionMask,
            QueryTriggerInteraction.Ignore);

        for (int i = 0; i < count; i++)
        {
            Collider hit = overlapResults[i];
            if (hit == null)
            {
                continue;
            }
            // Skip the drone's own colliders.
            if (hit.transform == transform || hit.transform.IsChildOf(transform))
            {
                continue;
            }

            RecordCollision();
            break;
        }
    }

    void RecordCollision()
    {
        if (Time.time - lastCollisionRecordTime < collisionDebounce)
        {
            return;
        }

        hadCollision = true;
        collisionCount += 1;
        lastCollisionRecordTime = Time.time;

        if (markCollisions)
        {
            SpawnCollisionMarker(transform.position);
        }
    }

    void SpawnCollisionMarker(Vector3 position)
    {
        GameObject marker = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        marker.name = $"CollisionMarker_{collisionCount}";
        marker.transform.position = position;
        marker.transform.localScale = Vector3.one * collisionMarkerSize;
        // The marker must not collide with anything (it would trigger more collisions).
        Destroy(marker.GetComponent<Collider>());
        Renderer rend = marker.GetComponent<Renderer>();
        rend.material = new Material(Shader.Find("Sprites/Default"));
        rend.material.color = new Color(1f, 0.15f, 0.1f, 0.95f);
        collisionMarkers.Add(marker);
    }

    void ClearFlightVisuals()
    {
        if (trail != null)
        {
            trail.Clear();
        }
        foreach (GameObject marker in collisionMarkers)
        {
            if (marker != null)
            {
                Destroy(marker);
            }
        }
        collisionMarkers.Clear();
    }

    void OnDrawGizmosSelected()
    {
        if (collisionProbeRadius <= 0f)
        {
            return;
        }
        Gizmos.color = new Color(1f, 0.4f, 0.2f, 0.6f);
        Gizmos.DrawWireSphere(transform.position, collisionProbeRadius);
    }

    void OnGUI()
    {
        GUIStyle style = new GUIStyle(GUI.skin.label);
        style.fontSize = 14;

        // Pipeline status banner (from the "msg" UDP command), top-center.
        if (!string.IsNullOrEmpty(statusMessage))
        {
            GUIStyle banner = new GUIStyle(GUI.skin.box);
            banner.fontSize = 22;
            banner.alignment = TextAnchor.MiddleCenter;
            banner.wordWrap = true;
            float age = Time.unscaledTime - statusMessageTime;
            GUI.color = age > 30f ? new Color(1f, 1f, 1f, 0.5f) : Color.white;
            float width = Mathf.Min(Screen.width - 40f, 720f);
            GUI.Box(new Rect((Screen.width - width) / 2f, 12f, width, 44f), statusMessage, banner);
        }

        // Candidate-preview UI: label + [이동]/[다음 후보] buttons, bottom-center.
        if (previewActive)
        {
            float panelW = Mathf.Min(Screen.width - 40f, 560f);
            float panelX = (Screen.width - panelW) / 2f;
            float panelY = Screen.height - 130f;

            if (!string.IsNullOrEmpty(previewLabel))
            {
                GUIStyle labelBox = new GUIStyle(GUI.skin.box);
                labelBox.fontSize = 20;
                labelBox.alignment = TextAnchor.MiddleCenter;
                labelBox.wordWrap = true;
                GUI.color = Color.white;
                GUI.Box(new Rect(panelX, panelY, panelW, 40f), previewLabel, labelBox);
            }

            GUIStyle button = new GUIStyle(GUI.skin.button);
            button.fontSize = 22;
            float btnW = (panelW - 20f) / 2f;
            float btnY = panelY + 48f;
            GUI.color = new Color(0.35f, 1f, 0.45f);
            if (GUI.Button(new Rect(panelX, btnY, btnW, 52f), "이동", button))
            {
                // The pipeline answers with preview_off; keep the preview up
                // until then so a slow link doesn't flicker the UI.
                SendEvent("confirm");
            }
            GUI.color = new Color(1f, 0.8f, 0.3f);
            if (GUI.Button(new Rect(panelX + btnW + 20f, btnY, btnW, 52f), "다음 후보", button))
            {
                SendEvent("next");
            }
        }

        GUI.color = isFlying ? Color.cyan : Color.yellow;
        GUI.Label(new Rect(10, 10, 520, 25), $"[Tello] State: {(isFlying ? "Flying" : "Landed")}", style);
        GUI.color = Color.white;
        GUI.Label(new Rect(10, 35, 520, 25), $"Last command: {lastCommand}", style);
        GUI.Label(new Rect(10, 60, 520, 25), $"RC LR:{targetLR:F2} FB:{targetFB:F2} UD:{targetUD:F2} Yaw:{targetYaw:F2}", style);
        GUI.Label(new Rect(10, 85, 520, 25), $"Position: {transform.position}", style);
        GUI.Label(new Rect(10, 110, 520, 25), $"State stream: {statePort} @ {stateSendHz:F0}Hz", style);
        GUI.Label(new Rect(10, 135, 520, 25), $"Collision: {(hadCollision ? "yes" : "no")} ({collisionCount})", style);
    }

    void OnApplicationQuit()
    {
        shouldQuit = true;
        Thread.Sleep(200);
        if (udpServer != null)
        {
            udpServer.Close();
            udpServer = null;
        }
    }
}
