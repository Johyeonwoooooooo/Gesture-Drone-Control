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

    [Header("Spawn (Home)")]
    [Tooltip("Teleport the drone to spawnPosition on Play, so it starts inside the house " +
             "instead of at the scene origin. Without this the drone only moves in once the " +
             "server connects and runs its home teleport.")]
    public bool spawnAtHome = true;
    [Tooltip("Unity-world home for building 00809: litept_backend.default_home() = " +
             "mosaic (4.50, -1.04, -2.06), mapped through " +
             "simulator/bridge/transforms/00809_Qpor2mEya8F.json. The server's `home` " +
             "command stays authoritative — this is just the pre-connection Play position, " +
             "so it needs updating if the building or the glb transform changes.")]
    public Vector3 spawnPosition = new Vector3(-22.51f, 5.2f, 5.22f);
    public float spawnYaw = 0f;

    [Header("Debug HUD")]
    [Tooltip("Raw telemetry text (state, last command, rc, position, collisions). " +
             "Off by default — CamcorderHUD draws the in-fiction overlay instead.")]
    public bool showDebugHud = false;

    [Header("Flight Visualization")]
    [Tooltip("Draw a colored trail behind the drone while it flies. Off by " +
             "default — the trail is still built and recording, so the " +
             "settings panel (Tab) can turn it back on mid-flight.")]
    public bool showFlightTrail = false;
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

    // Patrol scan state, driven by the `scan` verb. The pipeline waits for the
    // answering scan_done event, so every exit path must send one — otherwise
    // it sits there until its timeout and falls back to spinning us over rc.
    [NonSerialized] public bool scanActive = false;
    private float scanTargetDeg = 0f;
    private float scanTurnedDeg = 0f;
    private float scanDegPerSec = 50f;

    // Detection hold, set by PatrolPersonDetection.PauseForDetection. While it
    // runs the drone is frozen — rc is dropped and the sweep stops advancing,
    // so the paused seconds do not count toward scanTurnedDeg and the scan
    // resumes from the same angle.
    private float holdMovementUntil = -1f;

    // Read by CamcorderHUD (battery drains faster in flight).
    public bool IsFlying => isFlying;

    // How much of the current sweep is left. PatrolPersonDetection stops
    // capturing near the end so a detect never lands after scan_done, which the
    // pipeline treats as "this room is finished".
    public float ScanRemainingDeg => scanActive ? Mathf.Max(0f, scanTargetDeg - scanTurnedDeg) : 0f;

    // Freeze the drone in place for `seconds` — the visible "stop and look" when
    // the detector finds someone. Extends an active hold, never shortens it.
    public void PauseForDetection(float seconds)
    {
        holdMovementUntil = Mathf.Max(holdMovementUntil, Time.time + Mathf.Max(0f, seconds));
        targetLR = targetFB = targetUD = targetYaw = 0f;
        currentLR = currentFB = currentUD = currentYaw = 0f;
    }

    // Path/marker visibility, driven by SettingsPanel. The trail keeps recording
    // while hidden so toggling it back on shows the whole flight, not a stub.
    public void SetTrailVisible(bool value)
    {
        if (trail != null) trail.enabled = value;
    }

    public void SetCollisionMarkersVisible(bool value)
    {
        foreach (GameObject marker in collisionMarkers)
        {
            if (marker != null) marker.SetActive(value);
        }
    }

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
        if (spawnAtHome)
        {
            TeleportTo(spawnPosition, spawnYaw);
            Debug.Log($"[Tello] spawned at home {spawnPosition}");
        }
        // Always built, `enabled` carries the visibility — otherwise the panel
        // toggle has nothing to switch on when it starts hidden.
        SetupFlightTrail();
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
        trail.enabled = showFlightTrail;
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

        bool holdingForDetection = Time.time < holdMovementUntil;

        if (scanActive)
        {
            if (!isFlying)
            {
                StopScan("landed");
            }
            else if (holdingForDetection)
            {
                // Hold the angle. The sweep stays active and scanTurnedDeg stays
                // put, so scan_done cannot fire while we are stopped on someone.
                targetYaw = 0f;
            }
            else
            {
                targetYaw = Mathf.Clamp(scanDegPerSec / rotationSpeed, -1f, 1f);
                scanTurnedDeg += Mathf.Abs(currentYaw) * rotationSpeed * Time.deltaTime;
                if (scanTurnedDeg >= scanTargetDeg)
                {
                    StopScan("done");
                }
            }
        }

        if (holdingForDetection)
        {
            targetLR = targetFB = targetUD = targetYaw = 0f;
            currentLR = currentFB = currentUD = currentYaw = 0f;
        }

        if (isFlying && !holdingForDetection)
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
                float pyaw = transform.eulerAngles.y;
                if (parts.Length >= 5)
                {
                    float.TryParse(parts[4], NumberStyles.Float, CultureInfo.InvariantCulture, out pyaw);
                }
                TeleportTo(new Vector3(px, py, pz), pyaw);
                SendState();
            }
            else
            {
                Debug.LogWarning($"[Tello] Failed to parse setpos command: '{cmd}'");
            }
            return;
        }

        if (cmd.StartsWith("scan_stop"))
        {
            StopScan("stopped");
            return;
        }

        if (cmd.StartsWith("scan"))
        {
            // "scan <deg/s> [turns]" — sweep this room and report back.
            // PatrolPersonDetection captures the frames and calls YOLO; we only
            // own the spin, the hold, and scan_done. With no detector process
            // listening the sweep still runs, just without detects.
            string[] sp = cmd.Split(' ');
            float dps = 50f, turns = 1f;
            if (sp.Length > 1) float.TryParse(sp[1], NumberStyles.Float, CultureInfo.InvariantCulture, out dps);
            if (sp.Length > 2) float.TryParse(sp[2], NumberStyles.Float, CultureInfo.InvariantCulture, out turns);
            StartScan(dps, turns);
            return;
        }

        if (cmd.StartsWith("rc "))
        {
            // The pipeline keeps streaming rc while we are stopped on a
            // detection; swallow it so the hold is not overwritten mid-frame.
            if (Time.time < holdMovementUntil)
            {
                targetLR = targetFB = targetUD = targetYaw = 0f;
                return;
            }

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

    // A CharacterController caches its position, so it has to be disabled around a
    // teleport or the new transform is silently reverted. Used by both `setpos` and
    // the home spawn.
    void TeleportTo(Vector3 position, float yaw)
    {
        bool wasEnabled = cc != null && cc.enabled;
        if (wasEnabled) cc.enabled = false;
        transform.position = position;
        transform.rotation = Quaternion.Euler(0f, yaw, 0f);
        if (wasEnabled) cc.enabled = true;
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

    public void StartScan(float degPerSec, float turns)
    {
        if (!isFlying)
        {
            Debug.LogWarning("[Tello] scan while landed — nothing to sweep");
            SendEvent("scan_done", "\"degrees\":0");
            return;
        }
        scanDegPerSec = Mathf.Max(1f, degPerSec);
        scanTargetDeg = 360f * Mathf.Max(0.1f, turns);
        scanTurnedDeg = 0f;
        scanActive = true;
        Debug.Log($"[Tello] scan {scanDegPerSec:F0} deg/s x {turns:F1}");
        // Ack immediately. A quiet room sends nothing until scan_done seconds
        // later, and the pipeline has to tell "this build ignores scan" from
        // "this room is empty" long before that.
        SendEvent("scan_started",
                  "\"target\":" + scanTargetDeg.ToString("F1", CultureInfo.InvariantCulture));
    }

    // `reason` is for the log only — the pipeline just needs the event.
    void StopScan(string reason)
    {
        if (!scanActive)
        {
            return;
        }
        scanActive = false;
        targetYaw = 0f;
        holdMovementUntil = -1f;   // never leave the drone frozen past the sweep
        Debug.Log($"[Tello] scan {reason} at {scanTurnedDeg:F0} deg");
        if (reason != "stopped")
        {
            // Landing mid-sweep still has to answer, or the pipeline waits out
            // its timeout. Only an explicit scan_stop goes silent — whoever
            // sent it is not waiting.
            SendEvent("scan_done",
                      "\"degrees\":" + scanTurnedDeg.ToString("F1", CultureInfo.InvariantCulture));
        }
    }

    // Report a detection to the pipeline. `box` is percent of the camera frame
    // — the unit the web console draws in, so nothing downstream rescales it.
    public void ReportDetection(string label, float confidence,
                                float left, float top, float width, float height,
                                string imagePath = null)
    {
        string body =
            "\"label\":\"" + label + "\","
            + "\"conf\":" + confidence.ToString("F3", CultureInfo.InvariantCulture) + ","
            + "\"box\":{"
            + "\"l\":" + left.ToString("F1", CultureInfo.InvariantCulture) + ","
            + "\"t\":" + top.ToString("F1", CultureInfo.InvariantCulture) + ","
            + "\"w\":" + width.ToString("F1", CultureInfo.InvariantCulture) + ","
            + "\"h\":" + height.ToString("F1", CultureInfo.InvariantCulture)
            + "}";
        if (!string.IsNullOrEmpty(imagePath))
        {
            body += ",\"image_path\":\"" + imagePath.Replace("\\", "/") + "\"";
        }
        SendEvent("detect", body);
    }

    // Event back to the pipeline (scan_done, detect, ...). Shares the state
    // channel (statePort); the bridge tells the two apart by the "event" key.
    // `body` is raw JSON appended inside the object, or null for a bare event.
    void SendEvent(string name, string body = null)
    {
        if (udpServer == null || lastRemoteEndPoint == null)
        {
            return;
        }
        try
        {
            string payload = "{\"event\":\"" + name + "\""
                             + (string.IsNullOrEmpty(body) ? "" : "," + body) + "}";
            byte[] bytes = Encoding.ASCII.GetBytes(payload);
            IPEndPoint stateEndpoint = new IPEndPoint(lastRemoteEndPoint.Address, statePort);
            udpServer.Send(bytes, bytes.Length, stateEndpoint);
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[Tello] Failed to send event: {e.Message}");
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
            // y=72, not 12: on a narrow Game view the centered banner reaches far
            // enough left to sit on top of CamcorderHUD's REC block.
            GUI.Box(new Rect((Screen.width - width) / 2f, 72f, width, 44f), statusMessage, banner);
        }

        // Raw telemetry readout. Off by default — CamcorderHUD owns the top-left
        // corner now — but kept for debugging the link and the collision probe.
        if (showDebugHud)
        {
            GUI.color = isFlying ? Color.cyan : Color.yellow;
            GUI.Label(new Rect(10, 170, 520, 25), $"[Tello] State: {(isFlying ? "Flying" : "Landed")}", style);
            GUI.color = Color.white;
            GUI.Label(new Rect(10, 195, 520, 25), $"Last command: {lastCommand}", style);
            GUI.Label(new Rect(10, 220, 520, 25), $"RC LR:{targetLR:F2} FB:{targetFB:F2} UD:{targetUD:F2} Yaw:{targetYaw:F2}", style);
            GUI.Label(new Rect(10, 245, 520, 25), $"Position: {transform.position}", style);
            GUI.Label(new Rect(10, 270, 520, 25), $"State stream: {statePort} @ {stateSendHz:F0}Hz", style);
            GUI.Label(new Rect(10, 295, 520, 25), $"Collision: {(hadCollision ? "yes" : "no")} ({collisionCount})", style);
            GUI.color = Color.white;
        }
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
