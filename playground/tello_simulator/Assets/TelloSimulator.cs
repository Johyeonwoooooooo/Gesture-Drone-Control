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

    private CharacterController cc;

    void Start()
    {
        Application.runInBackground = true;
        cc = GetComponent<CharacterController>();
        StartUDPServer();
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

                string msg = Encoding.ASCII.GetString(data).Trim().ToLowerInvariant();
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

            Vector3 pos = transform.position;
            if (pos.y < minHeight)
            {
                pos.y = minHeight;
                transform.position = pos;
                if (targetUD < 0f)
                {
                    targetUD = 0f;
                    currentUD = 0f;
                }
            }

            transform.Rotate(Vector3.up, currentYaw * rotationSpeed * Time.deltaTime, Space.World);
        }

        if (autoSendState && Time.unscaledTime >= nextStateSendTime)
        {
            SendState();
            nextStateSendTime = Time.unscaledTime + (1f / Mathf.Max(1f, stateSendHz));
        }
    }

    void ProcessCommand(string cmd)
    {
        lastCommand = cmd;

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
            transform.position += Vector3.up * 1.0f;
            SendState();
            return;
        }

        if (cmd == "land")
        {
            isFlying = false;
            targetLR = targetFB = targetUD = targetYaw = 0f;
            currentLR = currentFB = currentUD = currentYaw = 0f;
            SendState();
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

    void OnControllerColliderHit(ControllerColliderHit hit)
    {
        if (!isFlying || hit.collider == null)
        {
            return;
        }

        if (Time.time - lastCollisionRecordTime < 0.2f)
        {
            return;
        }

        hadCollision = true;
        collisionCount += 1;
        lastCollisionRecordTime = Time.time;
    }

    void OnGUI()
    {
        GUIStyle style = new GUIStyle(GUI.skin.label);
        style.fontSize = 14;

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
