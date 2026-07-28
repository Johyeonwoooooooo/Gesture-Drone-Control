using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System;
using System.Collections.Concurrent;

public class TelloSimulator : MonoBehaviour
{
    [Header("Network Settings")]
    public int port = 9000;

    [Header("Movement Settings")]
    public float moveSpeed = 15f;
    public float rotationSpeed = 100f;
    public float smoothTime = 0.05f;

    [Header("Floor Settings")]
    [Tooltip("드론이 내려갈 수 없는 최소 Y 높이. Plane의 Y 위치 + 드론 높이 절반으로 설정.")]
    public float minHeight = 0.5f;

    private ConcurrentQueue<string> commandQueue = new ConcurrentQueue<string>();

    private UdpClient udpServer;
    private Thread receiveThread;

    private float targetLR, targetFB, targetUD, targetYaw;
    private float currentLR, currentFB, currentUD, currentYaw;
    private float velLR, velFB, velUD, velYaw;

    private bool isFlying = false;
    private bool shouldQuit = false;
    private string lastCommand = "";

    private CharacterController cc; // ← 추가

    void Start()
    {
        Application.runInBackground = true;

        cc = GetComponent<CharacterController>(); // ← Rigidbody 대신

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

            Debug.Log($"<color=green>[Tello] UDP 서버 시작 → 포트 {port} 대기 중...</color>");
        }
        catch (Exception e)
        {
            Debug.LogError($"[Tello] UDP 서버 시작 실패: {e.Message}");
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
                string msg = Encoding.ASCII.GetString(data).Trim().ToLower();

                byte[] ok = Encoding.ASCII.GetBytes("ok");
                udpServer.Send(ok, ok.Length, remote);

                commandQueue.Enqueue(msg);
            }
            catch (SocketException) { }
            catch (Exception e)
            {
                if (!shouldQuit)
                    Debug.LogWarning($"[Tello] 수신 오류: {e.Message}");
            }
        }
    }

    void Update()
    {
        while (commandQueue.TryDequeue(out string cmd))
        {
            ProcessCommand(cmd);
        }

        if (!isFlying) return;

        currentLR  = Mathf.SmoothDamp(currentLR,  targetLR,  ref velLR,  smoothTime);
        currentFB  = Mathf.SmoothDamp(currentFB,  targetFB,  ref velFB,  smoothTime);
        currentUD  = Mathf.SmoothDamp(currentUD,  targetUD,  ref velUD,  smoothTime);
        currentYaw = Mathf.SmoothDamp(currentYaw, targetYaw, ref velYaw, smoothTime);

        // ✅ CharacterController.Move로 충돌 감지하며 이동
        Vector3 localMove = new Vector3(currentLR, currentUD, currentFB) * moveSpeed * Time.deltaTime;
        Vector3 worldMove = transform.TransformDirection(localMove);
        cc.Move(worldMove);

        // 최소 높이 보정
        Vector3 pos = transform.position;
        if (pos.y < minHeight)
        {
            pos.y = minHeight;
            transform.position = pos;
            if (targetUD < 0f) { targetUD = 0f; currentUD = 0f; }
        }

        // Yaw 회전
        transform.Rotate(Vector3.up, currentYaw * rotationSpeed * Time.deltaTime, Space.World);
    }

    void ProcessCommand(string cmd)
    {
        lastCommand = cmd;

        if (cmd == "command")
        {
            Debug.Log("<color=grey>[Tello] 'command' 수신 → SDK 초기화 확인</color>");
            return;
        }

        if (cmd == "takeoff")
        {
            isFlying = true;
            transform.position += Vector3.up * 1.0f;
            Debug.Log("<color=cyan>[Tello] ✈ 이륙! (1m 상승 완료)</color>");
            return;
        }

        if (cmd == "land")
        {
            isFlying = false;
            targetLR = targetFB = targetUD = targetYaw = 0f;
            currentLR = currentFB = currentUD = currentYaw = 0f;
            Debug.Log("<color=yellow>[Tello] 착륙!</color>");
            return;
        }

        if (cmd.StartsWith("rc "))
        {
            string[] parts = cmd.Split(' ');
            if (parts.Length == 5)
            {
                if (float.TryParse(parts[1], out float lr) &&
                    float.TryParse(parts[2], out float fb) &&
                    float.TryParse(parts[3], out float ud) &&
                    float.TryParse(parts[4], out float yaw))
                {
                    targetLR  = lr  / 100f;
                    targetFB  = fb  / 100f;
                    targetUD  = ud  / 100f;
                    targetYaw = yaw / 100f;
                }
                else
                {
                    Debug.LogWarning($"[Tello] rc 파싱 실패: '{cmd}'");
                }
            }
            return;
        }

        Debug.Log($"<color=grey>[Tello] 알 수 없는 명령: '{cmd}'</color>");
    }

    void OnGUI()
    {
        GUIStyle style = new GUIStyle(GUI.skin.label);
        style.fontSize = 14;

        GUI.color = isFlying ? Color.cyan : Color.yellow;
        GUI.Label(new Rect(10, 10, 500, 25), $"[Tello] 상태: {(isFlying ? "비행 중 ✈" : "착륙")}", style);
        GUI.color = Color.white;
        GUI.Label(new Rect(10, 35, 500, 25), $"마지막 명령: {lastCommand}", style);
        GUI.Label(new Rect(10, 60, 500, 25), $"RC → LR:{targetLR:F2}  FB:{targetFB:F2}  UD:{targetUD:F2}  Yaw:{targetYaw:F2}", style);
        GUI.Label(new Rect(10, 85, 500, 25), $"위치: {transform.position}", style);
    }

    void OnApplicationQuit()
    {
        shouldQuit = true;
        Thread.Sleep(200);
        if (udpServer != null) { udpServer.Close(); udpServer = null; }
    }
}