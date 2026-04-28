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

    // 메인 스레드에서 처리할 명령 큐 (스레드 안전)
    private ConcurrentQueue<string> commandQueue = new ConcurrentQueue<string>();

    private UdpClient udpServer;
    private Thread receiveThread;

    private float targetLR, targetFB, targetUD, targetYaw;
    private float currentLR, currentFB, currentUD, currentYaw;
    private float velLR, velFB, velUD, velYaw;

    private bool isFlying = false;
    private bool shouldQuit = false;

    private string lastCommand = "";

    void Start()
    {
        Application.runInBackground = true; // 백그라운드 실행 활성화
        Rigidbody rb = GetComponent<Rigidbody>();
        if (rb != null)
        {
            rb.useGravity = false;
            rb.isKinematic = true;
            Debug.Log("<color=green>[Tello] Rigidbody isKinematic = true 설정 완료</color>");
        }

        StartUDPServer();
    }

    void StartUDPServer()
    {
        try
        {
            udpServer = new UdpClient(port);
            udpServer.Client.ReceiveTimeout = 100; // 블로킹 방지

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

    // ─── 수신 스레드 (메인 스레드와 완전 분리) ───────────────────────────────
    void ReceiveLoop()
    {
        IPEndPoint remote = new IPEndPoint(IPAddress.Any, 0);

        while (!shouldQuit)
        {
            try
            {
                byte[] data = udpServer.Receive(ref remote);
                string msg = Encoding.ASCII.GetString(data).Trim().ToLower();

                // 즉시 "ok" 응답 (파이썬 타임아웃 방지)
                byte[] ok = Encoding.ASCII.GetBytes("ok");
                udpServer.Send(ok, ok.Length, remote);

                // 명령을 큐에 추가 → 메인 스레드(Update)에서 처리
                commandQueue.Enqueue(msg);
            }
            catch (SocketException)
            {
                // ReceiveTimeout 정상 타임아웃 — 무시
            }
            catch (Exception e)
            {
                if (!shouldQuit)
                    Debug.LogWarning($"[Tello] 수신 오류: {e.Message}");
            }
        }
    }

    // ─── 메인 스레드 Update ──────────────────────────────────────────────────
    void Update()
    {
        // 1. 큐에서 명령 처리 (메인 스레드에서 안전하게 실행)
        while (commandQueue.TryDequeue(out string cmd))
        {
            ProcessCommand(cmd);
        }

        if (!isFlying) return;

        // 2. RC 값 부드럽게 보간
        currentLR  = Mathf.SmoothDamp(currentLR,  targetLR,  ref velLR,  smoothTime);
        currentFB  = Mathf.SmoothDamp(currentFB,  targetFB,  ref velFB,  smoothTime);
        currentUD  = Mathf.SmoothDamp(currentUD,  targetUD,  ref velUD,  smoothTime);
        currentYaw = Mathf.SmoothDamp(currentYaw, targetYaw, ref velYaw, smoothTime);

        // 3. 이동 적용 (로컬 좌표계)
        Vector3 move = new Vector3(currentLR, currentUD, currentFB) * moveSpeed * Time.deltaTime;
        transform.Translate(move, Space.Self);

        // 4. 바닥 뚫기 방지
        Vector3 pos = transform.position;
        if (pos.y < minHeight)
        {
            pos.y = minHeight;
            transform.position = pos;
            if (targetUD < 0f) { targetUD = 0f; currentUD = 0f; }
        }

        // 5. 회전 (Yaw)
        transform.Rotate(Vector3.up, currentYaw * rotationSpeed * Time.deltaTime, Space.World);
    }
    // ─── 명령 처리 ───────────────────────────────────────────────────────────
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
            // 이륙 시 즉시 1미터 상승하여 바닥과 분리
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

        // "rc lr fb ud yaw" 파싱
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

    // ─── 실시간 상태 오버레이 (Game 뷰에서 확인 가능) ───────────────────────
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