using UnityEngine;
using System;
using System.Collections.Generic;
using System.IO;

[Serializable]
public class PlannedPathPayload
{
    public string planner;
    public SerializableVector3[] path_world;
    public SerializableVector3 start_world;
    public SerializableVector3 goal_world;
}

[Serializable]
public class SerializableVector3
{
    public float x;
    public float y;
    public float z;

    public Vector3 ToVector3()
    {
        return new Vector3(x, y, z);
    }
}

public class PlannedPathRenderer : MonoBehaviour
{
    [Header("Path File")]
    public string relativePathFromAssets = "Resources/planned_path_3d.json";
    public bool autoReload = true;
    public float pollSeconds = 1.0f;

    [Header("Line Style")]
    public Color lineColor = new Color(0.93f, 0.23f, 0.27f, 1.0f);
    public float lineWidth = 0.35f;

    private LineRenderer lineRenderer;
    private DateTime lastWriteTimeUtc = DateTime.MinValue;
    private float nextPollTime = 0f;

    void Awake()
    {
        lineRenderer = GetComponent<LineRenderer>();
        if (lineRenderer == null)
        {
            lineRenderer = gameObject.AddComponent<LineRenderer>();
        }

        lineRenderer.useWorldSpace = true;
        lineRenderer.loop = false;
        lineRenderer.startWidth = lineWidth;
        lineRenderer.endWidth = lineWidth;
        lineRenderer.material = new Material(Shader.Find("Sprites/Default"));
        lineRenderer.startColor = lineColor;
        lineRenderer.endColor = lineColor;
        lineRenderer.positionCount = 0;
    }

    void Update()
    {
        if (!autoReload)
        {
            return;
        }

        if (Time.unscaledTime < nextPollTime)
        {
            return;
        }

        nextPollTime = Time.unscaledTime + Mathf.Max(0.2f, pollSeconds);
        TryReloadPath();
    }

    [ContextMenu("Reload Planned Path")]
    public void TryReloadPath()
    {
        string fullPath = Path.Combine(Application.dataPath, relativePathFromAssets);
        if (!File.Exists(fullPath))
        {
            return;
        }

        DateTime writeTime = File.GetLastWriteTimeUtc(fullPath);
        if (writeTime <= lastWriteTimeUtc && lineRenderer.positionCount > 0)
        {
            return;
        }

        string json = File.ReadAllText(fullPath);
        PlannedPathPayload payload = JsonUtility.FromJson<PlannedPathPayload>(json);
        if (payload == null || payload.path_world == null || payload.path_world.Length == 0)
        {
            return;
        }

        lineRenderer.positionCount = payload.path_world.Length;
        for (int i = 0; i < payload.path_world.Length; i++)
        {
            lineRenderer.SetPosition(i, payload.path_world[i].ToVector3());
        }

        lastWriteTimeUtc = writeTime;
    }
}
