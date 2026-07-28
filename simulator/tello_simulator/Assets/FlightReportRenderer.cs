using UnityEngine;
using System;
using System.Collections.Generic;
using System.IO;

[Serializable]
public class FlightReportPayload
{
    public string planner;
    public int intrusion_steps;
    public SerializableVector3[] trajectory_world;
    public SerializableVector3[] path_world;
    public SerializableVector3[] intrusions_world;
}

/// <summary>
/// Draws the last executed flight directly in the Unity scene: the flown
/// trajectory as a line plus a red marker on every voxel-intrusion position
/// computed by the Python post-run check.
///
/// unity_autopilot_3d.py writes Assets/Resources/flight_trajectory_3d.json after
/// every --execute run, so this refreshes automatically right after landing.
/// Pairs with PlannedPathRenderer (planned path) and VoxelMapRenderer (obstacles).
/// </summary>
public class FlightReportRenderer : MonoBehaviour
{
    [Header("Report File")]
    public string relativePathFromAssets = "Resources/flight_trajectory_3d.json";
    public bool autoReload = true;
    public float pollSeconds = 1.0f;

    [Header("Style")]
    public Color trajectoryColor = new Color(0.31f, 0.76f, 0.97f, 1.0f);
    public float trajectoryWidth = 0.25f;
    public Color intrusionColor = new Color(1.0f, 0.09f, 0.27f, 0.95f);
    public float intrusionMarkerSize = 1.4f;

    private LineRenderer lineRenderer;
    private readonly List<GameObject> markers = new List<GameObject>();
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
        lineRenderer.startWidth = trajectoryWidth;
        lineRenderer.endWidth = trajectoryWidth;
        lineRenderer.material = new Material(Shader.Find("Sprites/Default"));
        lineRenderer.startColor = trajectoryColor;
        lineRenderer.endColor = trajectoryColor;
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
        TryReloadReport();
    }

    [ContextMenu("Reload Flight Report")]
    // Same as PlannedPathRenderer.SetVisible, plus the intrusion spheres — they
    // are parented to this transform, so SetActive on each is enough.
    public void SetVisible(bool value)
    {
        if (lineRenderer != null) lineRenderer.enabled = value;
        foreach (GameObject marker in markers)
        {
            if (marker != null) marker.SetActive(value);
        }
    }

    public void TryReloadReport()
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
        FlightReportPayload payload = JsonUtility.FromJson<FlightReportPayload>(json);
        if (payload == null || payload.trajectory_world == null || payload.trajectory_world.Length == 0)
        {
            return;
        }

        lineRenderer.positionCount = payload.trajectory_world.Length;
        for (int i = 0; i < payload.trajectory_world.Length; i++)
        {
            lineRenderer.SetPosition(i, payload.trajectory_world[i].ToVector3());
        }

        ClearMarkers();
        if (payload.intrusions_world != null)
        {
            foreach (SerializableVector3 p in payload.intrusions_world)
            {
                SpawnIntrusionMarker(p.ToVector3());
            }
        }

        lastWriteTimeUtc = writeTime;
        Debug.Log($"[FlightReport] trajectory {payload.trajectory_world.Length} pts, " +
                  $"intrusions {payload.intrusion_steps}");
    }

    void SpawnIntrusionMarker(Vector3 position)
    {
        GameObject marker = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        marker.name = "IntrusionMarker";
        marker.transform.SetParent(transform, true);
        marker.transform.position = position;
        marker.transform.localScale = Vector3.one * intrusionMarkerSize;
        Destroy(marker.GetComponent<Collider>());
        Renderer rend = marker.GetComponent<Renderer>();
        rend.material = new Material(Shader.Find("Sprites/Default"));
        rend.material.color = intrusionColor;
        markers.Add(marker);
    }

    void ClearMarkers()
    {
        foreach (GameObject marker in markers)
        {
            if (marker != null)
            {
                Destroy(marker);
            }
        }
        markers.Clear();
    }
}
