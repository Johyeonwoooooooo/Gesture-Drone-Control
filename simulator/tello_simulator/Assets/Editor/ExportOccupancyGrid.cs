using UnityEditor;
using UnityEngine;
using System.Collections.Generic;
using System.IO;
using System.Text;

public class ExportOccupancyGrid : Editor
{
    private const float CellSize = 2.0f;
    private const float FlightHeight = 10.0f;
    private const float DroneRadius = 1.5f;
    private const float SampleHalfHeight = 2.0f;

    [MenuItem("Tools/Export Occupancy Grid (Selected Root)")]
    static void ExportSelectedRootGrid()
    {
        GameObject selected = Selection.activeGameObject;
        if (selected == null)
        {
            Debug.LogError("Select the imported scene root first.");
            return;
        }

        Bounds? maybeBounds = ComputeBounds(selected);
        if (!maybeBounds.HasValue)
        {
            Debug.LogError("Could not compute bounds from renderers under the selected object.");
            return;
        }

        Bounds bounds = maybeBounds.Value;
        int width = Mathf.CeilToInt(bounds.size.x / CellSize);
        int height = Mathf.CeilToInt(bounds.size.z / CellSize);
        int[,] grid = new int[height, width];

        Vector3 halfExtents = new Vector3(DroneRadius, SampleHalfHeight, DroneRadius);
        int occupiedCount = 0;

        for (int gy = 0; gy < height; gy++)
        {
            for (int gx = 0; gx < width; gx++)
            {
                float wx = bounds.min.x + (gx + 0.5f) * CellSize;
                float wz = bounds.min.z + (gy + 0.5f) * CellSize;
                Vector3 center = new Vector3(wx, FlightHeight, wz);

                Collider[] hits = Physics.OverlapBox(
                    center,
                    halfExtents,
                    Quaternion.identity,
                    ~0,
                    QueryTriggerInteraction.Ignore
                );

                bool occupied = false;
                foreach (Collider hit in hits)
                {
                    if (hit.transform.IsChildOf(selected.transform))
                    {
                        occupied = true;
                        break;
                    }
                }

                if (occupied)
                {
                    grid[gy, gx] = 1;
                    occupiedCount++;
                }
            }
        }

        string savePath = EditorUtility.SaveFilePanel(
            "Export Occupancy Grid",
            Application.dataPath,
            selected.name + "_occupancy_grid.json",
            "json"
        );

        if (string.IsNullOrEmpty(savePath))
        {
            return;
        }

        string json = BuildJson(bounds, width, height, grid);
        File.WriteAllText(savePath, json, Encoding.UTF8);

        Debug.Log(
            $"<color=green>Saved occupancy grid to {savePath}</color>\n" +
            $"Size: {width}x{height}, occupied cells: {occupiedCount}, flightHeight={FlightHeight}, cellSize={CellSize}"
        );
    }

    static Bounds? ComputeBounds(GameObject root)
    {
        Renderer[] renderers = root.GetComponentsInChildren<Renderer>();
        if (renderers.Length == 0)
        {
            return null;
        }

        Bounds bounds = renderers[0].bounds;
        for (int i = 1; i < renderers.Length; i++)
        {
            bounds.Encapsulate(renderers[i].bounds);
        }
        return bounds;
    }

    static string BuildJson(Bounds bounds, int width, int height, int[,] grid)
    {
        StringBuilder sb = new StringBuilder(1024 + (width * height * 2));
        sb.AppendLine("{");
        sb.AppendLine("  \"version\": 1,");
        sb.AppendLine("  \"coordinate_system\": \"unity_xz\",");
        sb.AppendLine($"  \"cell_size\": {CellSize.ToString("F4", System.Globalization.CultureInfo.InvariantCulture)},");
        sb.AppendLine($"  \"flight_height\": {FlightHeight.ToString("F4", System.Globalization.CultureInfo.InvariantCulture)},");
        sb.AppendLine($"  \"drone_radius\": {DroneRadius.ToString("F4", System.Globalization.CultureInfo.InvariantCulture)},");
        sb.AppendLine("  \"origin\": {");
        sb.AppendLine($"    \"x\": {bounds.min.x.ToString("F4", System.Globalization.CultureInfo.InvariantCulture)},");
        sb.AppendLine($"    \"z\": {bounds.min.z.ToString("F4", System.Globalization.CultureInfo.InvariantCulture)}");
        sb.AppendLine("  },");
        sb.AppendLine("  \"size\": {");
        sb.AppendLine($"    \"width\": {width},");
        sb.AppendLine($"    \"height\": {height}");
        sb.AppendLine("  },");
        sb.AppendLine("  \"grid\": [");

        for (int y = 0; y < height; y++)
        {
            List<string> row = new List<string>(width);
            for (int x = 0; x < width; x++)
            {
                row.Add(grid[y, x].ToString());
            }
            string suffix = y == height - 1 ? "" : ",";
            sb.AppendLine($"    [{string.Join(", ", row)}]{suffix}");
        }

        sb.AppendLine("  ]");
        sb.AppendLine("}");
        return sb.ToString();
    }
}
