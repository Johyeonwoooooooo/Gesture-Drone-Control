using UnityEditor;
using UnityEngine;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;

public class ExportVoxelMap3D : Editor
{
    private const float VoxelSize = 2.0f;
    private const float DroneRadius = 1.5f;
    private const float SampleHalfHeight = 1.5f;

    [MenuItem("Tools/Export 3D Voxel Map (Selected Root)")]
    static void ExportSelectedRootVoxelMap()
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
        int width = Mathf.CeilToInt(bounds.size.x / VoxelSize);
        int height = Mathf.CeilToInt(bounds.size.y / VoxelSize);
        int depth = Mathf.CeilToInt(bounds.size.z / VoxelSize);

        Vector3 halfExtents = new Vector3(DroneRadius, SampleHalfHeight, DroneRadius);
        List<Vector3Int> occupied = new List<Vector3Int>();

        for (int gy = 0; gy < height; gy++)
        {
            for (int gz = 0; gz < depth; gz++)
            {
                for (int gx = 0; gx < width; gx++)
                {
                    Vector3 center = new Vector3(
                        bounds.min.x + (gx + 0.5f) * VoxelSize,
                        bounds.min.y + (gy + 0.5f) * VoxelSize,
                        bounds.min.z + (gz + 0.5f) * VoxelSize
                    );

                    Collider[] hits = Physics.OverlapBox(
                        center,
                        halfExtents,
                        Quaternion.identity,
                        ~0,
                        QueryTriggerInteraction.Ignore
                    );

                    bool isOccupied = false;
                    foreach (Collider hit in hits)
                    {
                        if (hit.transform.IsChildOf(selected.transform))
                        {
                            isOccupied = true;
                            break;
                        }
                    }

                    if (isOccupied)
                    {
                        occupied.Add(new Vector3Int(gx, gy, gz));
                    }
                }
            }
        }

        string savePath = EditorUtility.SaveFilePanel(
            "Export 3D Voxel Map",
            Application.dataPath,
            selected.name + "_voxel_map_3d.json",
            "json"
        );

        if (string.IsNullOrEmpty(savePath))
        {
            return;
        }

        string json = BuildJson(bounds, width, height, depth, occupied);
        File.WriteAllText(savePath, json, Encoding.UTF8);

        Debug.Log(
            $"<color=green>Saved 3D voxel map to {savePath}</color>\n" +
            $"Size: {width}x{height}x{depth}, occupied voxels: {occupied.Count}, voxelSize={VoxelSize}"
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

    static string BuildJson(Bounds bounds, int width, int height, int depth, List<Vector3Int> occupied)
    {
        StringBuilder sb = new StringBuilder(4096 + occupied.Count * 24);
        sb.AppendLine("{");
        sb.AppendLine("  \"version\": 1,");
        sb.AppendLine("  \"coordinate_system\": \"unity_xyz\",");
        sb.AppendLine($"  \"voxel_size\": {VoxelSize.ToString("F4", CultureInfo.InvariantCulture)},");
        sb.AppendLine($"  \"drone_radius\": {DroneRadius.ToString("F4", CultureInfo.InvariantCulture)},");
        sb.AppendLine("  \"origin\": {");
        sb.AppendLine($"    \"x\": {bounds.min.x.ToString("F4", CultureInfo.InvariantCulture)},");
        sb.AppendLine($"    \"y\": {bounds.min.y.ToString("F4", CultureInfo.InvariantCulture)},");
        sb.AppendLine($"    \"z\": {bounds.min.z.ToString("F4", CultureInfo.InvariantCulture)}");
        sb.AppendLine("  },");
        sb.AppendLine("  \"size\": {");
        sb.AppendLine($"    \"width\": {width},");
        sb.AppendLine($"    \"height\": {height},");
        sb.AppendLine($"    \"depth\": {depth}");
        sb.AppendLine("  },");
        sb.AppendLine("  \"occupied\": [");

        for (int i = 0; i < occupied.Count; i++)
        {
            Vector3Int cell = occupied[i];
            string suffix = i == occupied.Count - 1 ? "" : ",";
            sb.AppendLine($"    [{cell.x}, {cell.y}, {cell.z}]{suffix}");
        }

        sb.AppendLine("  ]");
        sb.AppendLine("}");
        return sb.ToString();
    }
}
