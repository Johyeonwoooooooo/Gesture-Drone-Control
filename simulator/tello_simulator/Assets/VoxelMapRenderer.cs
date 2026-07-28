using UnityEngine;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text.RegularExpressions;

/// <summary>
/// Draws the occupied cells of a Unity-exported 3D voxel map (the JSON produced by
/// ExportVoxelMap3D) so you can visually check that the planner's obstacle map lines
/// up with the actual scene geometry.
///
/// Each cube is centered at origin + (g + 0.5) * voxelSize, which is exactly where
/// ExportVoxelMap3D sampled occupancy (Physics.OverlapBox) and where the Python
/// grid_to_world now places waypoints. If these cubes do NOT sit on the walls/house
/// mesh, the voxel map is misaligned with the scene.
///
/// Drawn with Gizmos so it works in the Scene view without spawning thousands of
/// GameObjects. Select this object (or enable "Always Draw") to see the voxels.
/// </summary>
[ExecuteAlways]
public class VoxelMapRenderer : MonoBehaviour
{
    [Header("Voxel Map File")]
    [Tooltip("Absolute path, or path relative to the project Assets/ folder.")]
    public string voxelMapPath = "Resources/voxel_map_3d.json";
    public bool autoReload = true;

    [Header("Draw")]
    public bool alwaysDraw = true;          // draw even when this object is not selected
    public bool solidCubes = false;         // solid (slower) vs wireframe cubes
    public Color occupiedColor = new Color(0.2f, 0.6f, 1.0f, 0.25f);
    [Tooltip("Safety cap so a huge map does not freeze the editor.")]
    public int maxCubes = 60000;

    private readonly List<Vector3> centers = new List<Vector3>();
    private float voxelSize = 1.0f;
    private string loadedPath = null;
    private System.DateTime lastWriteTimeUtc = System.DateTime.MinValue;

    private string ResolvePath()
    {
        if (string.IsNullOrEmpty(voxelMapPath))
        {
            return null;
        }
        if (Path.IsPathRooted(voxelMapPath))
        {
            return voxelMapPath;
        }
        return Path.Combine(Application.dataPath, voxelMapPath);
    }

    [ContextMenu("Reload Voxel Map")]
    public void Reload()
    {
        centers.Clear();
        loadedPath = null;

        string fullPath = ResolvePath();
        if (string.IsNullOrEmpty(fullPath) || !File.Exists(fullPath))
        {
            return;
        }

        string text = File.ReadAllText(fullPath);

        voxelSize = ParseFloat(text, "voxel_size", 1.0f);
        float ox = ParseFloat(text, "x", 0f, "origin");
        float oy = ParseFloat(text, "y", 0f, "origin");
        float oz = ParseFloat(text, "z", 0f, "origin");

        // occupied entries are the only integer triples in brackets, e.g. [3, 0, 6].
        // origin/size use objects {}, voxel_size/drone_radius are decimals, so this
        // regex matches occupied cells only.
        MatchCollection matches = Regex.Matches(text, @"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]");
        foreach (Match m in matches)
        {
            int gx = int.Parse(m.Groups[1].Value, CultureInfo.InvariantCulture);
            int gy = int.Parse(m.Groups[2].Value, CultureInfo.InvariantCulture);
            int gz = int.Parse(m.Groups[3].Value, CultureInfo.InvariantCulture);

            // Voxel CENTER, matching ExportVoxelMap3D sampling + Python grid_to_world.
            centers.Add(new Vector3(
                ox + (gx + 0.5f) * voxelSize,
                oy + (gy + 0.5f) * voxelSize,
                oz + (gz + 0.5f) * voxelSize
            ));

            if (centers.Count >= maxCubes)
            {
                Debug.LogWarning($"VoxelMapRenderer: hit maxCubes={maxCubes}; not all voxels drawn.");
                break;
            }
        }

        loadedPath = fullPath;
        lastWriteTimeUtc = File.GetLastWriteTimeUtc(fullPath);
    }

    private void EnsureLoaded()
    {
        string fullPath = ResolvePath();
        if (string.IsNullOrEmpty(fullPath) || !File.Exists(fullPath))
        {
            return;
        }

        bool changed = loadedPath != fullPath;
        if (autoReload && File.GetLastWriteTimeUtc(fullPath) > lastWriteTimeUtc)
        {
            changed = true;
        }
        if (changed)
        {
            Reload();
        }
    }

    void OnDrawGizmos()
    {
        if (alwaysDraw)
        {
            DrawVoxels();
        }
    }

    void OnDrawGizmosSelected()
    {
        if (!alwaysDraw)
        {
            DrawVoxels();
        }
    }

    private void DrawVoxels()
    {
        EnsureLoaded();
        if (centers.Count == 0)
        {
            return;
        }

        Gizmos.color = occupiedColor;
        Vector3 size = Vector3.one * voxelSize;
        for (int i = 0; i < centers.Count; i++)
        {
            if (solidCubes)
            {
                Gizmos.DrawCube(centers[i], size);
            }
            else
            {
                Gizmos.DrawWireCube(centers[i], size);
            }
        }
    }

    // Minimal number extraction. "section" optionally narrows the search to the text
    // following the first occurrence of that key (used so origin.x/y/z don't collide
    // with other x/y/z fields).
    private static float ParseFloat(string text, string key, float fallback, string section = null)
    {
        string haystack = text;
        if (!string.IsNullOrEmpty(section))
        {
            int sectionIdx = text.IndexOf("\"" + section + "\"", System.StringComparison.Ordinal);
            if (sectionIdx >= 0)
            {
                haystack = text.Substring(sectionIdx);
            }
        }

        Match m = Regex.Match(haystack, "\"" + Regex.Escape(key) + "\"\\s*:\\s*(-?\\d+(?:\\.\\d+)?)");
        if (m.Success && float.TryParse(m.Groups[1].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out float value))
        {
            return value;
        }
        return fallback;
    }
}
