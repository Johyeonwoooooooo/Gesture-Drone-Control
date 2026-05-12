using UnityEngine;
using UnityEditor;

public class AddMeshColliders : Editor
{
    [MenuItem("Tools/Add Mesh Colliders to Selected")]
    static void AddCollidersToSelected()
    {
        GameObject selected = Selection.activeGameObject;
        if (selected == null)
        {
            Debug.LogError("오브젝트를 먼저 선택하세요!");
            return;
        }

        MeshFilter[] filters = selected.GetComponentsInChildren<MeshFilter>();
        int count = 0;

        foreach (MeshFilter mf in filters)
        {
            if (mf.sharedMesh == null) continue;

            MeshCollider existing = mf.GetComponent<MeshCollider>();
            if (existing != null) continue; // 이미 있으면 스킵

            MeshCollider mc = mf.gameObject.AddComponent<MeshCollider>();
            mc.sharedMesh = mf.sharedMesh;
            mc.convex = false;
            count++;
        }

        Debug.Log($"<color=green>완료! {count}개 MeshCollider 추가됨</color>");
    }
}