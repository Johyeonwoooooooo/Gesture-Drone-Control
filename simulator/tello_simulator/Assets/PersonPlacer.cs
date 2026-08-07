using UnityEngine;
using System;
using System.Collections.Generic;

// Puts people in the house at Play: reads Resources/person_spawn_points.json
// (written by simulator/bridge/place_people.py) and clones the NPC prefabs
// already sitting in test.unity onto those spots.
//
// Why clone the scene's own NPCs instead of loading prefabs: the character pack
// lives outside Resources/ and is gitignored, so Resources.Load cannot reach it
// and a serialized prefab reference would need scene setup. Cloning also keeps
// VRAM flat — every clone shares the meshes, materials and textures already
// resident for the templates. Cloning *different* variants is what would cost:
// the full pack is ~157 MB of compressed texture, and only the variants actually
// placed in the scene are ever loaded.
//
// The spots come from the room index, not from hand-placing, so two things
// cannot go wrong: the floor is RoomInfo.floor_z (the ground floor sits at
// world z = -3.06 m, not 0 — easy to get wrong by hand) and the scale is read
// from the building's own transform (5 Unity units per metre; the three
// hand-placed NPCs are at 3, which makes them 1.08 m tall next to the house).
//
// With the character pack not imported there are no templates to clone, so this
// logs once and does nothing — same as before it existed.
public class PersonPlacer : MonoBehaviour
{
    [Tooltip("Resources/ 안의 파일 이름 (확장자 없이).")]
    public string resourceName = "person_spawn_points";

    [Tooltip("이 접두어로 시작하는 루트 오브젝트를 복제 원본으로 쓴다.")]
    public string templateNamePrefix = "npc_";

    [Tooltip("원본 3명은 손으로 놓인 자리라 크기가 안 맞는다. 복제본만 남기고 " +
             "원본은 숨긴다.")]
    public bool hideTemplates = true;

    [Tooltip("스폰 지점에서 아래로 살짝 쏴서 실제 바닥에 붙인다. 포인트 클라우드와 " +
             "메시가 조금 어긋난 경우를 흡수한다. 유니티 유닛.")]
    public bool snapToFloor = true;
    public float snapSearchUp = 1.5f;
    public float snapSearchDown = 3.0f;

    [Tooltip("복제본의 그림자를 끈다. 손전등이 soft shadow 라 원뿔에 든 사람마다 " +
             "그림자 패스가 붙는데, YOLO 는 그림자를 안 본다.")]
    public bool disableShadows = true;

    [Serializable]
    private class Spawn
    {
        public string room;
        public string display;
        public int index;
        public float[] unity_position;
        public float unity_yaw;
        public float unity_scale;
    }

    [Serializable]
    private class SpawnFile
    {
        public string building;
        public Spawn[] spawns;
    }

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void Bootstrap()
    {
        if (FindFirstObjectByType<PersonPlacer>() != null) return;
        if (FindFirstObjectByType<TelloSimulator>() == null) return;   // not the sim scene
        new GameObject("PersonPlacer").AddComponent<PersonPlacer>();
    }

    private void Start()
    {
        TextAsset asset = Resources.Load<TextAsset>(resourceName);
        if (asset == null)
        {
            Debug.LogWarning($"[PersonPlacer] Resources/{resourceName}.json 이 없다 — "
                             + "simulator/bridge/place_people.py 를 돌려서 만든다.");
            return;
        }

        SpawnFile file;
        try
        {
            file = JsonUtility.FromJson<SpawnFile>(asset.text);
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[PersonPlacer] spawn 파일을 못 읽었다: {e.Message}");
            return;
        }
        if (file == null || file.spawns == null || file.spawns.Length == 0)
        {
            Debug.LogWarning("[PersonPlacer] spawn 목록이 비어 있다.");
            return;
        }

        List<GameObject> templates = FindTemplates();
        if (templates.Count == 0)
        {
            Debug.LogWarning("[PersonPlacer] 복제할 NPC 가 씬에 없다 — 캐릭터 에셋을 "
                             + "안 받았거나 이름이 다르다 (README §8). 사람 없이 진행한다.");
            return;
        }

        // 원본을 먼저 끈다. 꺼진 오브젝트를 Instantiate 하면 복제본도 꺼진 채로
        // 나오므로 아래에서 명시적으로 켠다.
        if (hideTemplates)
        {
            foreach (GameObject t in templates)
            {
                t.SetActive(false);
            }
        }

        int placed = 0;
        for (int i = 0; i < file.spawns.Length; i++)
        {
            Spawn s = file.spawns[i];
            if (s == null || s.unity_position == null || s.unity_position.Length < 3)
            {
                continue;
            }
            GameObject template = templates[i % templates.Count];
            if (SpawnOne(template, s) != null)
            {
                placed++;
            }
        }

        Debug.Log($"<color=cyan>[PersonPlacer] {placed}/{file.spawns.Length} 명 배치 "
                  + $"(원본 {templates.Count} 종 복제, {file.building})</color>");
    }

    private List<GameObject> FindTemplates()
    {
        List<GameObject> found = new List<GameObject>();
        foreach (GameObject root in gameObject.scene.GetRootGameObjects())
        {
            if (root != null
                && root.name.StartsWith(templateNamePrefix, StringComparison.OrdinalIgnoreCase))
            {
                found.Add(root);
            }
        }
        return found;
    }

    private GameObject SpawnOne(GameObject template, Spawn s)
    {
        Vector3 pos = new Vector3(s.unity_position[0], s.unity_position[1], s.unity_position[2]);
        float scale = s.unity_scale > 0.001f ? s.unity_scale : template.transform.localScale.x;

        if (snapToFloor
            && Physics.Raycast(pos + Vector3.up * snapSearchUp, Vector3.down,
                               out RaycastHit hit, snapSearchUp + snapSearchDown,
                               ~0, QueryTriggerInteraction.Ignore))
        {
            pos.y = hit.point.y;
        }

        GameObject person = Instantiate(template, pos, Quaternion.Euler(0f, s.unity_yaw, 0f));
        person.name = $"person_{s.room}_{s.index}";
        person.transform.localScale = Vector3.one * scale;
        person.SetActive(true);

        // 같은 사람을 두 번 세지 않게 하는 id. 손으로 붙일 일이 없도록 여기서 단다.
        if (person.GetComponent<PersonTarget>() == null)
        {
            person.AddComponent<PersonTarget>();
        }

        if (disableShadows)
        {
            foreach (Renderer r in person.GetComponentsInChildren<Renderer>(true))
            {
                r.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            }
        }
        return person;
    }
}
