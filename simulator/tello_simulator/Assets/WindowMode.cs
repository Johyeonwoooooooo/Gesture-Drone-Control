using UnityEngine;

// 창 테두리(타이틀바) 제거.
//
// 웹 관제 화면은 이 창을 getDisplayMedia 로 화면 공유받아 <video> 로 그린다.
// 창 캡처는 타이틀바까지 같이 찍히기 때문에, 웹에서 전체 화면으로 띄우면
// 영상 위쪽에 "tello_simulator [ㅡ][ㅁ][X]" 바가 그대로 남는다.
// FullScreenMode.FullScreenWindow(테두리 없는 전체 화면)로 띄우면 캡처에
// 창 크롬이 아예 들어오지 않는다.
//
// ProjectSettings 의 fullscreenMode 는 이미 1(FullScreenWindow)이지만, Unity 는
// 플레이어가 마지막으로 쓴 창 모드를 레지스트리
// (HKCU\Software\<회사>\<제품>\Screenmanager Is Fullscreen mode)에 기억했다가
// 그걸 우선한다. 그래서 한 번 창 모드로 뜬 빌드는 계속 창 모드로 뜬다.
// 여기서 런타임에 덮어써 그 기억을 무시한다.
//
// 씬에 붙일 필요 없다 — RuntimeInitializeOnLoadMethod 로 스스로 올라온다.
// B 키로 창 모드와 오간다(데스크톱을 봐야 할 때).
public class WindowMode : MonoBehaviour
{
    // 시작할 때 테두리 없는 전체 화면으로 띄울지. false 면 저장된 모드를 따른다.
    public const bool StartBorderless = true;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void Bootstrap()
    {
        var go = new GameObject("WindowMode");
        DontDestroyOnLoad(go);
        go.AddComponent<WindowMode>();
    }

    void Start()
    {
        if (StartBorderless) SetBorderless(true);
    }

    void Update()
    {
        if (TogglePressed()) SetBorderless(!IsBorderless());
    }

    static bool IsBorderless()
    {
        return Screen.fullScreenMode == FullScreenMode.FullScreenWindow
            || Screen.fullScreenMode == FullScreenMode.ExclusiveFullScreen;
    }

    static void SetBorderless(bool on)
    {
        if (on)
        {
            // 주 디스플레이 해상도로. ExclusiveFullScreen 과 달리 포커스를 잃어도
            // 최소화되지 않아, 브라우저를 앞에 두고도 캡처가 계속 살아 있다.
            Screen.SetResolution(Display.main.systemWidth, Display.main.systemHeight,
                                 FullScreenMode.FullScreenWindow);
        }
        else
        {
            Screen.SetResolution(1280, 720, FullScreenMode.Windowed);
        }
    }

    // CameraFollow.TogglePressed 와 같은 이중 백엔드 가드
    // (Input System 패키지가 켜져 있으면 레거시 Input 은 예외를 던진다).
    static bool TogglePressed()
    {
#if ENABLE_INPUT_SYSTEM
        var kb = UnityEngine.InputSystem.Keyboard.current;
        if (kb != null && kb.bKey.wasPressedThisFrame) return true;
#endif
#if ENABLE_LEGACY_INPUT_MANAGER
        if (Input.GetKeyDown(KeyCode.B)) return true;
#endif
        return false;
    }
}
