import open3d as o3d
import numpy as np
import os
import json
import math

def setup_directories(base_dir="render_output"):
    """결과물을 저장할 폴더를 생성합니다."""
    img_dir = os.path.join(base_dir, "images")
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    return base_dir, img_dir

def render_pointcloud_mac_callback(ply_path, num_views=30):
    print(f"[{ply_path}] 파일을 로드 중입니다...")
    pcd = o3d.io.read_point_cloud(ply_path)
    
    if pcd.is_empty():
        print("포인트 클라우드를 불러오지 못했습니다. 경로를 확인해주세요.")
        return

    base_dir, img_dir = setup_directories()
    
    # 방해 공간 및 중심점 계산
    bbox = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    radius = max(extent) * 0.8  
    
    print(f"총 {num_views}장의 이미지를 렌더링합니다. (반경: {radius:.2f})")
    
    # Mac 충돌을 피하기 위한 상태(State) 딕셔너리
    state = {
        "view_idx": 0,
        "num_views": num_views,
        "radius": radius,
        "center": center,
        "extent": extent,
        "camera_data": [],
        "img_dir": img_dir,
        "done": False,
        "warmup_frames": 15 # 화면이 켜지고 15프레임 동안 대기 (안정화)
    }

    # 1. 렌더러 초기화
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=640, height=480, visible=True)
    vis.add_geometry(pcd)

    # 2. 콜백 함수 정의 (매 틱마다 Mac 엔진이 안전하게 호출함)
    def animation_callback(vis):
        if state["done"]:
            return False
            
        # 웜업 (화면이 완전히 뜰 때까지 대기)
        if state["warmup_frames"] > 0:
            state["warmup_frames"] -= 1
            return False

        idx = state["view_idx"]
        
        # 목표 장수 도달 시 종료
        if idx >= state["num_views"]:
            state["done"] = True
            vis.close() 
            return False
            
        # 카메라 위치(Eye) 계산
        angle = (idx / state["num_views"]) * 2.0 * math.pi
        eye_x = state["center"][0] + state["radius"] * math.cos(angle)
        eye_y = state["center"][1] + state["radius"] * math.sin(angle)
        eye_z = state["center"][2] + (state["extent"][2] * 0.2)
        
        eye = np.array([eye_x, eye_y, eye_z])
        lookat = state["center"]
        up = np.array([0, 0, 1])
        
        # Mac에서 안전한 카메라 조작 API 사용
        ctr = vis.get_view_control()
        front = eye - lookat
        front = front / np.linalg.norm(front) # 정규화
        
        ctr.set_lookat(lookat)
        ctr.set_front(front)
        ctr.set_up(up)
        
        # 이미지 캡처
        img_path = os.path.join(state["img_dir"], f"view_{idx:03d}.png")
        vis.capture_screen_image(img_path, do_render=True)
        
        # 카메라 파라미터 저장
        param = ctr.convert_to_pinhole_camera_parameters()
        extrinsic = np.asarray(param.extrinsic).tolist()
        intrinsic = np.asarray(param.intrinsic.intrinsic_matrix).tolist()
        
        state["camera_data"].append({
            "frame_id": idx,
            "image_path": f"images/view_{idx:03d}.png",
            "extrinsic": extrinsic,
            "intrinsic": intrinsic
        })
        
        print(f" - [view_{idx:03d}.png] 캡처 완료")
        state["view_idx"] += 1
        
        return True # 화면 업데이트 지시

    # 3. 콜백 등록 및 Mac 네이티브 이벤트 루프 실행 (핵심!)
    vis.register_animation_callback(animation_callback)
    vis.run() 
    vis.destroy_window()
    
    # 4. JSON 저장
    if len(state["camera_data"]) > 0:
        pose_file = os.path.join(base_dir, "camera_poses.json")
        with open(pose_file, 'w') as f:
            json.dump(state["camera_data"], f, indent=4)
        print(f"\n✅ 렌더링 완료! 결과물은 [{base_dir}] 폴더를 확인하세요.")
    else:
        print("\n❌ 렌더링에 실패했습니다.")

if __name__ == "__main__":
    TARGET_PLY = "playground/data/Scan at 21.39.ply"
    render_pointcloud_mac_callback(TARGET_PLY, num_views=30)