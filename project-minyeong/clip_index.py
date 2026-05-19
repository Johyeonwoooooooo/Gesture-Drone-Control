# clip_index.py
import numpy as np
import pickle
import torch
import open_clip

DET_FILE = '/data1/workspaces/jgshin22/Gesture-Drone-Control/unidet3d/data/detections.pkl'
OUT_FILE = '/data1/workspaces/jgshin22/Gesture-Drone-Control/unidet3d/data/clip_index.pkl'

def main():
    det = pickle.load(open(DET_FILE, 'rb'))
    classes    = det['classes']
    labels     = det['labels']       # (M,)
    scores     = det['scores']
    bboxes     = det['bboxes']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', device=device)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model.eval()

    # 클래스 이름 → CLIP 텍스트 임베딩
    class_tokens = tokenizer(
        [f'a photo of a {c}' for c in classes]
    ).to(device)

    with torch.no_grad():
        class_embeds = model.encode_text(class_tokens)          # (C, 512)
        class_embeds = class_embeds / class_embeds.norm(dim=-1, keepdim=True)

    # 각 bbox → 해당 클래스 임베딩 할당
    box_embeds = class_embeds[torch.tensor(labels, device=device)]  # (M, 512)
    box_embeds = box_embeds.cpu().numpy()

    index = dict(
        box_embeds = box_embeds,    # (M, 512)
        bboxes     = bboxes,
        scores     = scores,
        labels     = labels,
        classes    = classes,
        points     = det['points'],
        box_pts    = det['box_pts'],
    )
    pickle.dump(index, open(OUT_FILE, 'wb'))
    print(f'Built CLIP index for {len(bboxes)} boxes → {OUT_FILE}')

if __name__ == '__main__':
    main()
