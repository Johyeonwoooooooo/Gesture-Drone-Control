# clip_index.py — det .pkl → 박스별 CLIP 임베딩 .pkl, CLIP 1회 로드 후 모든 REGION 일괄
import os
import sys
import pickle
import torch
import clip

import scenes


def build_index(region, model, device):
    det_file = scenes.det_path(region)
    if not os.path.exists(det_file):
        print(f'[clip_index] skip {region}: {det_file} 없음 (infer 먼저 실행)')
        return

    det = pickle.load(open(det_file, 'rb'))
    classes = det['classes']
    labels = det['labels']       # (M,)
    scores = det['scores']
    bboxes = det['bboxes']

    # 클래스 이름 → CLIP 텍스트 임베딩
    class_tokens = clip.tokenize(
        [f'a photo of a {c}' for c in classes]
    ).to(device)

    with torch.no_grad():
        class_embeds = model.encode_text(class_tokens)          # (C, 512)
        class_embeds = class_embeds / class_embeds.norm(dim=-1, keepdim=True)

    # 각 bbox → 해당 클래스 임베딩 할당
    if len(labels) > 0:
        box_embeds = class_embeds[torch.tensor(labels, device=device)]  # (M, 512)
        box_embeds = box_embeds.cpu().numpy()
    else:
        box_embeds = class_embeds.new_zeros((0, class_embeds.shape[1])).cpu().numpy()

    index = dict(
        box_embeds=box_embeds,    # (M, 512)
        bboxes=bboxes,
        scores=scores,
        labels=labels,
        classes=classes,
        points=det['points'],
        box_pts=det['box_pts'],
    )
    out = scenes.index_path(region)
    pickle.dump(index, open(out, 'wb'))
    print(f'[clip_index] {region}: {len(bboxes)} boxes → {out}')


def main():
    regions = sys.argv[1:] or scenes.REGIONS
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = clip.load('ViT-B/32', device=device)
    for i, region in enumerate(regions):
        print(f'=== ({i + 1}/{len(regions)}) {region} ===')
        build_index(region, model, device)
    print('[clip_index] done.')


if __name__ == '__main__':
    main()
