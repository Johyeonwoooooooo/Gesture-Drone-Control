"""rl_policy.py — 학습된 SAC 정책(actor)을 numpy 로만 돌린다.

`playground/reinforce_learning/model_geo_best.zip` 의 actor 는 21 → 256 → 256 →
mu(3) MLP 하나(7.3만 파라미터)뿐이다. 결정론 행동은 `tanh(mu)` 라서 torch 도
stable-baselines3 도 필요 없다 — 가중치 4쌍을 npz 로 뽑아 두면 이 파일이
numpy 로 같은 값을 낸다. **로컬 PC 에 torch 를 들이지 않는다는 저장소 성질을
지키기 위한 것**이고, 그래서 뽑는 쪽(`--export`)만 conda env `tello` 에서 한 번
돌리고 순찰 파이프라인은 뽑아 놓은 npz 만 읽는다.

    conda activate tello                      # torch + SB3 있는 환경
    python simulator/bridge/rl_policy.py --export

    conda activate patrol                     # numpy 만
    python simulator/bridge/rl_policy.py --check    # 자가 검증

`--export` 는 무작위 관측 64개에 대한 torch actor 의 출력도 같이 저장한다.
`--check` 는 그 기준값을 numpy 구현이 재현하는지 확인한다 — 관측 규격이나 층
구성이 어긋나면 여기서 걸린다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = _REPO / "playground" / "reinforce_learning" / "model_geo_best.zip"
DEFAULT_NPZ = _REPO / "simulator" / "bridge" / "geo_actor.npz"

OBS_DIM = 21    # 목표 단위벡터 3 + 정규화 거리 1 + 레이 14 + 직전 행동 3
ACT_DIM = 3     # 속도 벡터 (스텝당 max_step 만큼 이동)


class NumpyActor:
    """SAC actor 의 결정론 정책. `tanh(mu(obs))`, 활성함수는 ReLU."""

    def __init__(self, npz_path: str | Path = DEFAULT_NPZ) -> None:
        d = np.load(str(npz_path))
        self.w0, self.b0 = d["w0"], d["b0"]
        self.w1, self.b1 = d["w1"], d["b1"]
        self.wmu, self.bmu = d["wmu"], d["bmu"]
        if self.w0.shape[1] != OBS_DIM or self.wmu.shape[0] != ACT_DIM:
            raise ValueError(f"actor 규격 불일치: {self.w0.shape} / {self.wmu.shape}")
        self.ref_obs = d["ref_obs"] if "ref_obs" in d else None
        self.ref_act = d["ref_act"] if "ref_act" in d else None

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32).reshape(-1, OBS_DIM)
        x = np.maximum(x @ self.w0.T + self.b0, 0.0)
        x = np.maximum(x @ self.w1.T + self.b1, 0.0)
        a = np.tanh(x @ self.wmu.T + self.bmu)
        return a[0] if np.ndim(obs) == 1 else a


def export(model_zip: Path, out_npz: Path, n_ref: int = 64) -> None:
    """SB3 zip → npz. torch 가 있는 환경(`tello`)에서 한 번만 돌린다."""
    import sys
    sys.path.insert(0, str(_REPO / "playground" / "reinforce_learning"))
    import asym_policy  # noqa: F401  SAC.load 가 AsymSACPolicy 를 찾게
    import torch as th
    from stable_baselines3 import SAC

    model = SAC.load(str(model_zip).replace(".zip", ""), device="cpu")
    p = dict(model.policy.actor.named_parameters())
    w = {k: v.detach().numpy().astype(np.float32) for k, v in p.items()}

    rng = np.random.default_rng(0)
    ref_obs = rng.uniform(-1.0, 1.0, size=(n_ref, OBS_DIM)).astype(np.float32)
    with th.no_grad():
        # actor.forward(deterministic=True) 가 곧 tanh(mu) 다. 관측은 Dict 이지만
        # SensorOnlyExtractor 가 'sensor' 만 통과시키므로 priv 는 0 으로 채운다.
        obs_t = {"sensor": th.as_tensor(ref_obs),
                 "priv": th.zeros((n_ref, 4), dtype=th.float32)}
        ref_act = model.policy.actor(obs_t, deterministic=True).numpy()

    np.savez(out_npz,
             w0=w["latent_pi.0.weight"], b0=w["latent_pi.0.bias"],
             w1=w["latent_pi.2.weight"], b1=w["latent_pi.2.bias"],
             wmu=w["mu.weight"], bmu=w["mu.bias"],
             ref_obs=ref_obs, ref_act=ref_act.astype(np.float32))
    print(f"[rl] {model_zip.name} → {out_npz}  "
          f"({sum(v.size for v in w.values()):,} params, 기준값 {n_ref}개 포함)")


def check(npz_path: Path) -> None:
    """numpy 구현이 torch actor 의 기준 출력을 재현하는지."""
    actor = NumpyActor(npz_path)
    assert actor.ref_obs is not None, "기준값이 없는 npz — --export 로 다시 뽑으세요"
    got = actor(actor.ref_obs)
    err = float(np.abs(got - actor.ref_act).max())
    print(f"[rl] numpy vs torch 최대 오차 {err:.2e} ({len(actor.ref_obs)}개 관측)")
    assert err < 1e-5, f"정책 출력이 다릅니다 (오차 {err})"
    print("[rl] OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--export", action="store_true", help="SB3 zip → npz (env tello)")
    ap.add_argument("--check", action="store_true", help="numpy 구현 자가 검증")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    args = ap.parse_args()
    if args.export:
        export(args.model, args.npz)
    check(args.npz)


if __name__ == "__main__":
    main()
