"""
asym_policy.py — Asymmetric actor-critic SAC 정책
──────────────────────────────────────────────────
특권 정보(측지거리 φ, 측지 하강방향)의 사용 경계를 '코드 구조'로 강제한다:

  actor  → obs['sensor'] 만 (목표벡터 + 레이 + 직전행동) — 배포되는 부분
  critic → obs['sensor'] + obs['priv'] — 학습 중에만 쓰이고 배포 안 됨

critic 이 진짜 측지거리를 알면 가치 추정이 정확해져 학습이 저분산·안정이 된다
(Pinto et al. 2017, asymmetric actor-critic). φ 는 어차피 보상 계산에 만들어
두므로 추가 비용이 거의 없다. "지도 어디서 났냐"는 질문에 대한 답:
시뮬레이터 특권 정보이고, critic 까지만 주고 actor 에서 끊었다.
"""
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import SACPolicy


class SensorOnlyExtractor(BaseFeaturesExtractor):
    """obs['sensor'] 만 통과 — actor 용. 학습 파라미터 없음."""
    def __init__(self, observation_space):
        super().__init__(observation_space,
                         features_dim=int(observation_space['sensor'].shape[0]))

    def forward(self, obs):
        return obs['sensor']


class SensorPrivExtractor(BaseFeaturesExtractor):
    """obs['sensor'] + obs['priv'] 연결 — critic 용. 학습 파라미터 없음."""
    def __init__(self, observation_space):
        n = int(observation_space['sensor'].shape[0] +
                observation_space['priv'].shape[0])
        super().__init__(observation_space, features_dim=n)

    def forward(self, obs):
        return th.cat([obs['sensor'], obs['priv']], dim=-1)


class AsymSACPolicy(SACPolicy):
    """actor=센서만 / critic=센서+특권. share_features_extractor 와 무관하게
       각자 전용 추출기를 강제 주입한다 (전달된 인자는 의도적으로 무시)."""

    def make_actor(self, features_extractor=None):
        return super().make_actor(
            features_extractor=SensorOnlyExtractor(self.observation_space))

    def make_critic(self, features_extractor=None):
        return super().make_critic(
            features_extractor=SensorPrivExtractor(self.observation_space))
