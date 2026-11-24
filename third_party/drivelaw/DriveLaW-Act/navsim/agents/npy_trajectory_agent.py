"""Agent that replays pre-computed trajectories from ``<token>.npy`` files (v1 devkit port of the
navsim-v2 NpyTrajectoryAgent). Lets any model's predictions be scored offline with the stock
run_pdm_score.py."""
from pathlib import Path
from typing import List

import numpy as np
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder


class NpyTrajectoryAgent(AbstractAgent):
    def __init__(self, pred_actions_path: str, trajectory_sampling: TrajectorySampling):
        super().__init__(requires_scene=True)
        self._pred_actions_path = Path(pred_actions_path)
        self._trajectory_sampling = trajectory_sampling

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        pass

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_no_sensors()

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return []

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return []

    def compute_trajectory(self, agent_input: AgentInput, scene=None) -> Trajectory:
        token = scene.scene_metadata.initial_token
        p = self._pred_actions_path / f"{token}.npy"
        if not p.exists():
            raise FileNotFoundError(f"missing trajectory for token {token}: {p}")
        poses = np.asarray(np.load(p), dtype=np.float32)
        if poses.ndim == 3 and poses.shape[0] == 1:
            poses = poses[0]
        assert poses.shape == (self._trajectory_sampling.num_poses, 3), poses.shape
        return Trajectory(poses=poses, trajectory_sampling=self._trajectory_sampling)
