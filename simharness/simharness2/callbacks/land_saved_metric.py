import logging
from typing import Dict

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy


logger = logging.getLogger(__name__)


class LandSavedMetric(DefaultCallbacks):
    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: EpisodeV2,
        env_index: int,
        **kwargs,
    ):
        env = base_env.get_sub_environments()[env_index]
        land_burned_benchmark = (
            env.harness_analytics.benchmark_sim_analytics.data.damaged[-1]
        )
        land_burned_sim = env.harness_analytics.sim_analytics.data.total_damaged
        episode.custom_metrics["land_saved"] = land_burned_benchmark - land_burned_sim
