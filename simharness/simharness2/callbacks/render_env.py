"""Callback for rendering gifs during evaluation."""

import logging
import os
from math import log
from typing import List, TYPE_CHECKING, Dict, Optional, Union
import time
from itertools import chain

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID, ResultDict  # AgentID, EnvType,

import simharness2.utils.utils as utils


if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.evaluation.worker_set import WorkerSet
    from simfire.sim.simulation import FireSimulation

    from simharness2.environments.fire_harness import FireHarness

logger = logging.getLogger(__name__)

TRAIN_KEY = "train"
EVAL_KEY = "evaluation"
# TODO: Add a config option to control rendering settings.
# Switch to enable rendering of training environments.
RENDER_TRAIN_ENVS = False
# NOTE: Probably better to use a dictionary so that "eval" and "train" are not forced to
# use the same interval setup, but good enough for the time being. When this update is
# added, the logic in RenderEnv.should_render_env will need to be updated accordingly.
# Options: "log" or "linear"
RENDER_INTERVAL_TYPE = "linear"
# Set the base for the logarithmic interval
LOGARITHMIC_BASE = 10
# Set the step size for the linear interval
LINEAR_INTERVAL_STEP = 1  # 0


class RenderEnv(DefaultCallbacks):
    """To use this callback, set {"callbacks": RenderEnv} in the algo config."""

    def __init__(self, legacy_callbacks_dict: Dict[str, callable] = None):
        super().__init__()
        # Utilities used throughout methods in the callback.
        self.render_current_episode = False
        self.curr_iter = -1
        logger.info("RenderEnv callback initialized.")

    def on_algorithm_init(
        self,
        *,
        algorithm: "Algorithm",
        **kwargs,
    ) -> None:
        """Callback run when a new algorithm instance has finished setup.

        This method gets called at the end of Algorithm.setup() after all
        the initialization is done, and before actually training starts.

        Args:
            algorithm: Reference to the Algorithm instance.
            kwargs: Forward compatibility placeholder.
        """
        utils.validate_evaluation_config(algorithm.config)
        self.has_local_eval_worker = utils.has_local_worker(algorithm.evaluation_config)
        self.has_local_train_worker = utils.has_local_worker(algorithm.config)

        # Make the trial result path accessible to each env (for gif saving).
        logdir = algorithm.logdir
        # workers = algorithm.workers  # If workers is a function that returns workers
        # logger.info(workers)  # Log the type of the return value

        # # Then if the returned object has foreach_worker:
        algorithm.env_runner_group.foreach_worker(
            lambda w: w.foreach_env(lambda env: setattr(env, "trial_logdir", logdir)),
            # local_worker=self.has_local_train_worker,
        )
        algorithm.eval_env_runner_group.foreach_worker(
            lambda w: w.foreach_env(lambda env: setattr(env, "trial_logdir", logdir)),
            # local_worker=self.has_local_eval_worker,
        )

    # def _set_trial_logdir_foreach_env(self, worker_set:)

    def on_episode_created(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        # policies: Dict[PolicyID, Policy],
        # episode: Union[Episode, EpisodeV2],
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Callback run right after an Episode has started.

        This method gets called after the Episode(V2)'s respective sub-environment's
        (usually a gym.Env) `reset()` is called by RLlib.

        1) Episode(V2) created: Triggers callback `on_episode_created`.
        2) Respective sub-environment (gym.Env) is `reset()`.
        3) Episode(V2) starts: This callback fires.
        4) Stepping through sub-environment/episode commences.

        Args:
            worker: Reference to the current rollout worker.
            base_env: BaseEnv running the episode. The underlying
                sub environment objects can be retrieved by calling
                `base_env.get_sub_environments()`.
            policies: Mapping of policy id to policy objects. In single
                agent mode there will only be a single "default" policy.
            episode: Episode object which contains the episode's
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            env_index: The index of the sub-environment that started the episode
                (within the vector of sub-environments of the BaseEnv).
            kwargs: Forward compatibility placeholder.
        """
        env: FireHarness[FireSimulation] = base_env.get_sub_environments()[env_index]
        env_type = EVAL_KEY if worker.config.in_evaluation else TRAIN_KEY
        self.render_current_episode = self.should_render_env(env, env_type)
        if self.render_current_episode:
            # NOTE: Weird rllib behavior - v_idx always 0 when num_rollout_workers == 0.
            w_idx = worker.env_context.worker_index
            v_idx = env_index
            logger.info(
                f"Preparing to render {env_type} environment (w: {w_idx}, v: {v_idx})..."
            )
            env._configure_env_rendering(True)

            # TODO: Look into save_history behavior in analytics module.
            env.harness_analytics.reset(
                env_is_rendering=True,
                reset_benchmark=env._new_fire_scenario,
            )

    def should_render_env(
        self, env: "FireHarness[FireSimulation]", env_type: str
    ) -> bool:
        """Check if the environment should be rendered."""
        if env_type == TRAIN_KEY:
            if not env.current_result:
                logger.info("No current_result, setting current iteration to 0...")
                self.curr_iter = 0
            else:
                self.curr_iter = env.current_result["training_iteration"]
        else:
            logger.info(f"Current evaluation iteration: {env.num_eval_iters}")
            self.curr_iter = env.num_eval_iters

        logger.debug(f"Current iteration for {env_type}: {self.curr_iter}")
        if env_type == TRAIN_KEY and RENDER_TRAIN_ENVS or env_type == EVAL_KEY:
            # Use specified interval type to determine if the env should be rendered.
            if RENDER_INTERVAL_TYPE == "log":
                # NOTE: +1 to avoid log(0) and to ensure the first iteration is rendered.
                value = log(self.curr_iter + 1, LOGARITHMIC_BASE)
                return value.is_integer() and value > 0
            elif RENDER_INTERVAL_TYPE == "linear":
                return self.curr_iter % LINEAR_INTERVAL_STEP == 0

    def on_episode_end(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[EpisodeV2, Exception],
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Runs when an episode is done.

        Args:
            worker: Reference to the current rollout worker.
            base_env: BaseEnv running the episode. The underlying
                sub environment objects can be retrieved by calling
                `base_env.get_sub_environments()`.
            policies: Mapping of policy id to policy
                objects. In single agent mode there will only be a single
                "default_policy".
            episode: Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
                In case of environment failures, episode may also be an Exception
                that gets thrown from the environment before the episode finishes.
                Users of this callback may then handle these error cases properly
                with their custom logics.
            env_index: The index of the sub-environment that ended the episode
                (within the vector of sub-environments of the BaseEnv).
            kwargs: Forward compatibility placeholder.
        """
        env: FireHarness[FireSimulation] = base_env.get_sub_environments()[env_index]
        env_type = EVAL_KEY if worker.config.in_evaluation else TRAIN_KEY
        # FIXME: Condition is overkill, but ensures callback, env, and env.sim are
        # "on the same page".
        if self.render_current_episode and env._should_render and env.sim.rendering:
            logdir = env.trial_logdir
            # NOTE: Weird rllib behavior - v_idx always 0 when num_rollout_workers == 0.
            w_idx = worker.env_context.worker_index
            v_idx = env_index

            # Save a GIF from the last episode
            # Check if there is a gif "ready" to be saved
            # FIXME Update logic to handle saving same gif when writing to Aim UI
            context_dict = {}
            # FIXME: Should we round lat, lon to a certain precision??
            if env.sim.config.landfire_lat_long_box:
                lat, lon = env.sim.config.landfire_lat_long_box.points[0]
                op_data_lat_lon = f"operational_lat_{lat}_lon_{lon}"
            else:
                op_data_lat_lon = "functional"
            fire_init_pos = env.sim.config.fire.fire_initial_position
            context_dict.update({"fire_initial_position": str(fire_init_pos)})
            # FIXME: Finalize path for saving gifs (and add note to docs) - for example,
            # save each gif in a folder that relates it to episode iter?
            current_time = time.strftime("%H%M%S")
            if env_type == EVAL_KEY:
                curr_iter = env.num_eval_iters
            else:
                curr_iter = self.curr_iter
            env_episode_id = f"iter_{curr_iter}_time_{current_time}_w_{w_idx}_v_{v_idx}"
            logdir = os.getenv("RAY_RESULTS_DIR", "/userdata/kerasData/jash/test/jash/Wildfire_Mitigation/simharness/gifs") 
            gif_save_path = os.path.join(
                logdir,
                env_type,
                "gifs",
                op_data_lat_lon,
                f"fire_init_pos_x_{fire_init_pos[0]}_y_{fire_init_pos[1]}",
                f"{env_episode_id}.gif",
            )
            
            logger.info(f"Total environment steps: {env.timesteps}")
            logger.info(f"Saving GIF to {gif_save_path}...")
            env.sim.save_gif(gif_save_path)
            # Save the gif_path so that we can write image to aim server, if desired
            # NOTE: `save_path` is a list after the above; do element access for now
            logger.debug(f"Type of gif_save_path: {type(gif_save_path)}")
            gif_data = {
                "path": gif_save_path,
                "name": op_data_lat_lon,
                "step": curr_iter,
                # "epoch":
                "context": context_dict,
            }
            episode.media.update({"gif_data": gif_data})

            # Try to collect and log episode history, if it was saved.
            if env.harness_analytics.sim_analytics.save_history:
                episode_history_dir = os.path.join(logdir, env_type)
                env.harness_analytics.save_sim_history(
                    episode_history_dir, env_episode_id
                )

            env._configure_env_rendering(False)

    def on_evaluate_start(
        self,
        *,
        algorithm: "Algorithm",
        **kwargs,
    ) -> None:
        """Callback before evaluation starts.

        This method gets called at the beginning of Algorithm.evaluate().

        Args:
            algorithm: Reference to the algorithm instance.
            kwargs: Forward compatibility placeholder.
        """
        # TODO: Add note in docs that the local worker IS NOT rendered. With this
        # assumption, we should always set `evaluation.evaluation_num_workers >= 1`.
        # TODO: Handle edge case where num_evaluation_workers == 0.
        # Increment the number of evaluation iterations
        logger.info("Incrementing evaluation iterations...")
        eval_iters = algorithm.eval_env_runner_group.foreach_worker(
            lambda w: w.foreach_env(lambda env: env._increment_evaluation_iterations()),
            # local_worker=self.has_local_eval_worker,
        )
        curr_eval_iter = {iter for iter in chain(*eval_iters)}
        if len(curr_eval_iter) > 1:
            logger.warning(f"Multiple evaluation iterations detected: {curr_eval_iter}.")
        elif len(curr_eval_iter) == 1:
            logger.info(f"Current evaluation iteration set to: {curr_eval_iter.pop()}")

    def on_train_result(
        self,
        *,
        algorithm: "Algorithm",
        result: ResultDict,
        **kwargs,
    ) -> None:
        """Called at the end of Algorithm.train().

        Args:
            algorithm: Current Algorithm instance.
            result: Dict of results returned from Algorithm.train() call.
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """
        # Update the current result for each environment.
        if result:
            logger.info("Updating current result for each environment...")
        else:
            logger.warning("No result to update for each environment...")

        algorithm.eval_env_runner_group.foreach_worker(
            lambda w: w.foreach_env(lambda env: setattr(env, "current_result", result)),
            # local_worker=self.has_local_train_worker,
        )

    def on_workers_recreated(
        self,
        *,
        algorithm: "Algorithm",
        worker_set: "WorkerSet",
        worker_ids: List[int],
        is_evaluation: bool,
        **kwargs,
    ) -> None:
        # Make the trial result path accessible to each env (for gif saving).
        logdir = algorithm.logdir
        if is_evaluation:
            has_local_worker = self.has_local_eval_worker
        else:
            has_local_worker = self.has_local_train_worker

        worker_set.env_runner_group.foreach_worker(
            lambda w: w.foreach_env(lambda env: setattr(env, "trial_logdir", logdir)),
            # local_worker=has_local_worker,
            remote_worker_ids=worker_ids,
        )
