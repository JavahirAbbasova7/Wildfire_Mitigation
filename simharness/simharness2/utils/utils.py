from typing import TYPE_CHECKING
import logging
from itertools import chain
from collections import namedtuple

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.evaluation.worker_set import WorkerSet


logger = logging.getLogger(__name__)
EnvCounts = namedtuple("EnvCounts", ["total_envs", "envs_per_worker"])


def validate_evaluation_config(algo_cfg: "AlgorithmConfig"):
    """Ensure the configuration of evaluation workers is valid.

    Arguments:
        algo_cfg: The rllib `AlgorithmConfig` instance.

    Raises:
        ValueError: If the evaluation duration unit is not `episodes`.
        ValueError: If the number of evaluation workers is 0.
        ValueError: If the evaluation duration is greater than the number of
            evaluation workers.
        ValueError: If the evaluation `WorkerSet` uses a local worker.
        ValueError: If the evaluation `WorkerSet` uses sub-environments (is vectorized).

    """
    num_eval_workers = 1
    eval_duration = algo_cfg.evaluation_duration
    eval_duration_unit = algo_cfg.evaluation_duration_unit

    use_train_local_worker = algo_cfg.create_env_on_local_worker
    use_eval_local_worker = algo_cfg.evaluation_config.get("create_env_on_local_worker")
    envs_per_train_worker = 1
    envs_per_eval_worker = algo_cfg.evaluation_config.get("num_envs_per_worker")

    if eval_duration_unit != "episodes":
        msg = "The `evaluation_duration_unit` must be set to `episodes`."
        raise ValueError(msg)
    # TODO: Handle `num_eval_workers == 0` edge case.
    elif num_eval_workers == 0:
        msg = "The `evaluation_num_workers` must be greater than 0."
        raise ValueError(msg)
    # TODO: Handle `eval_duration` greater than `evaluation_num_workers` edge case.
    elif eval_duration / num_eval_workers > 1:
        msg = "The `evaluation_duration` cannot be greater than `evaluation_num_workers`."
    # The logic gets weird if eval envs use a local worker; don't allow this for now.
    # TODO: Handle `use_eval_local_worker` edge case.
    elif use_train_local_worker and (
        use_eval_local_worker is None or use_eval_local_worker
    ):
        # Error if value is unspecified in evaluation config or set to True.
        if use_eval_local_worker is None:
            msg = (
                "The `create_env_on_local_worker` must be set to `False` for evaluation "
                "environments. This is unspecified, so we inherit the training workers "
                f"value for `create_env_on_local_worker`: {use_train_local_worker}. To "
                "override this behavior, set "
                "`evaluation.evaluation_config.create_env_on_local_worker` to `False`."
            )
        else:
            msg = (
                "The `create_env_on_local_worker` must be set to `False` for evaluation "
                f"environments, got: {use_eval_local_worker}. Update the value of "
                "`evaluation.evaluation_config.create_env_on_local_worker` to `False`."
            )
        raise ValueError(msg)
    # The logic gets weird if eval envs use sub-envs; don't allow this for now.
    # TODO: Handle `envs_per_eval_worker > 1` edge case.
    elif envs_per_train_worker > 1 and (
        envs_per_eval_worker is None or envs_per_eval_worker > 1
    ):
        # Error if unspecified in evaluation config or set to > 1.
        if envs_per_eval_worker is None:
            msg = (
                "The `num_envs_per_worker` must be set to 1 for evaluation "
                "workers. This is unspecified, so we inherit the training workers value "
                f"for `num_envs_per_worker`: {envs_per_train_worker}. To override this "
                "behavior, set `evaluation.evaluation_config.num_envs_per_worker` to 1."
            )
        else:
            msg = (
                "The `num_envs_per_worker` must be set to 1 for evaluation "
                f"workers, got: {envs_per_eval_worker}. Update the value of "
                "`evaluation.evaluation_config.num_envs_per_worker` to 1."
            )
        raise ValueError(msg)


def validate_rollouts_config(algo_cfg: "AlgorithmConfig"):
    """Ensure the configuration of rollout workers is valid."""
    num_train_workers = 1
    use_train_local_worker = algo_cfg.create_env_on_local_worker

    if num_train_workers > 0 and use_train_local_worker:
        msg = (
            "When num_rollout_workers > 0, the driver (local_worker; worker-idx=0) does "
            "not need an environment. This is because it doesn't have to sample (done "
            "by remote_workers; worker_indices > 0) nor evaluate (done by evaluation "
            "workers)."
        )
        raise ValueError(msg)


def get_total_envs(worker_set: "WorkerSet") -> EnvCounts:
    """Return the total number of envs in the respective WorkerSet.

    The `foreach_env` call will return a nested list. Each index contains a list
    of sub-environments contained within the respective worker. The `*envs` syntax
    expands the nested list into individual args to `chain()`, which concatenates them
    together into one long list, and `len()` returns its length.

    Arguments:
        worker_set: The rllib `WorkerSet` instance.

    Returns:
        A `namedtuple` that contains the total number of envs and the number of envs
        per worker.
    """

    envs = worker_set.foreach_env(lambda env: env)
    total_envs = len(list(chain(*envs)))
    envs_per_worker = max([len(worker_envs) for worker_envs in envs])

    return EnvCounts(total_envs=total_envs, envs_per_worker=envs_per_worker)


def has_local_worker(algo_cfg: "AlgorithmConfig") -> bool:
    """Return whether the respective WorkerSet uses a local worker.

    Arguments:
        algo_cfg: The rllib `AlgorithmConfig` instance, either for the training or
            evaluation WorkerSet.

    Returns:
        A boolean that indicates if the respective WorkerSet uses a local worker.
    # """
    # num_workers = algo_cfg.num_rollout_workers
    num_workers = 1
    use_local_worker = algo_cfg.create_env_on_local_worker
    if num_workers == 0 or use_local_worker:
        return True
    elif num_workers > 0 and not use_local_worker:
        return False
    else:
        msg = (
            "The number of rollout workers must be greater than or equal to 0, and the "
            "`create_env_on_local_worker` must be a boolean value."
        )
        raise ValueError(msg)


def get_default_seed() -> int:
    """Return the default seed value for reproducibility.

    The default seed is arbitrary, and was chosen by generating a random 128-bit number:
      import secrets
      secrets.randbits(128)
    """
    return 223203939272683461891947593725839432272
