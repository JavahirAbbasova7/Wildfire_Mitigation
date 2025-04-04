import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, OrderedDict, Tuple, TypeVar, TYPE_CHECKING

import gymnasium as gym
import numpy as np
from ray.rllib.utils.typing import ResultDict
from simfire.sim.simulation import Simulation
from simharness2.environments.utils import create_fire_simulation_from_config

if TYPE_CHECKING:
    from simharness2.environments.utils import RLlibEnvContextMetadata

logger = logging.getLogger(__name__)
# TODO: Remove `AnySimulation` and generics from environments subpkg.
AnySimulation = TypeVar("AnySimulation", bound=Simulation)


class Harness(gym.Env, ABC, Generic[AnySimulation]):
    def __init__(
        self,
        *,
        sim_init_cfg: Dict[str, Any],
        attributes: List[str],
        normalized_attributes: List[str],
        in_evaluation: bool = False,
        **kwargs,
    ):
        # Use provided simulation info to create a simulation object.
        self.sim = create_fire_simulation_from_config(sim_init_cfg)

        self.attributes = attributes
        # TODO: Maybe use `attributes_to_normalize` over `normalized_attributes`?
        self.normalized_attributes = normalized_attributes

        if not set(self.normalized_attributes).issubset(self.attributes):
            raise AssertionError(
                f"All normalized attributes ({str(self.normalized_attributes)}) must be "
                f"in attributes ({str(self.attributes)})!"
            )

        self.sim_attributes, self.nonsim_attributes = self._separate_sim_nonsim()

        # Count total timesteps that have occurred within an episode.
        self.timesteps = 0

        # Evaluation specific attributes.
        logger.info(f"Harness in evaluation mode: {in_evaluation}")
        self.in_evaluation = in_evaluation

        # Used to store recent episode results collected by Tune.
        self.current_result: ResultDict = {}

    @abstractmethod
    def create_agents(self):
        """Create agents for the simulation."""
        raise NotImplementedError

    @abstractmethod
    def get_nonsim_attribute_data(self) -> OrderedDict[str, np.ndarray]:
        """Get data that does not come from the simulation."""
        raise NotImplementedError

    @abstractmethod
    def get_nonsim_attribute_bounds(self) -> OrderedDict[str, Dict[str, int]]:
        """Get bounds for data that does not come from the simulation."""
        raise NotImplementedError

    @abstractmethod
    def get_harness_to_sim_action_map(self) -> Dict[int, int]:
        """Get the mapping from harness actions to sim actions."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def supported_attributes() -> List[str]:
        """Return the full list of attributes supported by the harness."""
        raise NotImplementedError

    @property
    def trial_logdir(self) -> str:
        """The path to the directory where (tune) trial results will be stored."""
        return self._trial_logdir

    @trial_logdir.setter
    def trial_logdir(self, path: str):
        if not os.path.isdir(path):
            raise ValueError(f"{path} is not a valid directory.")
        self._trial_logdir = path

    @property
    def rllib_env_context(self) -> "RLlibEnvContextMetadata":
        """The extra metadata that RLlib passes to the environment.

        The attributes of the returned object can be used to parameterize environments
        per process. For example, `worker_index` can be used to control which data file
        an environment reads in on initialization.
        """
        # Error handling for when the context is not set.
        if not hasattr(self, "_rllib_env_context"):
            raise AttributeError(
                "The RLlib environment context has not been set. This context is "
                "required for parameterizing each environment. Please ensure that the "
                "context is set before accessing it. This can be done by calling the "
                "`set_harness_env_context()` method defined in "
                "`simharness2/environments/utils.py`. For example usage, please refer to "
                "the `InitializeSimfire._set_rllib_context_foreach_env()` method in "
                "`simharness2/callbacks/initialize_simfire.py`."
            )
        return self._rllib_env_context

    @rllib_env_context.setter
    def rllib_env_context(self, context: "RLlibEnvContextMetadata"):
        self._rllib_env_context = context

    @property
    def num_eval_iters(self) -> int:
        """The number of times the environment has been evaluated."""
        if not self.in_evaluation:
            raise AssertionError(
                "The number of evaluation iterations can only be accessed when the "
                "environment is in evaluation mode."
            )

        # Initialize the number of evaluation iterations if it has not been set.
        if not hasattr(self, "_num_eval_iters"):
            logger.info("Setting initial value of `self.num_eval_iters` to -1.")
            self._num_eval_iters = -1

        return self._num_eval_iters

    @num_eval_iters.setter
    def num_eval_iters(self, value: int):
        if not self.in_evaluation:
            raise AssertionError(
                "The number of evaluation iterations can only be set when the "
                "environment is in evaluation mode."
            )

        self._num_eval_iters = value

    def _separate_sim_nonsim(self) -> Tuple[List[str], List[str]]:
        """Separate attributes based on if they are supported by the Simulation or not."""
        sim_attributes = self.sim.get_attribute_data()
        sim_attributes_list = []
        nonsim_attributes_list = []
        for attribute in self.attributes:
            if attribute not in sim_attributes.keys():
                nonsim_attributes_list.append(attribute)
            else:
                sim_attributes_list.append(attribute)
        return sim_attributes_list, nonsim_attributes_list

    def _normalize_obs(
        self, observations: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Convert an observation to the [0,1] range based on known min and max."""

        def normalize(data, min_max):
            # FIXME: Explain purpose/intention behind using a nested class here.
            return (data - min_max["min"]) / (min_max["max"] - min_max["min"])

        for attribute in self.normalized_attributes:
            observations[attribute] = normalize(
                observations[attribute], self.min_maxes[attribute]
            )

        return observations

    # FIXME: Does not use any instance or class attributes, make staticmethod?
    def _select_from_dict(
        self, dictionary: OrderedDict[str, Any], selections: List[str]
    ) -> OrderedDict[str, Any]:
        """Create an ordered subset with only specific keys from the input `dictionary`.

        Arguments:
            dictionary: A dictionary used to extract values from.
            selections: A list containing the desired keys to keep from `dictionary`.

        Returns:
            An ordered dictionary containing a subset of the input `dictionary`.
        """
        return_dict = OrderedDict()

        for selection in selections:
            return_dict[selection] = dictionary[selection]

        return return_dict

    # FIXME: Update name and make property?
    def _increment_evaluation_iterations(self) -> int:
        """Increment the number of calls to `Algorithm.evaluate()` (rllib)."""
        self.num_eval_iters += 1
        return self.num_eval_iters

    def _log_env_reset(self):
        """Log information about the environment that is being reset."""
        # if not self._debug_mode or self._episodes_debugged > self._debug_duration:
        #     return

        # TODO: What log level should we use here?
        # for idx, feat in enumerate(self.attributes):
        #     low, high = self._low[..., idx].min(), self._high[..., idx].max()
        #     obs_min = round(self.state[..., idx].min(), 2)
        #     obs_max = round(self.state[..., idx].max(), 2)
        # Log lower bound of the (obs space) and max returned obs for each attribute.
        #     logger.info(f"{feat} LB: {low}, obs min: {obs_min}")
        #     # Log upper (lower) bounds of the returned observations for each attribute.
        #     logger.info(f"{feat} UB: {high}, obs max: {obs_max}")
        pass

    def _log_env_init(self):
        """Log information about the environment that is being initialized."""
        # if self._is_eval_env:
        #     i, j = self.worker_idx, self.vector_idx
        # logger.warning(
        #     f"Object {hex(id(self))}: index (i+1)*(j+1) == {(i+1)*(j+1)}"
        #     )

        # if not self._debug_mode:
        #     return

        # # TODO: What log level should we use here?
        # logger.info(f"Object {hex(id(self))}: worker_index: {self.worker_idx}")
        # logger.info(f"Object {hex(id(self))}: vector_index: {self.vector_idx}")
        # logger.info(f"Object {hex(id(self))}: num_workers: {self.num_workers}")
        # logger.info(f"Object {hex(id(self))}: is_remote: {self.is_remote}")
        pass

    def _set_debug_options(self) -> None:
        """Set debug options for the simulation."""
        # self._debug_mode = config.get("debug_mode", False)
        # # unit == episodes
        # self._debug_duration = config.get("debug_duration", 1)
        # self._episodes_debugged = 0
        # logger.debug(f"Initializing environment {hex(id(self))}")
        pass


# TODO: Move to a more general file? maybe return a Set?
def get_unsupported_attributes(attributes_provided: List[str], cls: Harness) -> List[str]:
    """Return provided attributes that are NOT supported by class_name (a harness)."""
    return list(set(attributes_provided).difference(set(cls.supported_attributes())))
