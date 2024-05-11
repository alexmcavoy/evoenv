from abc import ABC, abstractmethod
from typing import Generic, Optional, Set, Tuple, TypeVar, Union

import numpy as np
from matplotlib.pyplot import Axes, Figure

ActType = TypeVar('ActType') # action type
ObsType = TypeVar('ObsType') # observation type

class EvolutionaryEnvironment(ABC, Generic[ActType, ObsType]):
	r'''
	Core class for evolutionary environments, which should be extended and implemented.
	'''
	
	def __init__(self, rng: Optional[np.random._generator.Generator] = None):
		r'''
		Initialize the environment.

		Parameters:
			rng (Optional[np.random._generator.Generator]): A random number generator (default seed is 12).
		'''
		self._rng: np.random._generator.Generator = rng if rng is not None else np.random.default_rng(12)
		self._has_been_reset: bool = False

	@abstractmethod
	def reset(self) -> Tuple[Union[ObsType, Tuple[ObsType, ...]], Union[Set[ActType], Tuple[Set[ActType], ...]]]:
		r'''
		Reset the environment to an initial state. Any method overriding reset must call the base method.

		Returns:
			Union[ObsType, Tuple[ObsType, ...]]: A tuple of observations of the initial state, one for each agent (or just one if all the same).
			Union[Set[ActType], Tuple[Set[ActType], ...]]]: Action sets, one for each player (or just one if all the same).
		'''
		self._has_been_reset = True

	@abstractmethod
	def step(self, actions: Tuple[ActType, ...]) -> Tuple[Union[ObsType, Tuple[ObsType, ...]], Union[Set[ActType], Tuple[Set[ActType], ...]], Tuple[float, ...], bool]:
		r'''
		Advance the environment (stochastic game) by one time step. Any method overriding step must call the base method.

		Parameters:
			actions (Tuple[ActType, ...]): A tuple of actions taken by the players.

		Returns:
			Union[ObsType, Tuple[ObsType, ...]]: A tuple of observations of the new state, one for each agent (or just one if all the same).
			Union[Set[ActType], Tuple[Set[ActType], ...]]: Action sets, one for each player (or just one if all the same).
			Tuple[float, ...]: A tuple of rewards, one for each agent.
			bool: Indicates whether a final state has been reached (True) or not (False).
		'''
		if self._has_been_reset is False:
			raise EnvironmentNotResetError

	@abstractmethod
	def render(self) -> Optional[Tuple[Figure, Axes]]:
		r'''
		Render a graphical representation of the environment and its state.

		Returns:
			Tuple[Figure, Axes] (optional): Tuple of a figure and axes showing the state of the environment.
		'''
		if self._has_been_reset is False:
			raise EnvironmentNotResetError

class EnvironmentNotResetError(Exception):
	r'''
	Custom error indicating that the user is trying to take a step before resetting the environment at least once.
	'''
	def __init__(self, message: str = 'The environment must be reset at least once.'):
		super(EnvironmentNotResetError, self).__init__(message)
