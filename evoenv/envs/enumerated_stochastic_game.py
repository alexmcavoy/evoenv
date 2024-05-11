from typing import Dict, Optional, Set, Tuple
from evoenv.matrixtools import FloatTupleDtype

import evoenv
import numpy as np

ActType = int
ObsType = int

class EnumeratedStochasticGame(evoenv.EvolutionaryEnvironment[ActType, ObsType]):

	def __init__(
		self,
		rewards: Dict[int, np.ndarray],
		init_dist: Optional[np.ndarray] = None,
		transitions: Optional[Dict[int, np.ndarray]] = None,
		final_state: Optional[int] = None,
		rng: Optional[np.random._generator.Generator] = None
	):
		r'''
		Initialize a stochastic game whose states are explicitly enumerated, from 0 to n, where n+1 is the number of states.

		Parameters:
			rewards (Dict[int, np.ndarray]): A dictionary of reward structures, one for each state.
			init_dist (np.ndarray, optional): An initial distribution over the states for resetting the environment.
			transitions (Dict[int, np.ndarray], optional): A dictionary of transition probabilities.
			final_state (int, optional): A final state, which ends the game when entered.
			rng (np.random._generator.Generator, optional): A random number generator.
		'''
		super(EnumeratedStochasticGame, self).__init__(rng=rng)

		self.rewards = rewards

		if init_dist is None:
			# uniform distribution over states
			self.init_dist = np.ones_like(self._states, dtype=np.float_)
		else:
			self.init_dist = init_dist

		if transitions is None:
			# uniform distribution over states
			self.transitions = {state: np.ones((*self._rewards[state].shape, self._n_states)) for state in self._states}
		else:
			self.transitions = transitions

		self.final_state = final_state

	@property
	def rewards(self) -> Dict[int, np.ndarray]:
		return self._rewards

	@rewards.setter
	def rewards(self, rewards: Dict[int, np.ndarray]):
		try:
			# both rewards and transitions were defined previously, and rewards are now changed
			if rewards.keys() != self._transitions.keys():
				raise ValueError(f'The states in the transition structure must equal those of the reward structure.')
			bad_states: Set[int] = set()
			
			for state in rewards.keys():
				if rewards[state].shape != self._transitions[state].shape[:-1]:
					# for each state, the actions available to each player must be consistent with the transitions
					bad_states.add(state)
			
			if len(bad_states) > 0:
				raise ValueError(f'The rewards in states {bad_states} are not consistent with the pre-existing transition structure.')

		except AttributeError:
			# rewards and transitions are both being defined for the first time
			if len(rewards) == 0:
				raise ValueError(f'There must be at least one state in the game.')

			elif any(k not in range(len(rewards)) for k in rewards.keys()):
				raise ValueError(f'All state labels must be in the set {{0,...,{len(rewards)-1}}}.')
			
			n_agents = set(matrix.ndim for matrix in rewards.values())
			if len(n_agents) > 1:
				raise ValueError('All reward matrices must have the same dimension.')

			self._n_agents: int = n_agents.pop()
			self._states: np.ndarray = np.arange(len(rewards), dtype=np.int_)
			self._n_states: int = len(self._states)
			
		finally:
			if hasattr(self, '_n_agents'):
				matrix_types = set(matrix.dtype for matrix in rewards.values())
				if len(matrix_types)>1 or FloatTupleDtype(self._n_agents) not in matrix_types:
					raise ValueError(f'All reward matrices must have the data type FloatTupleDtype({self._n_agents}).')

			self._rewards = rewards

	@property
	def init_dist(self) -> np.ndarray:
		return self._init_dist

	@init_dist.setter
	def init_dist(self, init_dist: np.ndarray):
		if init_dist.size != self._n_states or not (np.all(init_dist >= np.array(0)) and np.any(init_dist > np.array(0))):
			raise ValueError(f'The initial distribution must have {self._n_states} non-negative weights (not all zero).')

		self._init_dist = init_dist.flatten()
		self._init_dist = np.divide(self._init_dist, np.sum(self._init_dist))

	@property
	def transitions(self) -> Dict[int, np.ndarray]:
		return self._transitions

	@transitions.setter
	def transitions(self, transitions: Dict[int, np.ndarray]):
		if transitions.keys() != self._rewards.keys():
			raise ValueError(f'The states in the transition structure must equal those of the reward structure.')

		bad_states = set()
		for state in transitions.keys():
			if transitions[state].shape[:-1] != self._rewards[state].shape:
				# for each state, the actions available to each player must be consistent with the rewards
				bad_states.add(state)
			
			elif transitions[state].shape[-1] != self._n_states:
				# the number of weights must equal the number of states
				bad_states.add(state)
			
			elif not (np.all(transitions[state] >= np.array(0)) and (np.all(np.sum(transitions[state], axis=-1) > np.array(0)))):
				# for each state and action profile, the weights must be non-negative and not all zero
				bad_states.add(state)
		
		if len(bad_states) > 0:
			raise ValueError(f'The states in {bad_states} do not have {self._n_states} non-negative weights (not all zero) for every possible action profile.')

		self._transitions = transitions

		for state in self._transitions.keys():
			self._transitions.update({state: np.divide(self._transitions[state], np.sum(self._transitions[state], axis=-1, keepdims=True))})

	@property
	def final_state(self) -> Optional[int]:
		return self._final_state

	@final_state.setter
	def final_state(self, final_state: Optional[int]):
		if (final_state is not None) and (final_state not in self._states):
			raise ValueError(f'The final state, if given, must be in the set {{0,...,{self._n_states-1}}}.')
			
		self._final_state = final_state
	
	def reset(self) -> Tuple[ObsType, Tuple[Set[ActType], ...]]:
		super(EnumeratedStochasticGame, self).reset()
		
		self._state = self._rng.choice(self._states, p=self._init_dist)
		return self._state, self._action_sets(self._state)

	def step(self, actions: Tuple[int, ...]) -> Tuple[ObsType, Tuple[Set[ActType], ...], Tuple[float, ...], bool]:
		super(EnumeratedStochasticGame, self).step(actions)

		if len(actions) != self._rewards[self._state].ndim:
			raise ValueError(f'Exactly {self._rewards[self._state].ndim} actions are required but {len(actions)} were given.')
		
		elif not all(action < n_actions for action, n_actions in zip(actions, self._rewards[self._state].shape)):
			raise ValueError(f'The actions must be bounded from above by {self._rewards[self._state].shape}.')

		reward = self._rewards[self._state][actions]
		if self._transitions is not None:
			self._state = self._rng.choice(self._states, p=self._transitions[self._state][actions])
		
		return self._state, self._action_sets(self._state), reward, (self._state == self._final_state) # all agents can fully observe the state

	def render(self):
		pass

	def _action_sets(self, state: int) -> Tuple[Set[ActType], ...]:
		reward_shape = self._rewards[self._state].shape
		return tuple(set(range(reward_shape[idx])) for idx in range(len(reward_shape)))

class MatrixGame(EnumeratedStochasticGame):
	r'''
	Initialize a matrix game (i.e., a stochastic game with one state).

	Parameters:
		rewards (np.ndarray): A reward structure (matrix of tuples).
		rng (np.random._generator.Generator, optional): A random number generator.
	'''
	def __init__(
		self,
		rewards: np.ndarray,
		rng: Optional[np.random._generator.Generator] = None
	):
		super(MatrixGame, self).__init__({0: rewards}, rng=rng)
