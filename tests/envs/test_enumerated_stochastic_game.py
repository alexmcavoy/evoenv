import unittest

import numpy as np

from evoenv.envs import EnumeratedStochasticGame, MatrixGame
from evoenv.matrixtools import FloatTupleDtype, compress_matrices, extract_matrices

class TestEnumeratedStochasticGame(unittest.TestCase):

    def setUp(self):
        self.dtype = FloatTupleDtype(2)
        self.rewards = {
            0: np.array([[(1.0, -1.0), (-1.0, 1.0)], [(1.0, -1.0), (-1.0, 1.0)]], dtype=self.dtype),
            1: np.array([[(-1.0, 1.0), (1.0, -1.0)], [(1.0, -1.0), (-1.0, 1.0)]], dtype=self.dtype)
        }
        self.transitions = {
            0: np.array([[[0.9, 0.1], [0.8, 0.2]], [[0.7, 0.3], [0.6, 0.4]]], dtype=np.float_),
            1: np.array([[[0.6, 0.4], [0.7, 0.3]], [[0.8, 0.2], [0.9, 0.1]]], dtype=np.float_)
        }

    def test_enumerated_stochastic_game_rewards_validation(self):
        # valid rewards
        env = EnumeratedStochasticGame(rewards=self.rewards)
        for key in self.rewards:
            np.testing.assert_array_equal(env.rewards[key], self.rewards[key])

        # rewards with the wrong states
        rewards_invalid_keys = {
            0: np.array([[(1, 2), (3, 4)], [(5, 6), (7, 8)]], dtype=self.dtype),
            2: np.array([[(9, 10), (11, 12)], [(13, 14), (15, 16)]], dtype=self.dtype),
            3: np.array([[(17, 18), (19, 20)], [(21, 22), (23, 24)]], dtype=self.dtype)
        }
        with self.assertRaises(ValueError) as context:
            env = EnumeratedStochasticGame(rewards=rewards_invalid_keys)
        self.assertIn('All state labels must be in the set {0,...,2}.', str(context.exception))

        # rewards with different numbers of agents
        rewards_invalid_values = {
            0: np.array([[(1.0, -1.0), (-1.0, 1.0)], [(1.0, -1.0), (-1.0, 1.0)]], dtype=self.dtype),
            1: np.array([[[(-1.0, 1.0), (1.0, -1.0)]]], dtype=self.dtype)
        }
        with self.assertRaises(ValueError):
            env = EnumeratedStochasticGame(rewards=rewards_invalid_values)

        # empty rewards
        rewards_empty = {}
        with self.assertRaises(ValueError) as context:
            env = EnumeratedStochasticGame(rewards=rewards_empty)
        self.assertIn('There must be at least one state in the game.', str(context.exception))

        # rewards with non-sequential states
        rewards_non_sequential = {
            0: np.array([[(1, 2)], [(3, 4)]], dtype=self.dtype),
            1: np.array([[(5, 6)], [(7, 8)]], dtype=self.dtype),
            5: np.array([[(9, 10)], [(11, 12)]], dtype=self.dtype)
        }
        with self.assertRaises(ValueError) as context:
            env = EnumeratedStochasticGame(rewards=rewards_non_sequential)
        self.assertIn('All state labels must be in the set {0,...,2}.', str(context.exception))

    def test_rewards_modification_after_initialization(self):
        # valid modification of rewards
        env = EnumeratedStochasticGame(rewards=self.rewards)
        new_rewards = {
            0: np.array([[(1, 2), (3, 4)], [(5, 6), (7, 8)]], dtype=self.dtype),
            1: np.array([[(9, 10), (11, 12)], [(13, 14), (15, 16)]], dtype=self.dtype)
        }
        env.rewards = new_rewards
        for key in new_rewards:
            np.testing.assert_array_equal(env.rewards[key], new_rewards[key])

        # invalid modification of rewards (mismatched states)
        env = EnumeratedStochasticGame(rewards=self.rewards)
        mismatched_rewards = {
            0: np.array([[(1, 3), (3, 5)]], dtype=self.dtype)
        }
        with self.assertRaises(ValueError) as context:
            env.rewards = mismatched_rewards
        self.assertIn('The states in the transition structure must equal those of the reward structure.', str(context.exception))

    def test_enumerated_stochastic_game_transitions_validation(self):
        # valid transitions
        env = EnumeratedStochasticGame(rewards=self.rewards, transitions=self.transitions)
        for key in self.transitions:
            np.testing.assert_array_equal(env.transitions[key], self.transitions[key])

        # transitions with the wrong number of states
        transitions = {
            0: np.array([[[0.9, 0.1], [0.8, 0.2]], [[0.7, 0.3], [0.6, 0.4]]], dtype=np.float_)
        }
        with self.assertRaises(ValueError):
            env = EnumeratedStochasticGame(self.rewards, transitions=transitions)

    def test_transitions_modification_after_initialization(self):
        # valid modification of transitions
        env = EnumeratedStochasticGame(rewards=self.rewards, transitions=self.transitions)
        new_transitions = {
            0: np.array([[[0.8, 0.2], [0.9, 0.1]], [[0.6, 0.4], [0.5, 0.5]]], dtype=np.float_),
            1: np.array([[[0.5, 0.5], [0.6, 0.4]], [[0.7, 0.3], [0.8, 0.2]]], dtype=np.float_)
        }
        env.transitions = new_transitions
        np.testing.assert_array_almost_equal(env.transitions[1], new_transitions[1])

        # invalid modification of transitions (wrong depth)
        env = EnumeratedStochasticGame(rewards=self.rewards, transitions=self.transitions)
        wrong_structure_transitions = {
            0: np.array([[0.9, 0.1], [0.8, 0.2]], dtype=np.float_),
            1: np.array([[0.6, 0.4], [0.7, 0.3]], dtype=np.float_)
        }
        with self.assertRaises(ValueError) as context:
            env.transitions = wrong_structure_transitions
        self.assertIn('The states in {0, 1} do not have 2 non-negative weights (not all zero) for every possible action profile.', str(context.exception))

        # invalid modification of transitions (wrong states)
        env = EnumeratedStochasticGame(rewards=self.rewards, transitions=self.transitions)
        inconsistent_transitions = {
            0: np.array([[[0.8, 0.2], [0.9, 0.1]], [[0.6, 0.4], [0.5, 0.5]]], dtype=np.float_),
            2: np.array([[[0.5, 0.5], [0.6, 0.4]], [[0.7, 0.3], [0.8, 0.2]]], dtype=np.float_)
        }
        with self.assertRaises(ValueError) as context:
            env.transitions = inconsistent_transitions
        self.assertIn('The states in the transition structure must equal those of the reward structure.', str(context.exception))

        # invalid modification of transitions (negative entries)
        env = EnumeratedStochasticGame(rewards=self.rewards, transitions=self.transitions)
        negative_values_transitions = {
            0: np.array([[[0.8, -0.2], [0.9, 0.1]], [[0.6, 0.4], [0.5, 0.5]]], dtype=np.float_),
            1: np.array([[[0.5, 0.5], [0.6, 0.4]], [[0.7, -0.3], [0.8, 0.2]]], dtype=np.float_)
        }
        with self.assertRaises(ValueError) as context:
            env.transitions = negative_values_transitions
        self.assertIn('The states in {0, 1} do not have 2 non-negative weights (not all zero) for every possible action profile.', str(context.exception))

    def test_enumerated_stochastic_game_reset(self):
        env = EnumeratedStochasticGame(self.rewards)
        state, action_sets = env.reset()
        self.assertIn(state, (0, 1))
        self.assertEqual(action_sets, (set(range(2)), set(range(2))))

    def test_enumerated_stochastic_game_step(self):
        env = EnumeratedStochasticGame(rewards=self.rewards, transitions=self.transitions)
        env.reset()
        state, action_sets, reward, done = env.step((0, 1))
        self.assertIn(state, (0, 1))
        self.assertEqual(action_sets, (set(range(2)), set(range(2))))
        self.assertFalse(done)

class TestMatrixGame(unittest.TestCase):

    def setUp(self):
        self.rewards = np.array([[(1.0, -1.0), (-1.0, 1.0)], [(1.0, -1.0), (-1.0, 1.0)]], dtype=FloatTupleDtype(2))
        self.env = MatrixGame(self.rewards)

    def test_matrix_game_rewards(self):
        np.testing.assert_array_equal(self.env.rewards[0], self.rewards)

    def test_matrix_game_reset(self):
        state, action_sets = self.env.reset()
        self.assertEqual(state, 0)
        self.assertEqual(action_sets, (set(range(2)), set(range(2))))

    def test_matrix_game_step(self):
        state, action_sets = self.env.reset()
        state, action_sets, reward, done = self.env.step((0, 1))
        self.assertEqual(state, 0)
        self.assertEqual(action_sets, (set(range(2)), set(range(2))))
        np.testing.assert_array_equal(reward, self.rewards[(0, 1)])
        self.assertFalse(done)

if __name__ == '__main__':
    unittest.main()
