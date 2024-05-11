# evoenv: environments for evolutionary games

This repository provides _simple_ environments for studying the dynamics of interactions, with an emphasis on modeling conflicts of interest frequently encountered in evolutionary game theory. The idea is that there are already many interesting problems at the intersection of evolutionary game theory and multi-agent reinforcement learning related to "small" multi-state games.

We use an API inspired by [Gym](https://github.com/openai/gym) from OpenAI.

At present, there is one environment (an "enumerated" stochastic game) and a special case (a matrix game, i.e. a stochastic game with one state).

## Environments

### EnumeratedStochasticGame

* In each state, a multidimensional payoff array of _tuples_ must be specified. In symmetric games with two players, a single payoff matrix suffices to describe all possible payoffs to both players. With more than two players, a single multidimensional array is not equivalent to a symmetric stage game, so we require in all situations the payoff entries to encode payoff information for all players.

### MatrixGame

* An EnumeratedStochasticGame with a single state.

## Use

To install via pip, run the commands

```
pip3 install .
pip3 install -r requirements.txt
```

from the root directory.

All reward matrices have tuples of floats as entries, representing the payoffs for all agents, which requires a custom data type, `FloatTupleDtype`. The term "matrix" refers the payoff array for a stage game, which can be a multidimensional array. Within `matrixtools` are methods for converting standard arrays into arrays of tuples and vice versa. Example usage is as follows:

```python
import numpy as np
from evoenv.matrixtools import FloatTupleDtype, compress_matrices, extract_matrices

A = np.array([[3, 0], [5, 1]], dtype=np.float_)
B = np.array([[1, -1], [2, 0]], dtype=np.float_)

rewards = compress_matrices((A, B))
```
The original matrices can be recovered using

```python
extracted_matrices = extract_matrices(rewards)

# extracted_matrices[0] corresponds to A
# extracted_matrices[1] corresponds to B

```
The payoff matrix (of tuples) can be constructed directly using

```python
expected_rewards = np.array([[(3, 1), (0, -1)], [(5, 2), (1, 0)]], dtype=FloatTupleDtype(2))
```

## Examples

### Standard prisoner's dilemma

```python
from evoenv.envs import MatrixGame

R, S, T, P = 3, 0, 5, 1
rewards = np.array(
    [
        [(R, R), (S, T)],
        [(T, S), (P, P)]
    ],
    dtype=FloatTupleDtype(2)
)
env = MatrixGame(rewards=rewards)

# every environment must be reset initially
state, action_sets = env.reset()
# state is 0 and action_sets is ({0, 1}, {0, 1})

# X takes action 0 (cooperate) and Y takes action 1 (defect)
state, action_sets, reward, done = env.step((0, 1))
# state is 0, action_sets is ({0, 1}, {0, 1}), reward is (0., 5.), and done is False
```

### A two-state game with deterministic transitions
```python
from evoenv.envs import EnumeratedStochasticGame

b, c = 1, 2
rewards = {
    0: np.array(
        [
            [(b, 0), ((1/2)*(b-c), (1/2)*b)],
            [(b, 0), ((1/2)*(b-c), (1/2)*b)]
        ], dtype=FloatTupleDtype(2)
    ),
    1: np.array(
        [
            [(0, b), (0, b)],
            [((1/2)*b, (1/2)*(b-c)), ((1/2)*b, (1/2)*(b-c))]
        ], dtype=FloatTupleDtype(2)
  )
}

transitions = {state: np.zeros((2, 2, 2)) for state in range(2)}
transitions[0][:, :, 1] = 1
transitions[1][:, :, 0] = 1

env = EnumeratedStochasticGame(rewards=rewards, transitions=transitions)

# every environment must be reset initially
state, action_sets = env.reset()
# state is 0 and action_sets is ({0, 1}, {0, 1})

# X takes action 1 and Y takes action 1
state, action_sets, reward, done = env.step((1, 1))
# state is 1, action_sets is ({0, 1}, {0, 1}), reward is (-0.5, 0.5), and done is False

```

Above, `transitions[state][ax, ay, new_state]` is the probability of transitioning from `state` to `new_state` after _X_ takes action `aX` and _Y_ takes action `aY`. If transitions is not passed to `EnumeratedStochasticGame`, then by default the next state is chosen uniformly at random.

Further examples may be found in `tests/`.
