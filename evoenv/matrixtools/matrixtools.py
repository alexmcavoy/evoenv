import numpy as np

from typing import Any, Dict, Optional, Tuple

class FloatTupleDtype:
	r'''
	Custom data type for structured arrays, representing a tuple of floats.
	'''
	def __new__(cls, n: int):
		return np.dtype([(f'f{idx}', np.float_) for idx in range(n)])

def compress_matrices(matrices: Tuple[np.ndarray, ...]) -> np.ndarray:
	r'''
	Converts a tuple of multidimensional arrays into a single multidimensional array of tuples.

	Parameters:
		matrices (Tuple[np.ndarray, ...]): A tuple of multidimensional arrays, all of the same shape.

	Returns:
		np.ndarray: A multidimensional array of tuples, giving a well-defined payoff structure for a state.

	Raises:
		ValueError: If the input matrices do not all have the same shape.
	'''
	# validate payoff matrices
	if len(set([matrix.shape for matrix in matrices])) > 1:
		raise ValueError('All matrices must have the same shape.')
		
	return np.reshape(
		np.fromiter(
			map(
				lambda idx: tuple([matrix[idx] for matrix in matrices]),
				np.ndindex(matrices[0].shape)
			),
			dtype=FloatTupleDtype(len(matrices))
		),
		matrices[0].shape
	)

def extract_matrices(matrix: np.ndarray, indices: Optional[Tuple[int, ...]] = None) -> Dict[np.int_, np.ndarray]:
	r'''
	Converts a multidimensional array of tuples into a dictionary of multidimensional arrays.

	Parameters:
		matrix (np.ndarray): A multidimensional array of tuples.

		indices (Tuple[int, ...], optional): A tuple of indices whose corresponding entries are to be extracted.

	Returns:
		Dict[np.int_, np.ndarray]: A dictionary of integer-array pairs, whose keys correspond to the input indices.
	'''
	
	
	indices = indices if indices is not None else range(len(matrix.flat[0]))
	return {idx: np.fromiter(
			map(
				lambda x: matrix[x][idx],
				np.ndindex(matrix.shape)
			),
			dtype=np.float_
		).reshape(matrix.shape) for idx in indices}
