import unittest

import numpy as np
from evoenv.matrixtools import FloatTupleDtype, compress_matrices, extract_matrices

class TestMatrixTools(unittest.TestCase):
    
    def test_compress_matrices_same_shape(self):
        matrix1 = np.array([[1.3, 2.0], [3.1, 4.9]], dtype=np.float_)
        matrix2 = np.array([[5.6, 6.2], [7.0, 8.4]], dtype=np.float_)
        matrices = (matrix1, matrix2)
        result = compress_matrices(matrices)
        expected = np.array([[(1.3, 5.6), (2.0, 6.2)], [(3.1, 7.0), (4.9, 8.4)]], dtype=FloatTupleDtype(2))
        for row in range(2):
            for col in range(2):
                for idx in range(2):
                    np.testing.assert_allclose(result[row, col][idx], expected[row, col][idx])

    def test_compress_matrices_different_shape(self):
        matrix1 = np.array([[1.5, 2.1], [3.4, 4.0]], dtype=np.float_)
        matrix2 = np.array([[5.3, 6.7, 7.5], [8.2, 9.1, 10.0]], dtype=np.float_)
        matrices = (matrix1, matrix2)
        with self.assertRaises(ValueError) as context:
            compress_matrices(matrices)
        self.assertIn('All matrices must have the same shape.', str(context.exception))

    def test_extract_matrices_all_indices(self):
        matrix = np.array([[(1.2, 5.3), (2.8, 6.1)], [(3.9, 7.4), (4.0, 8.6)]], dtype=FloatTupleDtype(2))
        result = extract_matrices(matrix)
        expected = {
            0: np.array([[1.2, 2.8], [3.9, 4.0]], dtype=np.float_),
            1: np.array([[5.3, 6.1], [7.4, 8.6]], dtype=np.float_)
        }
        for key in result:
            np.testing.assert_allclose(result[key], expected[key])

    def test_extract_matrices_specific_indices(self):
        matrix = np.array([[(1.1, 5.9, 9.2), (2.4, 6.8, 10.0)], [(3.5, 7.7, 11.6), (4.3, 8.5, 12.1)]], dtype=FloatTupleDtype(3))
        result = extract_matrices(matrix, indices=(0, 2))
        expected = {
            0: np.array([[1.1, 2.4], [3.5, 4.3]], dtype=np.float_),
            2: np.array([[9.2, 10.0], [11.6, 12.1]], dtype=np.float_)
        }
        for key in result:
            np.testing.assert_allclose(result[key], expected[key])

if __name__ == '__main__':
    unittest.main()
