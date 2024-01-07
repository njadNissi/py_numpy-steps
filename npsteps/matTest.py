import numpy as np
import examples as ex


a = np.array([1, 2])
b =  np.array([3, 4, 7, 1])
A = np.array([[1, 2, 3],
              [2, 2, 1],
              [3, 4, 3]])
E = np.array([[1, 2, -3], [0, -2, 1], [1, 0, 1]])


# ex.ef_example(matrix=A)
# ex.ref_example(matrix=A)
# ex.gauss_det(A)
ex.sarrus_det(A, 'right')
# ex.sarrus_det(A, 'left')
# ex.cofactor_expansion(A, (0, 0))
# ex.lu_example(A) 
# ex.ldu_example(matrix=A)
# ex.lower_tri_example(A)
# ex.join_transf_example(A)
# ex.inverse_gaussian_example(A)