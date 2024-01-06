import numpy as np
import matrix as m
import transform as transf

def of_elementary_matrix(E):

    """
        given elementary matrix E   |       means
        a row like: [1   -3  2]         R1' <= R1 - 3R2 + 2R3
                    [2   5   3]         R2' <= 2R1 + 5R2 + 3R3
                    [0   0   1]         R3' <= R3

        The inverse of such matrices is found by inversing each operation:
        '+' <--> '-' and '*' <--> '/'
        So inverse(E)   |       means
        [1  +3  -2]         R1 <= R1' + 3R2 - 2R3
        [-2 1/5 -3]         R2 <= -2R1' - 5R2 - 3R3
        [0  0   1]          R3 <= R3'
    """
    I = [] # inverse
    for i in range(len(E)):
        row = []
        for j in range(len(E[0])):
            if i == j: # diagonal element => multiplication
                row.append(1 / E[i][j]) # divide by the multiplier of the current row
            else:
                v = E[i][j]
                row.append(-v if v != 0 else 0)
        I.append(row)
    return np.array(I)


def by_gaussian_elimination(matrix):

    """
        Given a square matrix A of size N, the gaussian elimination consists to work with
        an extended matrix: the matrix extended by the Identity of size N : [A | I]
        The extension can be horizontal or vertical. the main idea is to keep track of positions of A and I.
        by default we extended as [A | I] ---> [I | A'] where A' is the inverse of A.
        transform A to inverse and simultaneously apply the same changes to I. at finish it turns to be the A'.

        return
            steps = (comment, (current_matrix_state, current_identity_state))
            I = inverse of matrix
            """

    # if not m.is_square(matrix=matrix): return 'Not a square matrix'
    if np.linalg.det(matrix) == 0: return 'No Inverse, the determinant = 0'    
    
    N = len(matrix)
    steps = [(('Create an augmented matrix of A and Identity I', matrix.copy()), np.identity(N))] # just to conserve structure from uppertriangular
     
    #1. transform the matrix to upper triangular: U and apply transformatrion E(identity) 
    ut_steps, U = transf.to_upper_triangular(matrix=matrix) # upper _triangular
    steps.extend(ut_steps)

    elementary_mats = transf.split_elementary_and_states(ut_steps)[0] # this method returns (elementary_mats, states) 
    T = transf.join_transformations(elementary_mats)
    I_new = np.dot(T, np.identity(N))
    steps.append((('Product of elimination matrices in the reverse order gives T = Tn ... * T2 * T1.\nU: upper triangular matrix, Applying T to the identity: T(I)', T.copy()), I_new.copy()))

    #2. transform U to lower triangular by doing 'upper_triangular(A_transpose)', the resulting matrix is diagonal: D
    ut_steps, D = transf.to_upper_triangular(np.transpose(U)) # this will be a digonal matrix    
    steps.extend(ut_steps)

    elementary_mats = transf.split_elementary_and_states(ut_steps)[0] # this method returns (elementary_mats, states) 
    T = transf.join_transformations(elementary_mats)
    I_new = np.dot(T, np.transpose(I_new))
    steps.append((('Transforming to lower triangular equals applying upper triangular to the transpose of the matrix.' + 
                  "LOWER_TRIANGULAR(U) <=> UPPER_TRIANGULAR(U_transpose). Apply T' to U and I_new, the resulting matrix is diagonal.", T.copy()), I_new.copy()))

    #3. transform the D to identity by doing REDUCED ROW ECHELON FORM
    ref_steps, ref = transf.to_reduced_row_echelon_form(D)[1] # it returns (ef_steps, ef), (ref_steps, ref) 
    for i in range(len(ref_steps)):
        coef = ref_steps[i][1] # it returns new_row, coeficient
        I_new[i] /= coef

    steps.append((('Divide each row by the diagonal element of D(diagonal matrix), then apply to I_new)', T.copy()), I_new.copy()))

    return steps, I_new














