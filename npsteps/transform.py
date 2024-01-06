import numpy as np
import matrix as mat
import inverse as inv


def split_elementary_and_states(steps):
    """"
        The transformation process usually returns a list with elementary matrices and current state of the matrix as a tuple.
        args: steps
        return: list of elementary matrices, list of states of the matrix
    """
    elim_mats = []
    states = []
    for step in steps:
        comment, new_state = step
        try:
            meaning, el_mat = comment
            elim_mats.append(el_mat)
        except:
            pass
        states.append(new_state)
    
    return elim_mats, states


def apply_transformation(matrix, transformations):
    """
        Given a matrix and list of tranformations(a list of 1 or more elementary matrices)
        M = T(matrix) => (T1...Tn) * matrix
        
        args: matrix, T (list of transformatrions or elementary matrices)
        return: steps, M 
    """
    steps = []
    T = transformations[0]
    
    for i in range(1, len(transformations)):
        t = transformations[i]
        T = np.dot(T, t)
        steps.append(('Applying T' + str(i+1), t.copy()), T.copy())
    
    return steps, np.dot(T, matrix)


def to_upper_triangular(matrix):

    if mat.is_upper_triangular(matrix): return (['matrix is already upper triangular'], matrix)
    """
        arr: numpy 2D array
            a  b   c                 a  b   c
            d  e   f         ===>    0  e'  f'
            g  h   i                 0  0  i'

        returns detailed_steps, echelon_form
                => detailed_steps = [(labels, elimination_matrix)]
    """
    N = len(matrix)
    m = matrix.copy()
    final_matrix = [m[0]] # first row never changes
    new_state = matrix.copy()
    steps = []

    for r in range(1, N): # from 2nd row to last
        for c in range(r): # 0 ---> i-1 : 0 for row1, 1 for row2, n-1 for row n. 
            # a row has n-1 elements under the diagonal to set to zero.
            # we use row number x to eliminate element on column number x;
            elim_row = m[c] # column vector (3,) 
            coef = mat.elim_coef(elim_row[c], m[r][c]) # a, b 
            new_row = coef * elim_row + m[r] # -b/a 
            m[r] = new_row.T # current state of the matrix

            # elimination
            elim_step = np.identity(N)
            elim_step[r][c] = coef
            new_state[r] = new_row.T
            
            # documentation
            comment = ('a'+str(r+1)+str(c+1)+' => 0 : row'+str(r+1)+' = '+str(coef)+' * row'+str(c+1)+' + row'+str(r+1),
                           np.array(elim_step))
            step = (comment, new_state.copy())
            steps.append(step)

        final_matrix.append(new_row.T)

    # find the elimination matrix E such that A . E = U

    return steps, np.array(final_matrix)


def to_row_echelon_form(matrix):
    return to_upper_triangular(matrix=matrix)


def to_reduced_row_echelon_form(matrix):
    ref_steps, ref, ef_steps, ef = [], [], [], matrix # supposing the matrix is in echelon form
    
    if not mat.is_upper_triangular(matrix=matrix):
        print('First tranform to Row Echelon Form')
        ef_steps, ef = to_row_echelon_form(matrix=matrix)

    N = len(matrix)
    for i in range(N): # for each row
        coef = ef[i] / ef[i][i]
        ref.append(coef)
        
        # elimination documentation
        step = ('row'+str(i+1)+' = 1/' + str(ef[i][i]) +' * row'+str(i+1), ef[i][i])
        ref_steps.append(step)

    return (ef_steps, ef), (ref_steps, np.array(ref))


def to_lower_triangular(matrix):
    
    if mat.is_lower_triangular(matrix): return (['matrix is lower tirangular'], matrix)
    """
        arr: numpy 2D array
            a  b   c                 a  0   0
            d  e   f         ===>    d  e  0
            g  h   i                 g  h  i

        returns detailed_steps, echelon_form
                => detailed_steps = [(labels, elimination_matrix)]
        
        the technique applied here is to find the UPPER TRIANGULAR or the TRANSPOSE of the matrix.
        to_lower_triangualar(matrix) <=> to_upper_trangular(transpose(matrix)) 
    """
    A_t = np.transpose(matrix)
    steps, U = to_upper_triangular(A_t)
    steps.insert(0, ('lower-triangular of a matrix is equivalent to upper-triangular of the transpose of the matrix, given A_transpose', A_t))

    L = np.transpose(U)
    steps.append(('transpose back the matrix to get the lower triangular form', L))

    return steps, L



def join_transformations(matrices):
    """
        When transforming a matrix from the original state to another one, like
        upper triangular form, lower tirangular form, inverse matrix, Hermissian matrix etc...
        the transformation is done step by step and stored as an ELEMENTARY MATRIX
        
        args:
            matrices: list of elementary matrices. E: elementary matrix, E': inverse of e
        return
            T : transformation matrix, the product of the individual steps in the reverse order

        result => T(A) = (En....E1') * A
    """
    N = len(matrices)
    if N == 0:
        return np.identity(N)
    elif N == 1:
        return matrices[0]

    T = matrices[-1]
    transfs = matrices[::-1]
    for t in transfs:
        T = np.dot(T, t)
    
    return T