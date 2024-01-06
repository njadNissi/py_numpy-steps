import numpy as np


def display(steps, final_matrix):

    for step in steps:
        comment, new_state = step
        try:
            meaning, el_mat = comment
            print('Elimination matrix:\n', el_mat)
            print('Meaning: ' + meaning)
        except:
            print(comment)
        print('current state:\n', new_state)
        print('-*-' * 20)
    
    print('final state:\n', final_matrix)


def is_square(matrix):
    return len(matrix) == len(matrix[0])

def copy(matrix):
    copy = []
    for row in matrix:
        c_row = []
        for col in row:
            c_row.append(col)
        copy.append(c_row)
    return copy

def elim_coef(a, b):
    """
        given two numbers a & b, return c such that c*a + b = 0
    """
    return -b / a


def is_upper_triangular(matrix):
    """
        an upper triangular matrix is the with all the elements below the main diagonal are zeros.
    """
    for i in range(1, len(matrix)):
        for j in range(i):
            if matrix[i][j] != 0: return False
            
    return True


def is_lower_triangular(matrix):
    """
        a lower triangular matrix is the with all the elements above the main diagonal are zeros.
    """
    N = len(matrix)
    for i in range(N):
        for j in range(i+1, N):
            if matrix[i][j] != 0: return False

    return True


def main_diagonal(matrix):
    diag = []
    max_square_block_size = min(len(matrix), len(matrix[0]))
    for i in range(max_square_block_size):
        diag.append(matrix[i][i])
    return diag


def second_diagonal(matrix):
    diag = []
    max_square_block_size = min(len(matrix), len(matrix[0]))
    for i in range(max_square_block_size):
        diag.append(matrix[i][max_square_block_size - i - 1])
    return diag


def cofactor(matrix, entry):
    row, col = entry
    cof = np.delete(matrix, (row), axis=0)
    cof = np.delete(cof, (col), axis=1)
    return cof


def to_identity(matrix):
    """
        given a square matrix, generate all the step-by-step documentation
        to transform it into an identity matrix.
    """
    if not is_square(matrix=matrix): return False

    N = len(matrix)
    E = np.identity()    
    steps = []
    for i in range(N):
        row = []
        for j in range(N):
            if i == j: # digonal element ---> 1
                row.append()

    return steps, E # E has become the elementary matrix that transforms matrix to identity


def h_augmented(matrices):
    """
        Given a list of matrices, create an augmented matrix of them, one on the right of the previous one.
    """
    N = len(matrices)
    if N < 2 : return 'No augmented matrix for single matrix'

    str = ''
    for i in range(N): # for each row
        for M in matrices:
            row = M[i]
            "UNFINISHED"