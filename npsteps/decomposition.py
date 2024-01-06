import numpy as np
import transform as t
import inverse as I

def LU(matrix):
    
    """
        A = L . U 
        A: square matrix
        L : lower triangular matrix
        U: upper triangular matrix
        
        return 
            U_steps: steps to find U matrix
            L_steps: steps to find L matrix
    """

    U_steps, U = t.to_row_echelon_form(matrix)

    elim_matrices = [] # stack
    for step in U_steps:
        comment, new_state = step
        meaning, elmat = comment
        elim_matrices.append(elmat)

    elim_matrices = elim_matrices[1:] # remove the very first identity matrix, no step taken

    elim_matrices_inv = [I.of_elementary_matrix(e) for e in elim_matrices]
    L = np.identity(len(matrix))

    for e in elim_matrices_inv:
        L = np.dot(L, e) 

    L_steps = (elim_matrices, elim_matrices_inv)
    return (L_steps, L), (U_steps, U)


def LDU(matrix):
    """
        A = L . U 
        A: square matrix

        return 
            L : lower triangular matrix
            D: diagonal matrix
            U: upper triangular matrix
            U_steps: steps to find U matrix
            L_steps: steps to find L matrix
    """

    L, U = LU(matrix)
    L_steps, L = L # L becomes the L matrix
    U_steps, old_U = U

    #1. separate new_U = D . old_U
    D = np.diag(np.diag(old_U))
    D_inv = np.linalg.inv(D)
    U = np.dot(D_inv, old_U) # updates U as Unew
    # documentation for Diagonal matrix
    D_steps = [('Given the old U matrix from LU decomposition', old_U),
               ('Uold = D . Unew, D:the main diagonal of Uold.', D),
               ("Multiply by D'(inverse of D) on both sides: D'.Uold = D'.D'.Unew => Unew = D'.uold"),
               ("D'(inverse of D) is is computed with any method", D_inv),
               ("Unew", U)]

    #2. find newU
    return (L_steps, L),(D_steps, D), (U_steps, U)