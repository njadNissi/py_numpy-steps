import numpy as np
import matrix as m
import transform as transf
import determinant as det
import inverse as inv
import decomposition as factor


def show(title, elements):
    print(title + '\n')
    for e in elements: print(e, '\n')


def ef_example(matrix):
    ef_steps, ef = transf.to_row_echelon_form(matrix=matrix)
    m.display(ef_steps, ef)


def ref_example(matrix):
    ef, ref = transf.to_reduced_row_echelon_form(matrix=matrix)
    print('ECHELON FORM PROCESS')
    m.display(ef[0], ef[1])

    print('REDUCED ECHELON FORM PROCESS')
    steps, ref_mat = ref
    m.display(steps, ref_mat)


def upper_tri_example(matrix):
    steps, U = transf.to_upper_triangular(matrix=matrix)
    print('UPPER TRIANGULAR TRANSFORM PROCESS')
    m.display(steps, U)


def lower_tri_example(matrix):
    steps, L = transf.to_lower_triangular(matrix=matrix)
    print('LOWER TRIANGULAR TRANSFORM PROCESS')
    m.display(steps, L)


def gauss_det(matrix):
    steps, ef, d = det.by_gaussian_elimination(matrix)
    m.display(steps, ef)
    print('Determinant: ' + str(d))


def sarrus_det(matrix, opt):
    steps, d = det.by_sarrus_rule(matrix, opt)
    m.display(steps, 'Determinant: ' + str(d))

def cofactor_expansion(matrix, opt):
    step, d = det.by_cofactor_expansion(matrix, opt)
    
    m.display(step, 'Determinant: ' + str(d))


def lu_example(matrix):
    L, U = factor.LU(matrix=matrix)  
    U_steps, U_mat = U
    L_steps, L_mat = L
    print('A = L . U DECOMPOSITION')
    for el_mat, el_mat_inv in zip(L_steps[0], L_steps[1]):
        print('elim step:\n', el_mat)
        print('inv of elim step:\n', el_mat_inv)
        print('-*-' * 20)
    
    print('L = \n', L_mat)  
    print('U = \n', U_mat)  
    print('A = L.U = \n', np.dot(L_mat, U_mat))


def ldu_example(matrix):
    L, D, U = factor.LDU(matrix)
    U_steps, U = U
    D_steps, D = D
    L_steps, L = L

    print('A = L . D . U Decomposition')
    for el_mat, el_mat_inv in zip(L_steps[0], L_steps[1]):
        print('elim step:\n', el_mat)
        print('inv of elim step:\n', el_mat_inv)
        print('-*-' * 20)
    
    print('L = \n', L)
    print('D = \n', D)
    print('U = \n', U)  
    print('A = L.D.U = \n', np.dot(L, np.dot(D, U)))



def join_transf_example(matrix):
    ef_steps, ef = transf.to_row_echelon_form(matrix=matrix)
    elim_mats, states = transf.split_elementary_and_states(ef_steps)

    print('TRANSFORMATION COMPOSITION : T = T1 * T2 * ... Tn') 
    show('Elimination matrices:', elim_mats)
    show('States:', states)
    T = transf.join_transformations(elim_mats)    
    print('the final T:\n', T)


def inverse_gaussian_example(matrix):
    steps, I = inv.by_gaussian_elimination(matrix=matrix)
    m.display(steps, I)