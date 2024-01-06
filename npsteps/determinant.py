"""should be in folder: properties"""

import numpy as np
import transform as t
import matrix as m


def by_gaussian_elimination(matrix):
    steps, ef = t.to_row_echelon_form(matrix)
    # print(steps, ef)
    det = 1
    for i in range(len(ef)):
        det *= ef[i][i]
    steps.append(('Determinant = Product of diagonal entries of a matrix in echelon form(Upper-triangular).', det))

    return steps, ef, det


def sarrus_extension(matrix, opt='right'):
    """
        extends a matrix in a chosen way specied by opt.
        opt='r' or 'right' : copy n-1 first cols to the right. 
        opt='l' or 'left'  : copy n-1 last cols to the left. 
        opt='u' or 'up'    : copy n-1 last rows to the top. 
        opt='d' or down    : copy n-1 first rows to the bottom. 
    """
    match opt:
        case 'right':
            firstcols = matrix.T[:-1] 
            return np.hstack((matrix, firstcols.T))
        case 'left':
            lastcols = matrix.T[1:] 
            return np.hstack((lastcols.T, matrix))

        case 'top':
            lastrows = matrix[1:]  
            print(np.vstack((lastrows, matrix)))
            return np.vstack((lastrows, matrix))
        case 'bottom':
            firstrows = matrix[:-1]  
            return np.vstack((matrix, firstrows))


def by_sarrus_rule(matrix, opt='right'):
    if not m.is_square(matrix=matrix): return (['Not a square matrix'], np.NaN)
    size = len(matrix)
    mat = sarrus_extension(matrix, opt)

    steps = [('matrix extension on the ' + opt, mat)]
    
    main_diag_sum = 0 # sum of products of each main_diagonal of each block
    for c in range(size): # for each col
        block = mat.T[c: c + size] # get a nxn matrix starting from current col.
        steps.append(('block'+str(c+1)+': from col'+str(c+1)+' to col' + str(c+size), block))        
        diag = m.main_diagonal(block)
        prod = np.prod(diag)
        main_diag_sum += prod
        steps.append(('Product of entries of diagonal' + str(c+1) + '=' + str(prod), diag))
    steps.append(('Sum of products of main diagonals:', main_diag_sum))

    second_diag_sum = 0 # sum of products of each main_diagonal of each block
    for c in range(size): # for each col
        block = mat.T[c: c + size] # get a nxn matrix starting from current col.
        steps.append(('block'+str(c+1)+': from col'+str(c+1)+' to col' + str(c+size), block))        
        diag = m.second_diagonal(block)
        prod = np.prod(diag)
        second_diag_sum += prod
        steps.append(('Product of entries of diagonal' + str(c+1) + '=' + str(prod), diag))
        
    steps.append(('Sum of products of second diagonals:', second_diag_sum))
    
    return steps, main_diag_sum - second_diag_sum


def _cofactor_expansion(matrix): # for 2x2

    if not m.is_square(matrix=matrix): return 'Not a square matrix'

    if len(matrix) > 2 : return "Not 2x2 matrix"
    steps = []
    det = 0

    det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    steps = '2x2 matrix expansion = product of main diagonal - product of second diagonal'

    return steps, det
    

def by_cofactor_expansion(matrix, opt):
    """
        axis : expand along rows(0) or cols(1); by default, axis=0 and ind=0
        coefs : row or col of coefficients, not involved in the cofactors.
    """
    if not m.is_square(matrix=matrix): return (['Not a square matrix'], np.NaN)
    axis, ind = opt
    ax = 'row' if axis == 0 else 'col'

    steps = []
    det = 0
    N = len(matrix)
    cofactors = []  
    coeficients = matrix[ind] if axis == 0 else matrix.T[ind] # for each column along the row axis / row along column axis 
    for i in range(N): # there are N cofactors for a NxN matrix, the first round to document.
        coeficient = coeficients[i] * ((-1) ** (ind+1 + i+1))
        cofactor = m.cofactor(matrix, (0, i)) # expand along rows axis and row specified by 'ind';
        cofactors.append(cofactor)
        if N - 1 == 2: # 2x2 cofactors
            det += coeficient * _cofactor_expansion(cofactor)[1]
    steps.append(('First Cofactors expansion along ' + ax + str(coeficients), cofactors))

    n = len(cofactors[0])
    while n > 2: # repeat the process 
        cofactors = [by_cofactor_expansion(c, opt)[1] for c in cofactors]

    # the determinant is the a . d - b . c to calculate
        for cofactor in cofactors:
            det += coeficient * _cofactor_expansion(cofactor)[1]

    # n > N means we passed though the above while loop, since a square matrix of size N produces N cofactors of size (N-1).  'more cofactors in number' but 'smaller in size'
    if n > N: steps.append(('Last Cofactors expansion\nThe determinant is the sum of determinants of 2x2 cofactors premultiplied by their coeficients', cofactors))

    return steps, det