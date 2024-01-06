MATHS_STEPS LIBRARY WITH NUMPY SUPPORT

#1. The library supports numpy mainly for linear algebra

#2. The main goal is to provide the developper or the learner with the human readble steps of mathematical solutions.

#3. The library considers a matrix as numpy array, since no inbuilt functions of matrices are used directly. 

#4. the numpy.matrix class has inbuilt variables such as I(inverse), H(hermissian), T(tanspose), etc... pre-computed when the matrix is created. for the sake of performance, and since these elements are not or at least not used a lot for the step-by-step documentation of the solutions, we consider the input matrix as numpy.array.

#5. In case of need, feel free to cast the result to matrix or cast back from matrix to array for compatibility when calling functions.

#6. the file names are based on a specific topic, example inverse, determinat, decomposition, etc.

#7. the functions names in a file are based on specific methods for which to provide the solution and its step-by-step documentation. 

consider examples:


---file--- determinant.py
---function---by_gaussian_elimination
---function---by_sarrus_rule
---function---by_cofactor_expansion


---file--- inverse.py
---function---of_elementary_matrix
---function---by_gaussian_elimination
---function---by_adjugate

For use-case examples, please refer to npsteps/examples.py
