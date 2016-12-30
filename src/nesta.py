"""nesta.py
The Python implementation of the NESTA algorithm originally implemented in
MATLAB. The MATLAB reference implementation is available at
https://statweb.stanford.edu/~candes/nesta/, and the report describing the
original algorithm is available at
https://statweb.stanford.edu/~candes/nesta/NESTA.pdf

Written by: Liren Zhu (liren.zhu.cn@gmail.com)
Created: December 2016
"""


class USV:
    """The object containing the SVD results of operator A."""

    def __init__(self, U, S, V):
        self.U = U
        self.S = S
        self.V = V


class NestaOptions:
    """The object containing options used for the NESTA solver.
    See the documentation of the *nesta* function for details.
    """

    VALID_TYPE_MIN = ['l1', 'tv']

    def __init__(self):
        """Initialize the option object with default settings."""
        self.x0 = None
        self.type_min = 'l1'
        self.W = None
        self.Wt = None
        self.norm_W = None
        self.cont_iter = 5
        self.max_iter = 1000
        self.tol = 1e-7
        self.stop_test = 1
        self.USV = USV(None, None, None)
        self.verbose = False

    def validate(self):
        """Validate the option object."""
        if not self.type_min or \
           self.type_min.lower() not in self.VALID_TYPE_MIN:
            return False
        return True


def nesta(A, At, b, mu_f, delta, opts):
    """Solve a L1/TV minimization problem under a quadratic constraint using
    the Nesterov algorithm, with continuation:

        min_x || W x ||_1    s.t. || b - Ax ||_2 <= delta

    Continuation is performed by sequentially applying Nesterov's algorithm
    with a decreasing sequence of values of mu, which satisfy mu_0 >= mu >=
    mu_f.

    The primal prox-function is also adapted by accounting for a first guess x0
    that also tends towards x_muf

    The observation matrix A is required to be a projector, unless the SVD of A
    (contained in opts.USV) is provided.

    ========================================================================
    Inputs:

    A and At - observation operator and adjoint (either a matrix, in which case
               At is unused, or a callable). $m \times n$ dimensions.
    b        - observed data, a $m$ vector.
    mu_f     - the desired value of mu at the last continuation step. A smaller
               mu leads to higher accuracy.
    delta    - l2 error bound. This reference how close the variable must fit
               the observation b, i.e. || y - Ax ||_2 <= delta.
               If delta = 0, enforces y = Ax.
               Common heuristic: delta = sqrt(m + 2*sqrt(2*m))*sigma, where
               sigma = std(noise).
    opts     - An object containing additional options. Below is a list of
               field names:
               x0        - the initial guess for the primal prox-function, and
                           also the initial point for x_k. Default: x0 = At(b).
               type_min  - 'l1' or 'tv', the type of objective function used.
               W         - the analysis operator (a matrix or a callable).
               Wt        - the synthesis operator (a callable, ignored if given
                           W).
               norm_W    - the operator norm of W; required if W is provided.
               cont_iter - number of continuation steps. Default: 5.
               max_iter  - max number of iterations in an inner loop.
                           Default: 1,000
               tol       - tolerance for the stopping criteria. Default: 1e-7
               stop_test - the stopping criteria:
                           1: relative change of the objective function.
                           2: l_infinity norm of difference in x_k.
               USV       - An object containing fields U, S, and V, which are
                           the results of the SVD of A. U and V are matrices
                           and S is a vector.
               verbose   - if true, more information will be displayed.

    ========================================================================
    Outputs:

    x_k      - estimate of the solution x.
    n_iter   - number of iterations.
    res      - a vector containing the residuals at each step.
    f_mu     - a vector of values of f_mu at each step.
    """
