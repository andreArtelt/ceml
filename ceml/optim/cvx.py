# -*- coding: utf-8 -*-
import logging
from abc import ABC, abstractmethod
import numpy as np
import cvxpy as cp


def cvx_desc_to_solver(solver_desc):
    if solver_desc == "SCS":
        return cp.SCS
    elif solver_desc == "MOSEK":
        return cp.MOSEK
    elif solver_desc == "CVXOPT":
        return cp.CVXOPT
    elif solver_desc == "OSQP":
        return cp.OSQP
    elif solver_desc == "ECOS":
        return cp.ECOS
    elif solver_desc == "CPLEX":
        return cp.CPLEX
    elif solver_desc == "CBC":
        return cp.CBC
    elif solver_desc == "NAG":
        return cp.NAG
    elif solver_desc == "GLPK":
        return cp.GLPK
    elif solver_desc == "GLPK_MI":
        return cp.GLPK_MI
    elif solver_desc == "GUROBI":
        return cp.GUROBI
    elif solver_desc == "SCIP":
        return cp.SCIP
    elif solver_desc == "XPRESS":
        return cp.XPRESS
    else:
        raise ValueError(f"Solver '{solver_desc}' is not supported or unkown.")


class MathematicalProgram():
    """Base class for a mathematical program.
    """
    def __init__(self, **kwds):
        super().__init__(**kwds)

    @abstractmethod
    def solve(self):
        raise NotImplementedError()


class SupportAffinePreprocessing():
    """Base class for a mathematical programs that support an affine preprocessing.
    """
    def __init__(self, **kwds):
        self.A = None
        self.b = None

        super().__init__(**kwds)

    def set_affine_preprocessing(self, A, b):
        self.A = A
        self.b = b

    def get_affine_preprocessing(self):
        return {"A": self.A, "b": self.b}

    def is_affine_preprocessing_set(self):
        return self.A is not None and self.b is not None

    def _apply_affine_preprocessing_to_var(self, var_x):
        if self.A is not None and self.b is not None:
            return self.A @ var_x + self.b
        else:
            return var_x
    
    def _apply_affine_preprocessing_to_const(self, x):
        if self.A is not None and self.b is not None:
            return np.dot(self.A, x) + self.b
        else:
            return x


class ConvexQuadraticProgram(ABC, SupportAffinePreprocessing):
    """Base class for a convex quadratic program - for computing counterfactuals.

    Attributes
    ----------
    epsilon : `float`
        "Small" non-negative number for relaxing strict inequalities.
    """
    def __init__(self, **kwds):
        self.epsilon = 1e-2
        self.solver = cp.SCS
        self.solver_verbosity = False

        super().__init__(**kwds)

    @abstractmethod
    def _build_constraints(self, var_x, y):
        """Creates and returns all constraints.

        Parameters
        ----------
        var_x : `cvx.Variable`
            Optimization variable.
        y : `int` or `float`
            The requested prediction of the counterfactual - e.g. a class label.

        Returns
        -------
        `list`
            List of cvxpy constraints.
        """
        raise NotImplementedError()

    def _solve(self, prob):
        prob.solve(solver=self.solver, verbose=self.solver_verbosity)

    def build_solve_opt(self, x_orig, y, features_whitelist=None, mad=None, optimizer_args=None):
        """Builds and solves the convex quadratic optimization problem.
        
        Parameters
        ----------
        x_orig : `numpy.ndarray`
            The original data point.
        y : `int` or `float`
            The requested prediction of the counterfactual - e.g. a class label.
        features_whitelist : `list(int)`, optional
            List of feature indices (dimensions of the input space) that can be used when computing the counterfactual.
        
            If `features_whitelist` is None, all features can be used.

            The default is None.
        mad : `numpy.ndarray`, optional
            Weights for the weighted Manhattan distance.

            If `mad` is None, the Euclidean distance is used.

            The default is None.
        optimizer_args : `dict`, optional
            Dictionary for overriding the default hyperparameters of the optimization algorithm.

            The default is None.
        
        Returns
        -------
        `numpy.ndarray`
            The solution of the optimization problem.

            If no solution exists, `None` is returned.
        """
        if optimizer_args is not None:
            if "epsilon" in optimizer_args:
                self.epsilon = optimizer_args["epsilon"]
            if "solver" in optimizer_args:
                self.solver = cvx_desc_to_solver(optimizer_args["solver"])
            if "solver_verbosity" in optimizer_args:
                self.solver_verbosity = optimizer_args["solver_verbosity"]

        dim = x_orig.shape[0]
        
        # Variables
        x = cp.Variable(dim)
        beta = cp.Variable(dim)
        
        # Constants
        c = np.ones(dim)
        z = np.zeros(dim)
        I = np.eye(dim)

        # Construct constraints
        constraints = self._build_constraints(x, y)

        # If requested, fix some features
        if features_whitelist is not None:
            A = []
            a = []

            for j in range(dim):
                if j not in features_whitelist:
                    t = np.zeros(dim)
                    t[j] = 1.
                    A.append(t)
                    a.append(x_orig[j])
            
            if len(A) != 0:
                A = np.array(A)
                a = np.array(a)

                constraints += [A @ x == a]

        # If necessary, construct the weight matrix for the weighted Manhattan distance
        Upsilon = None
        if mad is not None:
            alpha = 1. / mad
            Upsilon = np.diag(alpha)

        # Build the final program
        f = None
        if mad is not None:
            f = cp.Minimize(c.T @ beta)    # Minimize (weighted) Manhattan distance
            constraints += [Upsilon @ (x - x_orig) <= beta, (-1. * Upsilon) @ (x - x_orig) <= beta, I @ beta >= z]
        else:
            f = cp.Minimize((1/2)*cp.quad_form(x, I) - x_orig.T@x)  # Minimize L2 distance
        
        prob = cp.Problem(f, constraints)
        
        # Solve it!
        self._solve(prob)
        
        return x.value


class SDP(ABC):
    """Base class for a semi-definite program (SDP) - for computing counterfactuals.

    Attributes
    ----------
    epsilon : `float`
        "Small" non-negative number for relaxing strict inequalities.
    """
    def __init__(self, **kwds):
        self.epsilon = 1e-2
        self.solver = cp.SCS
        self.solver_verbosity = False

        super().__init__(**kwds)
    
    @abstractmethod
    def _build_constraints(self, var_X, var_x, y):
        """Creates and returns all constraints.

        Parameters
        ----------
        var_X : `cvx.Variable`
            The artificial optimization variable X - a symmetric matrix (see paper for details).
        var_x : `cvx.Variable`
            Optimization variable.
        y : `int` or `float`
            The requested prediction of the counterfactual - e.g. a class label.

        Returns
        -------
        `list`
            List of cvxpy constraints.
        """
        raise NotImplementedError()
    
    def _solve(self, prob):
        prob.solve(solver=self.solver, verbose=self.solver_verbosity)

    def build_solve_opt(self, x_orig, y, features_whitelist=None, optimizer_args=None):
        """Builds and solves the SDP.
        
        Parameters
        ----------
        x_orig : `numpy.ndarray`
            The original data point.
        y : `int` or `float`
            The requested prediction of the counterfactual - e.g. a class label.
        features_whitelist : `list(int)`, optional
            List of feature indices (dimensions of the input space) that can be used when computing the counterfactual.
        
            If `features_whitelist` is None, all features can be used.

            The default is None.
        optimizer_args : `dict`, optional
            Dictionary for overriding the default hyperparameters of the optimization algorithm.

            The default is None.

        Returns
        -------
        `numpy.ndarray`
            The solution of the optimization problem.

            If no solution exists, `None` is returned.
        """
        if optimizer_args is not None:
            if "epsilon" in optimizer_args:
                self.epsilon = optimizer_args["epsilon"]
            if "solver" in optimizer_args:
                self.solver = cvx_desc_to_solver(optimizer_args["solver"])
            if "solver_verbosity" in optimizer_args:
                self.solver_verbosity = optimizer_args["solver_verbosity"]

        dim = x_orig.shape[0]

        # Variables
        X = cp.Variable((dim, dim), symmetric=True)
        x = cp.Variable((dim, 1))
        one = np.array([[1]]).reshape(1, 1)
        I = np.eye(dim)

        # Construct constraints
        constraints = self._build_constraints(X, x, y)
        constraints += [cp.bmat([[X, x], [x.T, one]]) >> 0]

        # If requested, fix some features
        if features_whitelist is not None:
            A = []
            a = []

            for j in range(dim):
                if j not in features_whitelist:
                    t = np.zeros(dim)
                    t[j] = 1.
                    A.append(t)
                    a.append(x_orig[j])
            
            if len(A) != 0:
                A = np.array(A)
                a = np.array(a)

                constraints += [A @ x == a]

        # Build the final program
        f = cp.Minimize(cp.trace(I @ X) - 2. * x.T @ x_orig)
        prob = cp.Problem(f, constraints)
        
        # Solve it!
        self._solve(prob)

        return x.value.reshape(dim)


class DCQP(SupportAffinePreprocessing):
    """Class for a difference-of-convex-quadratic program (DCQP) - for computing counterfactuals.

    .. math:: \\underset{\\vec{x} \\in \\mathbb{R}^d}{\\min} \\vec{x}^\\top Q_0 \\vec{x} + \\vec{q}^\\top \\vec{x} + c - \\vec{x}^\\top Q_1 \\vec{x} \\quad \\text{s.t. } \\vec{x}^\\top A0_i \\vec{x} + \\vec{x}^\\top \\vec{b_i} + r_i - \\vec{x}^\\top A1_i \\vec{x} \\leq 0 \\; \\forall\\,i

    Attributes
    ----------
    pccp : instance of :class:`ceml.optim.cvx.PenaltyConvexConcaveProcedure`
        Implementation of the penalty convex-concave procedure for approximately solving the DCQP.
    epsilon : `float`
        "Small" non-negative number for relaxing strict inequalities.
    """
    def __init__(self, **kwds):
        self.pccp = None

        super().__init__(**kwds)

    def build_program(self, model, x_orig, y_target, Q0, Q1, q, c, A0_i, A1_i, b_i, r_i, features_whitelist=None, mad=None, optimizer_args=None):
        """Builds the DCQP.

        Parameters
        ----------
        model : `object`
            The model that is used for computing the counterfactual - must provide a method `predict`.
        x : `numpy.ndarray`
            The data point `x` whose prediction has to be explained.
        y_target : `int` or `float`
            The requested prediction of the counterfactual - e.g. a class label.
        Q0 : `numpy.ndarray`
            The matrix Q_0 of the DCQP.
        Q1 : `numpy.ndarray`
            The matrix Q_1 of the DCQP.
        q : `numpy.ndarray`
            The vector q of the DCQP.
        c : `float`
            The constant c of the DCQP.
        A0_i : `list(numpy.ndarray)`
            List of matrices A0_i of the DCQP.
        A1_i : `list(numpy.ndarray)`
            List of matrices A1_i of the DCQP.
        b_i : `list(numpy.ndarray)`
            List of vectors b_i of the DCQP.
        r_i : `list(float)`
            List of constants r_i of the DCQP.
        features_whitelist : `list(int)`, optional
            List of feature indices (dimensions of the input space) that can be used when computing the counterfactual.
        
            If `features_whitelist` is None, all features can be used.

            The default is None.
        mad : `numpy.ndarray`, optional
            Weights for the weighted Manhattan distance.

            If `mad` is None, the Euclidean distance is used.

            The default is None.
        optimizer_args : `dict`, optional
            Dictionary for overriding the default hyperparameters of the optimization algorithm.

            The default is None.
        """
        self.x_orig = x_orig
        self.y_target = y_target
        self.pccp = PenaltyConvexConcaveProcedure(model, Q0, Q1, q, c, A0_i, A1_i, b_i, r_i, features_whitelist, mad, optimizer_args)

    def solve(self, x0):
        """Approximately solves the DCQP by using the penalty convex-concave procedure.

        Parameters
        ----------
        x0 : `numpy.ndarray`
            The initial data point for the penalty convex-concave procedure - this could be anything, however a "good" initial solution might lead to a better result.
        """
        self.pccp.set_affine_preprocessing(**self.get_affine_preprocessing())
        return self.pccp.compute_counterfactual(self.x_orig, self.y_target, x0)


class PenaltyConvexConcaveProcedure(SupportAffinePreprocessing):
    """Implementation of the penalty convex-concave procedure for approximately solving a DCQP.
    """
    def __init__(self, model, Q0, Q1, q, c, A0_i, A1_i, b_i, r_i, features_whitelist=None, mad=None, optimizer_args=None, **kwds):      
        self.model = model
        self.mad = mad
        self.features_whitelist = features_whitelist
        self.Q0 = Q0
        self.Q1 = Q1
        self.q = q
        self.c = c
        self.A0s = A0_i
        self.A1s = A1_i
        self.bs = b_i
        self.rs = r_i

        self.dim = None

        self.epsilon = 1e-2
        self.tao = 1.2
        self.tao_max = 100
        self.mu = 1.5

        self.solver = cp.SCS
        self.solver_verbosity = False

        if optimizer_args is not None:
            if "epsilon" in optimizer_args:
                self.epsilon = optimizer_args["epsilon"]
            if "tao" in optimizer_args:
                self.tao = optimizer_args["tao"]
            if "tao_max" in optimizer_args:
                self.tao_max = optimizer_args["tao_max"]
            if "mu" in optimizer_args:
                self.mu = optimizer_args["mu"]
            if "solver" in optimizer_args:
                self.solver = cvx_desc_to_solver(optimizer_args["solver"])
            if "solver_verbosity" in optimizer_args:
                self.solver_verbosity = optimizer_args["solver_verbosity"]
        
        if not(len(self.A0s) == len(self.A1s) and len(self.A0s) == len(self.bs) and len(self.rs) == len(self.bs)):
            raise ValueError("Inconsistent number of constraint parameters")
            
        super().__init__(**kwds)

    def _solve(self, prob):
        prob.solve(solver=self.solver, verbose=self.solver_verbosity)

    def solve_aux(self, xcf, tao, x_orig):
        try:
            self.dim = x_orig.shape[0]

            # Variables
            var_x = cp.Variable(self.dim)
            s = cp.Variable(len(self.A0s))

            var_x_prime = self._apply_affine_preprocessing_to_var(var_x)

            # Constants
            s_z = np.zeros(len(self.A0s))
            s_c = np.ones(len(self.A0s))
            c = np.ones(self.dim)

            # Build constraints
            constraints = []
            for i in range(len(self.A0s)):
                A = cp.quad_form(var_x_prime, self.A0s[i])
                q = var_x_prime.T @ self.bs[i]
                c = self.rs[i] + np.dot(xcf, np.dot(xcf, self.A1s[i])) - 2. * var_x_prime.T @ np.dot(xcf, self.A1s[i]) - s[i]

                constraints.append(A + q + c + self.epsilon <= 0)
            
            # If requested, fix some features
            if self.features_whitelist is not None:
                A = []
                a = []

                for j in range(self.dim):
                    if j not in self.features_whitelist:
                        t = np.zeros(self.dim)
                        t[j] = 1.
                        A.append(t)
                        a.append(x_orig[j])
                
                if len(A) != 0:
                    A = np.array(A)
                    a = np.array(a)

                    constraints += [A @ var_x == a]

            # Build the final program
            f = None
            if self.mad is not None:    # TODO: Right now, mad != 1 is not supported.
                f = cp.Minimize(cp.norm(var_x - x_orig, 1) + s.T @ (tao*s_c))
            else:
                f = cp.Minimize(cp.quad_form(var_x_prime, self.Q0) + self.q.T @ var_x_prime + self.c + np.dot(xcf, np.dot(xcf, self.Q1)) - 2. * var_x_prime.T @ np.dot(xcf, self.Q1) + s.T @ (tao*s_c))
            constraints += [s >= s_z]
        
            prob = cp.Problem(f, constraints)
        
            # Solve it!
            self._solve(prob)

            if var_x.value is None:
                raise Exception("No solution found!")
            else:
                return var_x.value
        except Exception as ex:
            logging.debug(str(ex))

            return x_orig

    def compute_counterfactual(self, x_orig, y_target, x0):
        ####################################
        # Penalty convex-concave procedure #
        ####################################

        # Initial feasible solution
        xcf = x0

        # Hyperparameters
        cur_tao = self.tao

        # Solve a bunch of CCPs
        while cur_tao < self.tao_max:
            cur_xcf = xcf
            if cur_xcf.shape == x_orig.shape:   # Apply transformation is necessary - xcf is computed in the original space which can be different from the space the model works on!
                cur_xcf = self._apply_affine_preprocessing_to_const(cur_xcf)

            xcf_ = self.solve_aux(cur_xcf, cur_tao, x_orig)
            xcf = xcf_

            if y_target == self.model.predict([self._apply_affine_preprocessing_to_const(xcf_)])[0]:
                break

            # Increase penalty parameter
            cur_tao *= self.mu
        
        return xcf


#################################################
# Stuff for computing plausible counterfactuals #
#################################################

class HighDensityEllipsoids:
    def __init__(self, X, X_densities, cluster_probs, means, covariances, density_threshold=None, optimizer_args=None, **kwds):
        self.X = X
        self.X_densities = X_densities
        self.density_threshold = density_threshold if density_threshold is not None else float("-inf")
        self.cluster_probs = cluster_probs
        self.means = means
        self.covariances = covariances

        self.epsilon = 1e-5
        self.solver = cp.SCS
        if optimizer_args is not None:
            if "epsilon" in optimizer_args:
                self.epsilon = optimizer_args["epsilon"]
            if "solver" in optimizer_args:
                self.solver = cvx_desc_to_solver(optimizer_args["solver"])

        super().__init__(**kwds)

    def compute_ellipsoids(self):        
        return self.build_solve_opt()
    
    def _solve(self, prob):
        prob.solve(solver=self.solver, verbose=False)

    def build_solve_opt(self):
        n_ellipsoids = self.cluster_probs.shape[1]
        n_samples = self.X.shape[0]
        
        # Variables
        r = cp.Variable(n_ellipsoids, pos=True)

        # Construct constraints
        constraints = []
        for i in range(n_ellipsoids):
            mu_i = self.means[i]
            cov_i = np.linalg.inv(self.covariances[i])

            for j in range(n_samples):
                if self.X_densities[j][i] >= self.density_threshold:  # At least as good as a requested NLL
                    x_j = self.X[j,:]
                    
                    a = (x_j - mu_i)
                    b = np.dot(a, np.dot(cov_i, a))
                    constraints.append(b <= r[i])

        # Build the final program
        f = cp.Minimize(cp.sum(r))
        prob = cp.Problem(f, constraints)

        # Solve it!
        self._solve(prob)

        return r.value


class PlausibleCounterfactualOfHyperplaneClassifier():
    def __init__(self, w, b, n_dims, **kwds):
        self.hyperplane_w = w
        self.hyperplane_b = b

        self.n_dims = n_dims
        self.gmm_weights = None
        self.gmm_means = None
        self.gmm_covariances = None
        self.ellipsoids_r = None
        self.projection_matrix = None
        self.projection_mean_sub = None
        self.density_constraint = None

        self.min_density = None
        self.epsilon = 1e-2
        self.solver = cp.SCS
        self.gmm_cluster_index = 0  # For internal use only!

        super().__init__(**kwds)

    def setup_plausibility_params(self, ellipsoids_r, gmm_weights, gmm_means, gmm_covariances, projection_matrix=None, projection_mean_sub=None, density_constraint=True, density_threshold=-85):
        self.gmm_weights = gmm_weights
        self.gmm_means = gmm_means
        self.gmm_covariances = gmm_covariances
        self.ellipsoids_r = ellipsoids_r
        self.projection_matrix = np.eye(self.n_dims) if projection_matrix is None else projection_matrix
        self.projection_mean_sub = np.zeros(self.n_dims) if projection_mean_sub is None else projection_mean_sub
        self.density_constraint = density_constraint

        self.min_density = density_threshold

    def _build_constraints_plausibility_opt(self, var_x, y):
        constraints = []
        if self.hyperplane_w.shape[0] > 1:
            for i in range(self.hyperplane_w.shape[0]):
                if i != y:
                    constraints += [(self.projection_matrix @ (var_x - self.projection_mean_sub)).T @ (self.hyperplane_w[i,:] - self.hyperplane_w[y,:]) + (self.hyperplane_b[i] - self.hyperplane_b[y]) + self.epsilon <= 0]
        else:
            if y == 0:
                return [(self.projection_matrix @ (var_x - self.projection_mean_sub)).T @ self.hyperplane_w.reshape(-1, 1) + self.hyperplane_b + self.epsilon <= 0]
            else:
                return [(self.projection_matrix @ (var_x - self.projection_mean_sub)).T @ self.hyperplane_w.reshape(-1, 1) + self.hyperplane_b - self.epsilon >= 0]

        return constraints

    def compute_plausible_counterfactual(self, x, y, regularizer="l1"):        
        mad = None
        if regularizer == "l1":
            mad = np.ones(x.shape[0])
        
        xcf = None
        s = float("inf")
        for i in range(self.gmm_weights[y].shape[0]):
            try:
                self.gmm_cluster_index = i
                xcf_ = self.build_solve_plausibility_opt(x, y, mad)
                if xcf_ is None:
                    continue

                s_ = None
                if regularizer == "l1":
                    s_ = np.sum(np.abs(xcf_ - x))
                else:
                    s_ = np.linalg.norm(xcf_ - x, ord=2)

                if s_ <= s:
                    s = s_
                    xcf = xcf_
            except Exception as ex:
                pass    # TODO: Proper exception handling
        return xcf
    
    def _solve_plausibility_opt(self, prob):
        prob.solve(solver=self.solver, verbose=False)

    def build_solve_plausibility_opt(self, x_orig, y, mad=None):
        dim = x_orig.shape[0]
        
        # Variables
        x = cp.Variable(dim)
        beta = cp.Variable(dim)

        # Constants
        c = np.ones(dim)
        z = np.zeros(dim)
        I = np.eye(dim)

        # Construct constraints
        constraints = self._build_constraints_plausibility_opt(x, y)

        if self.density_constraint is True:
            i = self.gmm_cluster_index
            x_i = self.gmm_means[y][i]
            cov = self.gmm_covariances[y][i]
            cov = np.linalg.inv(cov)

            constraints += [cp.quad_form(self.projection_matrix @ (x - self.projection_mean_sub) - x_i, cov) - self.ellipsoids_r[i] <= 0] # Numerically much more stable than the explicit density component constraint

        # If necessary, construct the weight matrix for the weighted Manhattan distance
        Upsilon = None
        if mad is not None:
            alpha = 1. / mad
            Upsilon = np.diag(alpha)

        # Build the final program
        f = None
        if mad is not None:
            f = cp.Minimize(c.T @ beta)    # Minimize (weighted) Manhattan distance
            constraints += [Upsilon @ (x - x_orig) <= beta, (-1. * Upsilon) @ (x - x_orig) <= beta, I @ beta >= z]
        else:
            f = cp.Minimize((1/2)*cp.quad_form(x, I) - x_orig.T@x)  # Minimize L2 distance
        
        prob = cp.Problem(f, constraints)

        # Solve it!
        self._solve_plausibility_opt(prob)

        return x.value