# -*- coding: utf-8 -*-
import logging
from abc import ABC, abstractmethod
import numpy as np
import cvxpy as cp


class MathematicalProgram():
    """Base class for a mathematical program.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def solve(self):
        raise NotImplementedError()


class ConvexQuadraticProgram(ABC):
    """Base class for a convex quadratic program - for computing counterfactuals.

    Attributes
    ----------
    epsilon : `float`
        "Small" non-negative number for relaxing strict inequalities.
    """
    def __init__(self):
        self.epsilon = 1e-2

        super().__init__()
    
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
        prob.solve(solver=cp.SCS, verbose=False)

    def build_solve_opt(self, x_orig, y, features_whitelist=None, mad=None):
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
        
        Returns
        -------
        `numpy.ndarray`
            The solution of the optimization problem.

            If no solution exists, `None` is returned.
        """
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
    def __init__(self):
        self.epsilon = 1e-2

        super().__init__()
    
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
        prob.solve(solver=cp.SCS, verbose=False)

    def build_solve_opt(self, x_orig, y, features_whitelist=None):
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

        Returns
        -------
        `numpy.ndarray`
            The solution of the optimization problem.

            If no solution exists, `None` is returned.
        """
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


class DCQP():
    """Class for a difference-of-convex-quadratic program (DCQP) - for computing counterfactuals.

    .. math:: \\underset{\\vec{x} \\in \\mathbb{R}^d}{\\min} \\vec{x}^\\top Q_0 \\vec{x} + \\vec{q}^\\top \\vec{x} + c - \\vec{x}^\\top Q_1 \\vec{x} \\quad \\text{s.t. } \\vec{x}^\\top A0_i \\vec{x} + \\vec{x}^\\top \\vec{b_i} + r_i - \\vec{x}^\\top A1_i \\vec{x} \\leq 0 \\; \\forall\\,i

    Attributes
    ----------
    pccp : instance of :class:`ceml.optim.cvx.PenaltyConvexConcaveProcedure`
        Implementation of the penalty convex-concave procedure for approximately solving the DCQP.
    epsilon : `float`
        "Small" non-negative number for relaxing strict inequalities.
    """
    def __init__(self):
        self.pccp = None
        self.epsilon = 1e-2

        super().__init__()

    def build_program(self, model, x_orig, y_target, Q0, Q1, q, c, A0_i, A1_i, b_i, r_i, features_whitelist=None, mad=None):
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
        """
        self.x_orig = x_orig
        self.y_target = y_target
        self.pccp = PenaltyConvexConcaveProcedure(model, Q0, Q1, q, c, A0_i, A1_i, b_i, r_i, features_whitelist, mad)

    def solve(self, x0, tao=1.2, tao_max=100, mu=1.5):
        """Approximately solves the DCQP by using the penalty convex-concave procedure.

        Parameters
        ----------
        x0 : `numpy.ndarray`
            The initial data point for the penalty convex-concave procedure - this could be anything, however a "good" initial solution might lead to a better result.
        tao : `float`, optional
            Hyperparameter - see paper for details.

            The default is 1.2
        tao_max : `float`, optional
            Hyperparameter - see paper for details.

            The default is 100
        mu : `float`, optional
            Hyperparameter - see paper for details.

            The default is 1.5
        """
        return self.pccp.compute_counterfactual(self.x_orig, self.y_target, x0, tao=1.2, tao_max=100, mu=1.5)


class PenaltyConvexConcaveProcedure():
    """Implementation of the penalty convex-concave procedure for approximately solving a DCQP.
    """
    def __init__(self, model, Q0, Q1, q, c, A0_i, A1_i, b_i, r_i, features_whitelist=None, mad=None):      
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
        
        if not(len(self.A0s) == len(self.A1s) and len(self.A0s) == len(self.bs) and len(self.rs) == len(self.bs)):
            raise ValueError("Inconsistent number of constraint parameters")

    def _solve(self, prob):
        prob.solve(solver=cp.SCS, verbose=False)

    def solve_aux(self, xcf, tao, x_orig):
        try:
            self.dim = x_orig.shape[0]

            # Variables
            x = cp.Variable(self.dim)
            beta = cp.Variable(self.dim)
            s = cp.Variable(len(self.A0s))

            # Constants
            s_z = np.zeros(len(self.A0s))
            s_c = np.ones(len(self.A0s))
            z = np.zeros(self.dim)
            c = np.ones(self.dim)
            I = np.eye(self.dim)

            # Build constraints
            constraints = []
            for i in range(len(self.A0s)):
                A = cp.quad_form(x, self.A0s[i])
                q = x.T @ self.bs[i]
                c = self.rs[i] + np.dot(xcf, np.dot(xcf, self.A1s[i])) - 2. * x.T @ np.dot(xcf, self.A1s[i]) - s[i]

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

                    constraints += [A @ x == a]

            # If necessary, construct the weight matrix for the weighted Manhattan distance
            Upsilon = None
            if self.mad is not None:
                alpha = 1. / self.mad
                Upsilon = np.diag(alpha)

            # Build the final program
            f = None
            if self.mad is not None:    # TODO: Right now, mad != 1 is not supported.
                f = cp.Minimize(cp.norm(x - x_orig, 1) + s.T @ (tao*s_c))
            else:
                f = cp.Minimize(cp.quad_form(x, self.Q0) + self.q.T @ x + self.c + np.dot(xcf, np.dot(xcf, self.Q1)) - 2. * x.T @ np.dot(xcf, self.Q1) + s.T @ (tao*s_c))
            constraints += [s >= s_z]
        
            prob = cp.Problem(f, constraints)
        
            # Solve it!
            self._solve(prob)

            if x.value is None:
                raise Exception("No solution found!")
            else:
                return x.value
        except Exception as ex:
            logging.debug(str(ex))

            return x_orig

    def compute_counterfactual(self, x_orig, y_target, x0, tao, tao_max, mu):
        ####################################
        # Penalty convex-concave procedure #
        ####################################

        # Initial feasible solution
        xcf = x0

        # Hyperparameters
        cur_tao = tao

        # Solve a bunch of CCPs
        while cur_tao < tao_max:
            xcf_ = self.solve_aux(xcf, cur_tao, x_orig)
            xcf = xcf_

            if y_target == self.model.predict([xcf_])[0]:
                break

            # Increase penalty parameter
            cur_tao *= mu
        
        return xcf
