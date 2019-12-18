# -*- coding: utf-8 -*-
import logging
import numpy as np
import sklearn_lvq
import cvxpy as cp

from ..backend.jax.layer import create_tensor
from ..backend.jax.costfunctions import MinOfListDistCost, MinOfListDistExCost, RegularizedCost
from ..model import ModelWithLoss
from .utils import desc_to_dist
from .counterfactual import SklearnCounterfactual
from ..optim import MathematicalProgram, ConvexQuadraticProgram, DCQP


class LVQ(ModelWithLoss):
    """Class for rebuilding/wrapping the :class:`sklearn_lvq.GlvqModel`, :class:`sklearn_lvq.GmlvqModel`, :class:`sklearn_lvq.LgmlvqModel`, :class:`sklearn_lvq.RslvqModel`, :class:`sklearn_lvq.MrslvqModel` and :class:`sklearn_lvq.LmrslvqModel` classes.

    The :class:`LVQ` class rebuilds a `sklearn-lvq` lvq model.

    Parameters
    ----------
    model : instance of :class:`sklearn_lvq.GlvqModel`, :class:`sklearn_lvq.GmlvqModel`, :class:`sklearn_lvq.LgmlvqModel`, :class:`sklearn_lvq.RslvqModel`, :class:`sklearn_lvq.MrslvqModel` or :class:`sklearn_lvq.LmrslvqModel`
        The lvq model.
    dist : `str` or callable, optional
        Computes the distance between a prototype and a data point.

        Supported values:

            - l1: Penalizes the absolute deviation.
            - l2: Penalizes the squared deviation.

        You can use your own custom distance function by setting `dist` to a callable that can be called on a data point and returns a scalar.

        The default is "l2".

        **Note:** `dist` must not be None.

    Attributes
    ----------
    prototypes : `numpy.array`
        The prototypes.
    labels : `numpy.array`
        The labels of the prototypes. 
    dist : `callable`
        The distance function.
    model : `object`
        The original `sklearn-lvq` model.
    model_class : `class`
        The class of the `sklearn-lvq` model.
    dim  : `int`
        Dimensionality of the input data.

    Raises
    ------
    TypeError
        If `model` is not an instance of :class:`sklearn_lvq.GlvqModel`, :class:`sklearn_lvq.GmlvqModel`, :class:`sklearn_lvq.LgmlvqModel`, :class:`sklearn_lvq.RslvqModel`, :class:`sklearn_lvq.MrslvqModel` or :class:`sklearn_lvq.LmrslvqModel`
    """
    def __init__(self, model, dist="l2"):
        if not any([isinstance(model, t) for t in [sklearn_lvq.GlvqModel, sklearn_lvq.GmlvqModel, sklearn_lvq.LgmlvqModel, sklearn_lvq.RslvqModel, sklearn_lvq.MrslvqModel, sklearn_lvq.LmrslvqModel]]):
            raise TypeError(f"model has to be an instance of 'sklearn_lvq.GlvqModel', 'sklearn_lvq.GmlvqModel', 'sklearn_lvq.LgmlvqModel', 'sklearn_lvq.RslvqModel', 'sklearn_lvq.MrslvqModel' or 'sklearn_lvq.LmrslvqModel' but not of {type(model)}")
        
        self.prototypes = model.w_
        self.labels = model.c_w_
        self.dist = dist
        self.model = model
        self.model_class = model.__class__
        self.dim = model.w_[0].shape[0]

        # The model might have learned its own distance metric
        if isinstance(model, sklearn_lvq.GmlvqModel) or isinstance(model, sklearn_lvq.MrslvqModel):
            self.dist_mat = create_tensor(np.dot(model.omega_.T, model.omega_))
        elif isinstance(model, sklearn_lvq.LgmlvqModel) or isinstance(model, sklearn_lvq.LmrslvqModel):
            if model.classwise == True:
                self.omegas = [create_tensor(np.dot(omega.T, omega)) for omega in model.omegas_]
                self.classwise = True
                self.dist_mats = None
            else:
                self.dist_mats = [create_tensor(np.dot(omega.T, omega)) for omega in model.omegas_]
                self.classwise = False

        super(LVQ, self).__init__()

    def _get_omega(self):
        if isinstance(self.model, sklearn_lvq.GlvqModel) or isinstance(self.model, sklearn_lvq.RslvqModel):
            return np.eye(self.dim)
        elif isinstance(self.model, sklearn_lvq.GmlvqModel) or isinstance(self.model, sklearn_lvq.MrslvqModel):
            return np.dot(self.model.omega_.T, self.model.omega_)
        elif isinstance(self.model, sklearn_lvq.LgmlvqModel) or isinstance(self.model, sklearn_lvq.LmrslvqModel):
            raise TypeError("A localized model has more than one distance matrix - calling `_get_omega` is ambiguous.")

    def predict(self, x):
        """
        Note
        ----
        This function is a placeholder only.

        This function does not predict anything and just returns the given input.
        """
        return x    # Note: Identity function is necessary because our lvq loss function works on the input (not on the final classification) 

    def get_loss(self, y_target, pred=None):
        """Creates and returns a loss function.

        Builds a cost function where we penalize the minimum distance to the nearest prototype which is consistent with the target `y_target`.

        Parameters
        ----------
        y_target : `int`
            The target class.
        pred : `callable`, optional
            A callable that maps an input to an input. E.g. using the :class:`ceml.optim.input_wrapper.InputWrapper` class.

            If `pred` is None, no transformation is applied to the input before putting it into the loss function.

            The default is None.
        
        Returns
        -------
        :class:`ceml.backend.jax.costfunctions.MinOfListDistCost`
            Initialized cost function. Target label is `y_target`.
        """
        # Collect all prototypes of the target class
        if callable(y_target):
            target_samples = create_tensor(self.prototypes[[y_target(y) for y in self.labels], :])
        else:
            target_samples = create_tensor(self.prototypes[self.labels == y_target, :])

        # Build a loss function that penalize the distance to the nearest prototype - note that the model might have learned its own distance matrix
        if self.model_class == sklearn_lvq.GmlvqModel or self.model_class == sklearn_lvq.MrslvqModel:
            self.dist_mats = [self.dist_mat for _ in target_samples]

            return MinOfListDistExCost(self.dist_mats, target_samples, pred)
        elif self.model_class == sklearn_lvq.LgmlvqModel or self.model_class == sklearn_lvq.LmrslvqModel:
            if self.classwise == True:
                self.dist_mats = []
                if callable(y_target):
                    for y in self.labels:
                        if y_target(y) is True:
                            self.dist_mats.append(self.omegas[y])
                else:
                    for y in self.labels:
                        if y == y_target:
                            self.dist_mats.append(self.omegas[y])
            
            return MinOfListDistExCost(self.dist_mats, target_samples, pred)
        else:
            if not callable(self.dist):
                self.dist = desc_to_dist(self.dist)

            return MinOfListDistCost(self.dist, target_samples, pred)


class CQPHelper(ConvexQuadraticProgram):
    def __init__(self, mymodel, x_orig, y_target, indices_other_prototypes, features_whitelist=None, regularization="l1"):
        self.mymodel = mymodel
        self.features_whitelist = features_whitelist
        self.regularization = regularization
        self.x_orig = x_orig
        self.y_target = y_target
        self.other_prototypes = [self.mymodel.prototypes[i] for i in indices_other_prototypes]
        self.target_prototype = -1

        super(CQPHelper, self).__init__()

    def _build_constraints(self, var_x, y):
        Omega = self.mymodel._get_omega()
        p_i = self.mymodel.prototypes[self.target_prototype]

        results = []
        for k in range(len(self.other_prototypes)):
            p_j = self.other_prototypes[k]
            qj = np.dot(Omega, p_j - p_i)
            bj = -.5 * (np.dot(p_i, np.dot(Omega, p_i)) - np.dot(p_j, np.dot(Omega, p_j)))
            results.append(qj.T @ var_x + bj + self.epsilon <= 0)

        return results

    def solve(self, target_prototype, features_whitelist=None):
        self.target_prototype = target_prototype
        
        return self.build_solve_opt(self.x_orig, self.y_target, features_whitelist, mad=None if self.regularization != "l1" else np.ones(self.mymodel.dim))


class LvqCounterfactual(SklearnCounterfactual, MathematicalProgram, DCQP):
    """Class for computing a counterfactual of a lvq model.

    See parent class :class:`ceml.sklearn.counterfactual.SklearnCounterfactual`.
    """
    def __init__(self, model, dist="l2", cqphelper=CQPHelper):
        self.dist = dist    # TODO: Extract distance from model
        self.cqphelper = cqphelper

        super(LvqCounterfactual, self).__init__(model)
    
    def rebuild_model(self, model):
        """Rebuilds a :class:`sklearn_lvq.GlvqModel`, :class:`sklearn_lvq.GmlvqModel`, :class:`sklearn_lvq.LgmlvqModel`, :class:`sklearn_lvq.RslvqModel`, :class:`sklearn_lvq.MrslvqModel` or :class:`sklearn_lvq.LmrslvqModel` model.

        Converts a :class:`sklearn_lvq.GlvqModel`, :class:`sklearn_lvq.GmlvqModel`, :class:`sklearn_lvq.LgmlvqModel`, :class:`sklearn_lvq.RslvqModel`, :class:`sklearn_lvq.MrslvqModel` or :class:`sklearn_lvq.LmrslvqModel` instance into a :class:`ceml.sklearn.lvq.LVQ` instance.

        Parameters
        ----------
        model : instace of :class:`sklearn_lvq.GlvqModel`, :class:`sklearn_lvq.GmlvqModel`, :class:`sklearn_lvq.LgmlvqModel`, :class:`sklearn_lvq.RslvqModel`, :class:`sklearn_lvq.MrslvqModel` or :class:`sklearn_lvq.LmrslvqModel`
            The `sklearn-lvq` lvq model. 

        Returns
        -------
        :class:`ceml.sklearn.lvq.LVQ`
            The wrapped lvq model.
        """
        if not any([isinstance(model, t) for t in [sklearn_lvq.GlvqModel, sklearn_lvq.GmlvqModel, sklearn_lvq.LgmlvqModel, sklearn_lvq.RslvqModel, sklearn_lvq.MrslvqModel, sklearn_lvq.LmrslvqModel]]):
            raise TypeError(f"model has to be an instance of 'sklearn_lvq.GlvqModel', 'sklearn_lvq.GmlvqModel', 'sklearn_lvq.LgmlvqModel', 'sklearn_lvq.RslvqModel', 'sklearn_lvq.MrslvqModel' or 'sklearn_lvq.LmrslvqModel' but not of {type(model)}")
    
        return LVQ(model, self.dist)

    def _compute_counterfactual_via_convex_quadratic_programming(self, x_orig, y_target, features_whitelist, regularization):
        xcf = None
        xcf_dist = float("inf")

        dist = lambda x: np.linalg.norm(x - x_orig, 2)
        if regularization == "l1":
            dist = lambda x: np.sum(np.abs(x - x_orig))
        
        # Search for suitable prototypes
        target_prototypes = []
        other_prototypes = []
        for i, l in zip(range(len(self.mymodel.labels)), self.mymodel.labels):
            if l == y_target:
                target_prototypes.append(i)
            else:
                other_prototypes.append(i)

        # Compute a counterfactual for each prototype
        solver = CQPHelper(mymodel=self.mymodel, x_orig=x_orig, y_target=y_target, indices_other_prototypes=other_prototypes, features_whitelist=features_whitelist, regularization=regularization)
        for i in range(len(target_prototypes)):
            try:
                xcf_ = solver.solve(target_prototype=i, features_whitelist=features_whitelist)
                ycf_ = self.mymodel.model.predict([xcf_])[0]

                if ycf_ == y_target:
                    if dist(xcf_) < xcf_dist:
                        xcf = xcf_
                        xcf_dist = dist(xcf_)
            except Exception as ex:
                logging.debug(str(ex))
    
        if xcf is None:
            # It might happen that the solver (for a specific set of parameter values) does not find a counterfactual, although the feasible region is always non-empty
            j = np.argmin([dist(self.mymodel.prototypes[proto]) for proto in target_prototypes]) # Select the nearest prototype!
            xcf = self.mymodel.prototypes[j]

        return xcf

    def _build_solve_dcqp(self, x_orig, y_target, target_prototype_id, other_prototypes, features_whitelist, regularization):
        p_i = self.mymodel.prototypes[target_prototype_id]
        o_i = self.mymodel.dist_mats[target_prototype_id] if not self.mymodel.classwise else self.mymodel.omegas[y_target]
        ri = .5 * np.dot(p_i, np.dot(o_i, p_i))
        qi = np.dot(o_i, p_i)

        Q0 = None
        Q1 = None
        q = None
        c = None
        if regularization == "l2":
            Q0 = np.eye(self.mymodel.dim)
            Q1 = np.zeros((self.mymodel.dim, self.mymodel.dim))
            q = -x_orig
            c = 0.0

        A0_i = []
        A1_i = []
        b_i = []
        r_i = []
        for j in other_prototypes:
            p_j = self.mymodel.prototypes[j]
            o_j = self.mymodel.omegas[self.mymodel.labels[j]] if self.mymodel.classwise else self.mymodel.dist_mats[j]

            q =  np.dot(o_j, p_j) - qi
            r = ri - .5 * np.dot(p_j, np.dot(o_j, p_j))
            
            A0_i.append(.5 * o_i)
            A1_i.append(.5 * o_j)
            b_i.append(q)
            r_i.append(r)
        
        self.build_program(self.model, x_orig, y_target, Q0, Q1, q, c, A0_i, A1_i, b_i, r_i, features_whitelist=features_whitelist, mad=None if regularization != "l1" else np.ones(self.mymodel.dim))

        return DCQP.solve(self, x0=p_i, tao=1.2, tao_max=100, mu=1.5)

    def _compute_counterfactual_via_dcqp(self, x_orig, y_target, features_whitelist, regularization):
        xcf = None
        xcf_dist = float("inf")

        dist = lambda x: np.linalg.norm(x - x_orig, 2)
        if regularization == "l1":
            dist = lambda x: np.sum(np.abs(x - x_orig))
        
        # Search for suitable prototypes
        target_prototypes = []
        other_prototypes = []
        for i, l in zip(range(len(self.mymodel.labels)), self.mymodel.labels):
            if l == y_target:
                target_prototypes.append(i)
            else:
                other_prototypes.append(i)
        
        # Compute a counterfactual for each prototype
        for i in range(len(target_prototypes)):
            try:
                xcf_ = self._build_solve_dcqp(x_orig, y_target, i, other_prototypes, features_whitelist, regularization)
                ycf_ = self.model.predict([xcf_])[0]

                if ycf_ == y_target:
                    if dist(xcf_) < xcf_dist:
                        xcf = xcf_
                        xcf_dist = dist(xcf_)
            except Exception as ex:
                logging.debug(str(ex))
    
        if xcf is None:
            # It might happen that the solver (for a specific set of parameter values) does not find a counterfactual, although the feasible region is always non-empty
            j = np.argmin([dist(self.mymodel.prototypes[proto]) for proto in target_prototypes]) # Select the nearest prototype!
            xcf = self.mymodel.prototypes[j]

        return xcf

    def solve(self, x_orig, y_target, regularization, features_whitelist, return_as_dict):
        xcf = None
        if isinstance(self.model, sklearn_lvq.LgmlvqModel) or isinstance(self.model, sklearn_lvq.LmrslvqModel):
            xcf = self._compute_counterfactual_via_dcqp(x_orig, y_target, features_whitelist, regularization)
        else:
            xcf = self._compute_counterfactual_via_convex_quadratic_programming(x_orig, y_target, features_whitelist, regularization)
        delta = x_orig - xcf

        if self.model.predict([xcf]) != y_target:
            raise Exception("No counterfactual found - Consider changing parameters 'regularization', 'features_whitelist', 'optimizer' and try again")

        return self.__build_result_dict(xcf, y_target, delta) if return_as_dict else xcf, y_target, delta


def lvq_generate_counterfactual(model, x, y_target, features_whitelist=None, dist="l2", regularization="l1", C=1.0, optimizer="nelder-mead", return_as_dict=True, done=None):
    """Computes a counterfactual of a given input `x`.

    Parameters
    ----------
    model : a :class:`sklearn.neighbors.sklearn_lvq.GlvqModel`, :class:`sklearn_lvq.GmlvqModel`, :class:`sklearn_lvq.LgmlvqModel`, :class:`sklearn_lvq.RslvqModel`, :class:`sklearn_lvq.MrslvqModel` or :class:`sklearn_lvq.LmrslvqModel` instance.
        The lvq model that is used for computing the counterfactual.

        **Note:** Only lvq models from sklearn-lvq are supported.
    x : `numpy.ndarray`
        The input `x` whose prediction has to be explained.
    y_target : `int` or `float` or a callable that returns True if a given prediction is accepted.
        The requested prediction of the counterfactual.
    features_whitelist : `list(int)`, optional
        List of feature indices (dimensions of the input space) that can be used when computing the counterfactual.
        
        If `features_whitelist` is None, all features can be used.

        The default is None.
    dist : `str` or callable, optional
        Computes the distance between a prototype and a data point.
        
        Supported values:
        
            - l1: Penalizes the absolute deviation.
            - l2: Penalizes the squared deviation.
        
        You can use your own custom distance function by setting `dist` to a callable that can be called on a data point and returns a scalar.

        The default is "l1".

        **Note:** `dist` must not be None.
    regularization : `str` or callable, optional
        Regularizer of the counterfactual. Penalty for deviating from the original input `x`.
        Supported values:

            - l1: Penalizes the absolute deviation.
            - l2: Penalizes the squared deviation.
        
        `regularization` can be a description of the regularization, an instance of :class:`ceml.costfunctions.costfunctions.CostFunction` (or :class:`ceml.costfunctions.costfunctions.CostFunctionDifferentiable` if your cost function is differentiable) or None if no regularization is requested.

        If `regularization` is None, no regularization is used.

        The default is "l1".
    C : `float` or `list(float)`, optional
        The regularization strength. If `C` is a list, all values in `C` are tried and as soon as a counterfactual is found, this counterfactual is returned and no other values of `C` are tried.

        `C` is ignored if no regularization is used (`regularization=None`).

        The default is 1.0
    optimizer : `str` or instance of :class:`ceml.optim.optimizer.Optimizer`, optional
        Name/Identifier of the optimizer that is used for computing the counterfactual.
        See :func:`ceml.optim.optimizer.prepare_optim` for details.

        As an alternative, we can use any (custom) optimizer that is derived from the :class:`ceml.optim.optimizer.Optimizer` class.

        The default is "nelder-mead".

        Learning vector quantization supports the use of mathematical programs for computing counterfactuals - set `optimizer` to "mp" for using a convex quadratic program (G(M)LVQ) or a DCQP (otherwise) for computing the counterfactual.
        Note that in this case the hyperparameter `C` is ignored.
        Because the DCQP is a non-convex problem, we are not guaranteed to find the best solution (it might even happen that we do not find a solution at all) - we use the penalty convex-concave procedure for approximately solving the DCQP.
    return_as_dict : `boolean`, optional
        If True, returns the counterfactual, its prediction and the needed changes to the input as dictionary.
        If False, the results are returned as a triple.

        The default is True.
    done : `callable`, optional
        Not used.

    Returns
    -------
    `dict` or `triple`
        A dictionary where the counterfactual is stored in 'x_cf', its prediction in 'y_cf' and the changes to the original input in 'delta'.

        (x_cf, y_cf, delta) : triple if `return_as_dict` is False
    
    Raises
    ------
    Exception
        If no counterfactual was found.
    """
    cf = LvqCounterfactual(model, dist)

    return cf.compute_counterfactual(x, y_target, features_whitelist, regularization, C, optimizer, return_as_dict)    
