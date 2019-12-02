# -*- coding: utf-8 -*-
import sklearn.pipeline
import sklearn_lvq

from .softmaxregression import SoftmaxRegression
from .naivebayes import GaussianNB
from .linearregression import LinearRegression
from .knn import KNN
from .lvq import LVQ
from .lda import Lda
from .qda import Qda
from ..model import ModelWithLoss
from ..backend.jax.preprocessing import StandardScaler, PCA, PolynomialFeatures, Normalizer, MinMaxScaler
from ..backend.jax.costfunctions import CostFunctionDifferentiable, RegularizedCost
from ..costfunctions import RegularizedCost as RegularizedCostNonDifferentiable
from .utils import build_regularization_loss
from .counterfactual import SklearnCounterfactual


class PipelineModel(ModelWithLoss):
    """Class for rebuilding/wrapping the :class:`sklearn.pipeline.Pipeline` class

    The :class:`PipelineModel` class rebuilds a pipeline model from a given list of `sklearn` models.

    Parameters
    ----------
    models : list(object)
        Ordered list of all `sklearn` models in the pipeline.

    Attributes
    ----------
    models : `list(objects)`
        Ordered list of all `sklearn` models in the pipeline.
    """
    def __init__(self, models):
        self.models = models

        super(PipelineModel, self).__init__()
    
    def predict(self, x):
        """Predicts the output of a given input.

        Computes the prediction of a given input `x`.

        Parameters
        ----------
        x : `numpy.ndarray`
            The input `x`.
        
        Returns
        -------
        `numpy.array`
            Output of the pipeline (might be scalar or smth. higher-dimensional).
        """
        pred = x
        for m in self.models:
            pred = m.predict(pred)

        return pred

    def get_loss(self, y_target, pred=None):
        """Creates and returns a loss function.

        Builds a cost function where the target is `y_target`.

        Parameters
        ----------
        y_target : `int` or `float`
            The requested output.
        pred : `callable`, optional
            A callable that maps an input to the output.

            If `pred` is None, the class method `predict` is used for mapping the input to the output.

            The default is None.
        
        Returns
        -------
        :class:`ceml.costfunctions.costfunctions.CostFunction`
            Initialized cost function. Target is set to `y_target`.
        """
        last_model = self.models[-1]
        if not isinstance(last_model, ModelWithLoss):
            raise TypeError(f"The last model in the pipeline has to e an instance of 'ceml.model.ModelWithLoss' but not of {type(last_model)}")

        return last_model.get_loss(y_target, pred)


class PipelineCounterfactual(SklearnCounterfactual):
    """Class for computing a counterfactual of a softmax regression model.

    See parent class :class:`ceml.sklearn.counterfactual.SklearnCounterfactual`.
    """
    def __init__(self, model):
        super(PipelineCounterfactual, self).__init__(model)

    def wrap_model(self, model):
        if isinstance(model, sklearn.preprocessing.data.StandardScaler):
            return StandardScaler(model.mean_ if model.with_mean else 0, model.scale_ if model.with_std else 1)
        elif isinstance(model, sklearn.preprocessing.data.RobustScaler):
            return StandardScaler(model.center_  if model.with_centering else 0, model.scale_ if model.with_scaling else 1)
        elif isinstance(model, sklearn.preprocessing.data.MaxAbsScaler):
            return StandardScaler(0, model.scale_)
        elif isinstance(model, sklearn.preprocessing.data.MinMaxScaler):
            return MinMaxScaler(model.min_, model.scale_)
        elif isinstance(model, sklearn.preprocessing.Normalizer):
            return Normalizer()
        elif isinstance(model, sklearn.decomposition.PCA):
            return PCA(model.components_)
        elif isinstance(model, sklearn.preprocessing.PolynomialFeatures):
            return PolynomialFeatures(model.powers_)
        elif isinstance(model, sklearn.linear_model.LogisticRegression):
            return SoftmaxRegression(model)
        elif isinstance(model, sklearn.linear_model.base.LinearModel):
            return LinearRegression(model)
        elif isinstance(model, sklearn.naive_bayes.GaussianNB):
            return GaussianNB(model)
        elif isinstance(model, sklearn.discriminant_analysis.LinearDiscriminantAnalysis):
            return Lda(model)
        elif isinstance(model, sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis):
            return Qda(model)
        elif isinstance(model, sklearn.tree.DecisionTreeClassifier) or isinstance(model, sklearn.tree.DecisionTreeRegressor):
            raise NotImplementedError()
        elif isinstance(model, sklearn.ensemble.RandomForestClassifier) or isinstance(model, sklearn.ensemble.RandomForestRegressor):
            raise NotImplementedError()
        elif isinstance(model, sklearn.neighbors.KNeighborsClassifier) or isinstance(model, sklearn.neighbors.KNeighborsRegressor):
            return KNN(model)
        elif any([isinstance(model, t) for t in [sklearn_lvq.GlvqModel, sklearn_lvq.GmlvqModel, sklearn_lvq.LgmlvqModel, sklearn_lvq.RslvqModel, sklearn_lvq.MrslvqModel, sklearn_lvq.LmrslvqModel]]):
            return LVQ(model)
        else:
            raise ValueError(f"{type(model)} is not supported")

    def rebuild_model(self, model):
        """Rebuild a :class:`sklearn.pipeline.Pipeline` model.

        Converts a :class:`sklearn.pipeline.Pipeline` into a :class:`ceml.sklearn.pipeline.PipelineModel`.

        Parameters
        ----------
        model : instance of :class:`sklearn.pipeline.Pipeline`
            The `sklearn` pipeline model. 

        Returns
        -------
        :class:`ceml.sklearn.pipeline.Pipeline`
            The wrapped pipeline model.
        """
        if not isinstance(model, sklearn.pipeline.Pipeline):
            raise TypeError(f"model has to be an instance of 'sklearn.pipeline.Pipeline' but not of {type(model)}")

        # Rebuild pipeline
        models = []
        for _, item in self.model.named_steps.items():
            if item is not None:
                models.append(self.wrap_model(item))
    
        return PipelineModel(models)
    
    def build_loss(self, regularization, x_orig, y_target, pred, grad_mask, C, input_wrapper):
        """Build a loss function.

        Overwrites the  `build_loss` method from base class :class:`ceml.sklearn.counterfactual.SklearnCounterfactual`.

        Parameters
        ----------
        regularization : `str` or :class:`ceml.costfunctions.costfunctions.CostFunction`
            Regularizer of the counterfactual. Penalty for deviating from the original input `x`.

            Supported values:
            
                - l1: Penalizes the absolute deviation.
                - l2: Penalizes the squared deviation.

            `regularization` can be a description of the regularization, an instance of :class:`ceml.costfunctions.costfunctions.CostFunction` (or :class:`ceml.costfunctions.costfunctions.DifferentiableCostFunction` if your cost function is differentiable) or None if no regularization is requested.

            If `regularization` is None, no regularization is used.
        x_orig : `numpy.array`
            The original input whose prediction has to be explained.
        y_target : `int` or `float`
            The requested output.
        pred : `callable`
            A callable that maps an input to the output.

            If `pred` is None, the class method `predict` is used for mapping the input to the output.
        grad_mask : `numpy.array`
            Gradient mask determining which dimensions can be used.
        C : `float` or `list(float)`
            The regularization strength. If `C` is a list, all values in `C` are tried and as soon as a counterfactual is found, this counterfactual is returned and no other values of `C` are tried.

            `C` is ignored if no regularization is used (`regularization=None`).
        input_wrapper : `callable`
            Converts the input (e.g. if we want to exclude some features/dimensions, we might have to include these missing features before applying any function to it).
        
        Returns
        -------
        :class:`ceml.costfunctions.costfunctions.CostFunction`
            Initialized cost function. Target is set to `y_target`.
        """
        regularization = build_regularization_loss(regularization, x_orig)
        penalize_output = self.mymodel.get_loss(y_target, pred)

        loss_grad = None
        if isinstance(penalize_output, CostFunctionDifferentiable):
            loss = RegularizedCost(regularization, penalize_output, C=C)
            loss_grad = loss.grad(grad_mask)
        else:
            loss = RegularizedCostNonDifferentiable(regularization, penalize_output)
        
        return loss, loss_grad


def pipeline_generate_counterfactual(model, x, y_target, features_whitelist=None, regularization="l1", C=1.0, optimizer="nelder-mead", return_as_dict=True, done=None):
    """Computes a counterfactual of a given input `x`.

    Parameters
    ----------
    model : a :class:`sklearn.pipeline.Pipeline` instance.
        The modelpipeline that is used for computing the counterfactual.
    x : `numpy.ndarray`
        The input `x` whose prediction has to be explained.
    y_target : `int` or `float` or a callable that returns True if a given prediction is accepted.
        The requested prediction of the counterfactual.
    features_whitelist : `list(int)`, optional
        List of feature indices (dimensions of the input space) that can be used when computing the counterfactual.
        
        If `features_whitelist` is None, all features can be used.

        The default is None.
    regularization : `str` or :class:`ceml.costfunctions.costfunctions.CostFunction`, optional
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

        Some models (see paper) support the use of mathematical programs for computing counterfactuals. In this case, you can use the option "mp" - please read the documentation of the corresponding model for further information.
    return_as_dict : `boolean`, optional
        If True, returns the counterfactual, its prediction and the needed changes to the input as dictionary.
        If False, the results are returned as a triple.

        The default is True.
    done : `callable`, optional
        A callable that returns `True` if a counterfactual with a given output/prediction is accepted and `False` otherwise.

        If `done` is None, the output/prediction of the counterfactual must match `y_target` exactly.

        The default is None.

        Note
        ----
        In case of a regression it might not always be possible to achieve a given output/prediction exactly.

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
    cf = PipelineCounterfactual(model)

    return cf.compute_counterfactual(x, y_target, features_whitelist, regularization, C, optimizer, return_as_dict, done)
