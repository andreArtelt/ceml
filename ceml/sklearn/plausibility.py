# -*- coding: utf-8 -*-
import numpy as np

from ..optim.cvx import HighDensityEllipsoids


def estimate_densities_of_training_samples_via_gmm(X, y, gmms):
    densities = []
    densities_ex = []

    # Compute densities of all samples
    for i in range(X.shape[0]):
        gmm = gmms[y[i]]    # Select the class dependent GMM
        
        x = X[i,:]
        z = []
        dim = x.shape[0]
        for j in range(gmm.weights_.shape[0]):
            x_i = gmm.means_[j]
            w_i = gmm.weights_[j]
            cov = gmm.covariances_[j]
            cov = np.linalg.inv(cov)

            b = -2.*np.log(w_i) + dim*np.log(2.*np.pi) - np.log(np.linalg.det(cov))
            z.append(np.dot(x - x_i, np.dot(cov, x - x_i)) + b) # NLL
        
        densities.append(np.min(z))
        densities_ex.append(z)

    return np.array(densities), np.array(densities_ex)


def prepare_computation_of_plausible_counterfactuals(X, y, gmms, projection_mean_sub=None, projection_matrix=None, density_thresholds=None):
    """
    Computes all steps that are independent of a concrete sample when computing a plausible counterfactual explanations.
    Because the computation of a plausible counterfactual requires quite an amount of computation that does not depend on the concret sample we want to explain, it make sense to pre compute as much as possible (reduce redundant computations).

    Parameters
    ----------
    X : `numpy.ndarray`
        Data points.
    y : `numpy.ndarray`
        Labels of data points `X`. Assumed to be [0, 1, 2, ...].
    gmms : `list(int)`
        List of class dependent Gaussian Mixture Models (GMMs).
    projection_mean_sub : `numpy.ndarray`, optional
        The negative bias of the affine preprocessing.

        The default is None.
    projection_matrix : `numpy.ndarray`, optional
        The projection matrix of the affine preprocessing.

        The default is None.
    density_threshold : `float`, optional
        Density threshold at which we consider a counterfactual to be plausible.

        If no density threshold is specified (`density_threshold` is set to None), the median density of the samples `X` is chosen as a threshold.

        The default is None.

    Returns
    -------
    `dict`
        All necessary (pre computable) stuff needed for the computation of plausible counterfactuals.
    """
    results = {}
    results["use_density_constraints"] = True

    # If necessary, project the data
    X_ = X
    if projection_mean_sub is not None and projection_matrix is not None:
        X_ = np.dot(X - projection_mean_sub, projection_matrix.T)
    results["projection_mean_sub"] = projection_mean_sub
    results["projection_matrix"] = projection_matrix

    # Estimate densities under the approximated GMM
    densities, densities_ex = estimate_densities_of_training_samples_via_gmm(X_, y, gmms)

    # Create a list with all possible labels/classes
    y_targets = np.unique(y)
    results["y_targets"] = y_targets

    # Store GMMs
    results["gmm_weights"] = []
    results["gmm_means"] = []
    results["gmm_covariances"] = []
            
    for i in y_targets:
        results["gmm_weights"].append(np.array(gmms[i].weights_))
        results["gmm_means"].append(np.array(gmms[i].means_))
        results["gmm_covariances"].append(np.array(gmms[i].covariances_))

    # If no (class dependent) density thresholds are specified, choose the median density of the training samples
    if density_thresholds is None:
        density_thresholds = [np.median(densities[y == y_]) for y_ in y_targets]
    results["density_thresholds"] = density_thresholds

    # Compute (class dependent) high density ellipsoids - constraint: test if sample is included in ellipsoid -> this is the same as the proposed constraint but nummerically much more stable, in particular when we add a dimensionality reduction from a high dimensional space to a low dimensional space
    results["ellipsoids_r"] = []
    for y_ in y_targets:
        gmm = gmms[y_]    # Select the class dependent GMM
        idx = y == y_     # Select all samples of the current label

        cluster_prob_ = gmm.predict_proba(X_[idx,:])    # TODO: Currently not needed!
        r = HighDensityEllipsoids(X_[idx,:], densities_ex[idx], cluster_prob_, gmm.means_, gmm.covariances_, density_thresholds[list(y_targets).index(y_)]).compute_ellipsoids()
        results["ellipsoids_r"].append(r)

    return results