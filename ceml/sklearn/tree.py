# -*- coding: utf-8 -*-
import numpy as np
import sklearn


def get_leafs_from_tree(tree_, classifier=False):
    leafs = []
    def traversal(node, path):
        if tree_.feature[node] != sklearn.tree._tree.TREE_UNDEFINED:
            feature_id = tree_.feature[node]
            threshold = tree_.threshold[node]

            traversal(tree_.children_left[node], path + [(node, feature_id, threshold, "<")])
            traversal(tree_.children_right[node], path + [(node, feature_id, threshold, ">")])
        else:
            if classifier is False:
                path += [(node, -1, tree_.value[node].flatten()[0])]
            else:
                path += [(node, -1, np.argmax(tree_.value[node]))]
            leafs.append(path)

    traversal(0, [])

    return leafs


def leaf_to_path(path_to_leaf):
    return [p[0] for p in path_to_leaf]


def compute_overlap_of_path(p1, p2):
    if p1 == [] or p2 == []:
        return 0
    if p1[0] != p2[0]:
        return 0
    else:
        return 1 + compute_overlap_of_path(p1[1:], p2[1:])


def compute_change(x, threshold, direction):
    eps = 1.e-5

    if direction == "<":
        if x >= threshold:
            return -1. * np.abs(threshold - x) - eps
        else:
            return 0
    elif direction == ">":
        if x <= threshold:
            return np.abs(threshold - x) + eps
        else:
            return 0


def compute_path_adjustment(x_orig, x_orig_path, target_path):
    r = {}
    overlap = 0
    y = target_path[-1][2]

    for i in range(len(target_path) - 1):
        if x_orig_path[i+1] != target_path[i+1][0]: # Starting point found?
            for j in range(i, len(target_path) - 1):
                feature_id = target_path[j][1]
                threshold = target_path[j][2]
                direction = target_path[j][3]
                x = x_orig[feature_id]

                delta = compute_change(x, threshold, direction)
                if delta != 0:
                    if feature_id in r:
                        if abs(delta) > abs(r[feature_id]):
                            r[feature_id] = delta
                    else:
                        r[feature_id] = delta

            break
        else:
            overlap += 1
    return overlap, y, r


def apply_adjustment(x_orig, adjustment):
    x_new = np.array(x_orig)

    for i, d in adjustment.items():
        x_new[i] += d

    return x_new


def score_adjustments(x_orig, x_orig_path, leafs_path, dist):
    r = []

    for leaf_path in leafs_path:
        _, y, adjustment = compute_path_adjustment(x_orig, x_orig_path, leaf_path)
        x_new = apply_adjustment(x_orig, adjustment)
        cost = dist(x_new)

        r.append((cost, y, adjustment))

    r.sort(key=lambda item: item[0])        

    return r
