# -*- coding: utf-8 -*-
import numpy as np
from six import string_types


# Print iterations progress
def print_progress_bar(
        iteration, total, prefix="", suffix="", decimals=1, length=10, fill="="
):
    """
    Call in a loop to create terminal progress bar

    :param iteration: current iteration (Int)
    :param total: total iterations (Int)
    :param prefix: prefix string (Str)
    :param suffix: suffix string (Str)
    :param decimals: number of decimals in percent complete (Int)
    :param length: character length of bar (Int)
    :param fill: bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bars = fill * filledLength + "-" * (length - filledLength)
    print("\r%s |%s| %s%% %s" % (prefix, bars, percent, suffix), end="\r")
    # Print New Line on Complete
    if iteration == total:
        print("\n")


def damerau_levenshtein_distance(s1, s2):
    """
    Calculates an edit distance, for typo detection. Code based on :
    https://en.wikipedia.org/wiki/Damerau–Levenshtein_distance
    """
    d = {}
    string_1_length = len(s1)
    string_2_length = len(s2)
    for i in range(-1, string_1_length + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, string_2_length + 1):
        d[(-1, j)] = j + 1

    for i in range(string_1_length):
        for j in range(string_2_length):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1,  # deletion
                d[(i, j - 1)] + 1,  # insertion
                d[(i - 1, j - 1)] + cost,  # substitution
            )
            if i and j and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                d[(i, j)] = min(d[(i, j)], d[i - 2, j - 2] + cost)  # transposition

    return d[string_1_length - 1, string_2_length - 1]


def look_up_operations(type_str, supported):
    """
    This function validates the ``type_str`` against the supported set.

    if ``supported`` is a ``set``, returns ``type_str``
    if ``supported`` is a ``dict``, return ``supported[type_str]``
    else:
    raise an error possibly with a guess of the closest match.

    :param type_str:
    :param supported:
    :return:
    """
    assert isinstance(type_str, string_types), "unrecognised type string"
    if isinstance(supported, dict) and type_str in supported:
        return supported[type_str]

    if isinstance(supported, set) and type_str in supported:
        return type_str

    try:
        set_to_check = set(supported)
    except TypeError:
        set_to_check = set()

    edit_distances = {}
    for supported_key in set_to_check:
        edit_distance = damerau_levenshtein_distance(supported_key, type_str)
        if edit_distance <= 3:
            edit_distances[supported_key] = edit_distance
    if edit_distances:
        guess_at_correct_spelling = min(edit_distances, key=edit_distances.get)
        raise ValueError(
            'By "{0}", did you mean "{1}"?\n'
            '"{0}" is not a valid option.\n'
            "Available options are {2}\n".format(
                type_str, guess_at_correct_spelling, supported
            )
        )
    else:
        raise ValueError(
            'No supported option "{}" '
            "is not found.\nAvailable options are {}\n".format(type_str, supported)
        )


def otsu_threshold(img, nbins=256):
    """
    Implementation of otsu thresholding

    :param img:
    :param nbins:
    :return:
    """
    hist, bin_edges = np.histogram(img.ravel(), bins=nbins)
    hist = hist.astype(float)
    half_bin_size = (bin_edges[1] - bin_edges[0]) * 0.5
    bin_centers = bin_edges[:-1] + half_bin_size

    weight_1 = np.copy(hist)
    mean_1 = np.copy(hist)
    weight_2 = np.copy(hist)
    mean_2 = np.copy(hist)
    for i in range(1, hist.shape[0]):
        weight_1[i] = weight_1[i - 1] + hist[i]
        mean_1[i] = mean_1[i - 1] + hist[i] * bin_centers[i]

        weight_2[-i - 1] = weight_2[-i] + hist[-i - 1]
        mean_2[-i - 1] = mean_2[-i] + hist[-i - 1] * bin_centers[-i - 1]

    target_max = 0
    threshold = bin_centers[0]
    for i in range(0, hist.shape[0] - 1):
        ratio_1 = mean_1[i] / weight_1[i]
        ratio_2 = mean_2[i + 1] / weight_2[i + 1]
        target = weight_1[i] * weight_2[i + 1] * (ratio_1 - ratio_2) ** 2
        if target > target_max:
            target_max, threshold = target, bin_centers[i]
    return threshold
