"""Tinycat custom base components"""

import collections
from six import string_types


__all__ = [
    "namedtuple_with_defaults",
    "look_up_operations",
    "damerau_levenshtein_distance",
]


def namedtuple_with_defaults(typename, field_names, default_values=()):
    """Efficient data structures with default attributes and fields

    Args:
        typename (str): name of the subclass
        field_names ([type]): name of the fields
        default_values (tuple, optional): Defaults to (). 
            default values for fields

    Returns:
        Subclass: subclass with typename and field_names specified
    """

    T = collections.namedtuple(typename, field_names)
    T.__new__.__defaults__ = (None,) * len(T._fields)
    if isinstance(default_values, collections.Mapping):
        prototype = T(**default_values)
    else:
        prototype = T(*default_values)
    T.__new__.__defaults__ = tuple(prototype)
    T.attributes = field_names
    return T


def look_up_operations(type_str, supported):
    assert isinstance(type_str, string_types), "unrecognised type string"
    if type_str in supported and isinstance(supported, dict):
        return supported[type_str]

    if type_str in supported and isinstance(supported, set):
        return type_str

    if isinstance(supported, set):
        set_to_check = supported
    elif isinstance(supported, dict):
        set_to_check = set(supported)
    else:
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
