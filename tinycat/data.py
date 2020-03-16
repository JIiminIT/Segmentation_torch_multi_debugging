"""Tinycat custom data structures and utilities"""

import collections
import argparse
from typing import Mapping, Iterator
from six import string_types


__all__ = [
    "gen_batch",
    "namedtuple_with_defaults",
    "look_up_operations",
    "damerau_levenshtein_distance",
]


TRUE_VALUE = {"yes", "true", "t", "y", "1"}
FALSE_VALUE = {"no", "false", "f", "n", "0"}


def str2boolean(string_input: str) -> bool:
    """
    convert user input config string to boolean

    :param string_input: any string in TRUE_VALUE or FALSE_VALUE
    :return: True or False
    """
    if string_input.lower() in TRUE_VALUE:
        return True
    if string_input.lower() in FALSE_VALUE:
        return False
    raise argparse.ArgumentTypeError(
        "Boolean value expected, received {}".format(string_input)
    )


def gen_batch(iterable: Mapping, batch_size: int = 1) -> Iterator[Mapping]:
    """Yields batches.
    Args:
        iterable (Mapping): sized object.
        batch_size (int, optional): Defaults to 1.

    Returns:
        Iterator[Mapping]: batch of iterator
    """
    length = len(iterable)
    for ndx in range(0, length, batch_size):
        yield iterable[ndx : min(ndx + batch_size, length)]


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
    """supported 배열 내에서 type_str이 matching하는 케이스를 조회하고
    존재하지 않는 경우 가장 근접한 type_str을 유추하여 오류를 raise

    Args:
        type_str (str): 입력된 인자
        supported (List): 지원하는 함수, argument 등

    Raises:
        ValueError: 잘못된 인자에 대한 오류

    Returns:
        type_str(str): 올바른 인자인 경우 반환
    """
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


# pylint: disable=invalid-name
def damerau_levenshtein_distance(x, y):
    """
    Calculates an edit distance, for typo detection. Code based on :
    https://en.wikipedia.org/wiki/Damerau–Levenshtein_distance
    """
    distances = {}
    x_length = len(x)
    y_length = len(y)

    for i in range(-1, x_length + 1):
        distances[(i, -1)] = i + 1
    for j in range(-1, y_length + 1):
        distances[(-1, j)] = j + 1

    for i in range(x_length):
        for j in range(y_length):
            if x[i] == y[j]:
                cost = 0
            else:
                cost = 1
            distances[(i, j)] = min(
                distances[(i - 1, j)] + 1,  # deletion
                distances[(i, j - 1)] + 1,  # insertion
                distances[(i - 1, j - 1)] + cost,  # substitution
            )
            if i and j and x[i] == y[j - 1] and x[i - 1] == y[j]:
                # transposition
                distances[(i, j)] = min(
                    distances[(i, j)], distances[i - 2, j - 2] + cost
                )

    return distances[x_length - 1, y_length - 1]
