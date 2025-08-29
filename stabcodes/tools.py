from stabcodes.pauli import PauliOperator, Stabilizer2D
from typing import Iterable, Any, Hashable, Union
from collections.abc import Sized
import numpy as np


def pretty_print_operators(ops: Iterable[PauliOperator]):
    """
    Pretty-print of an iterable of PauliOperators

    Parameters
    ----------
    ops: Iterable[PauliOperator]
        Operators to display.

    """
    print()
    for op in ops:
        print(PauliOperator.__repr__(op))
    print()


def pretty_print_stabilizers(stabs: Iterable[Stabilizer2D]):
    """
    Pretty-print of an iterable of Stabilizer2D.

    Parameters
    ----------
    stabs: Iterable of Stabilizers
        Stabilizers to display.

    """
    print()
    for op in stabs:
        print(Stabilizer2D.__repr__(op))
    print()


def largest(lists: Iterable[Sized]) -> Sized:
    """
    Return the largest element of the given list of elements with len method.

    Parameters
    ----------
    lists: Iterable[Sized]
        Non-empty collections of sized elements from which to output the longest.
        
    Raises
    ------
    ValueError:
        `lists` parameter cannot be empty.

    Returns
    -------
    Sized
        The longest element.
        
    Examples
    --------
    >>> largest([range(5), [], (1,)])
    range(0, 5)
    >>> largest([])
    Traceback (most recent call last):
    ...
    ValueError: max() iterable argument is empty

    """
    return max(lists, key=len)


def cycle_to_operator(cycle: list[tuple[int, int]]) -> list[int]:
    """
    Convert a path of edges to a list of vertices. No verification performed.

    Parameters
    ----------
    cycle: list[tuple[int, int]]
    
    Returns
    -------
    list[int]
        List of consecutive vertices along the given path.
        
    Examples
    --------
    >>> cycle_to_operator([(0, 1), (1, 2), (2, 0)])
    [0, 1, 2]
    >>> cycle_to_operator([(0, 1), (2, 3)]) # improper use
    [0, 2]
    """
    return [e[0] for e in cycle]


def split_operator(operator: PauliOperator, support: list[int]) -> tuple[PauliOperator, PauliOperator]:
    """
    Splits a PauliOperator into two parts, restricting it on a support
    or complementing it.

    Parameters
    ----------
    operator: PauliOperator
        Operator to split.
    support: list[int]
        List of qubits to split along.

    Returns
    -------
    tuple[PauliOperator, PauliOperator]
        2 PauliOperators whose product equals the given one.
        
    Examples
    --------
    >>> P = PauliOperator.from_support([0, 1, 3], "X", 5)
    >>> (s := split_operator(P, [0, 1, 2]))
    (PauliOperator([X(0), X(1)], 5), PauliOperator([X(3)], 5))
    >>> P == s[0] * s[1]
    True

    """
    PA = []
    PB = []

    for p in operator.paulis:
        if p.qubit in support:
            PA.append(p)
        else:
            PB.append(p)

    return PauliOperator(PA, operator.nb_qubits), PauliOperator(PB, operator.nb_qubits)


def rotate_left(l: list[Any]):
    """
    Cyclically rotate a list in-place to the left.
    The second element becomes the first and the first the last.
    
    Parameters
    ----------
    l: list[Any]
        List to rotate to the
        
    Examples
    --------
    >>> rotate_left(l := [0, 1, 2])
    >>> l
    [1, 2, 0]
    >>> rotate_left(l := [])
    >>> l
    []

    """
    if l:
        l[:] = l[1:] + [l[0]]


def rotate_to_place_first(l: list[Any], element: Any):
    """
    Cyclically rotate a list in-place to place an element in the list in first position.
    
    Parameters
    ----------
    l: list[Any]
        List to rotate to the
    element: Any
        Element to put in first position
    
    Raises
    ------
    ValueError:
        Raised if given element is not in the list.

    Examples
    --------
    >>> rotate_to_place_first(l := [0, 1, 2], 2)
    >>> l
    [2, 0, 1]
    >>> rotate_to_place_first(l := [], 1)
    Traceback (most recent call last):
    ...
    ValueError: 1 is not in list

    """
    indice = l.index(element)
    l[:indice], l[len(l):] = [], l[:indice]


def reverse_dict(dic: dict[Hashable, Union[Hashable, list[Hashable]]]) -> dict[Hashable, Hashable]:
    """
    Unique hashable values of dic are returned as key in the new dictionnary,
    with new values being the former keys.

    Fails if a value in not hashable or if a value is the same for different keys.

    Parameters
    ----------
    dic: dict[Hashable, Hashable]
        Dictionary to be inverted.
        
    Raises
    ------
    TypeError
        Raised if a value is not hashable.
    ValueError
        Raised if some values are duplicated.
        
    Returns
    -------
    dict[Hashable, Hashable]
        Inverted dict.
        
    Examples
    --------
    >>> d = {"a": [1, 2], "b": 3}
    >>> reverse_dict(d)
    {1: 'a', 2: 'a', 3: 'b'}

    """
    new_dic = dict()
    for (key, value) in dic.items():
        if isinstance(value, list):
            for v in value:
                if v not in new_dic:
                    if not isinstance(v, Hashable):
                        raise TypeError(f"Unhashable type : {type(value)}")
                    new_dic[v] = key
        else:
            if value not in new_dic:
                if not isinstance(value, Hashable):
                    raise TypeError(f"Unhashable type : {type(value)}")
                new_dic[value] = key
            else:
                raise ValueError(f"Value {repr(value)} is present multiple times.")
    return new_dic


def transpositions_from_mapping(mapping: dict[int, int]) -> list[tuple[int, int]]:
    """
    Computes a list of transpositions implementing the given mapping.
    Mapping must be exhaustive from 0 to n-1.
    
    Parameters
    ----------
    mapping: dict[int, int]
        A valid mapping of qubits. Qubits must be 
        
    Returns
    -------
    list[tuple[int, int]]
        List of transpositions implementing the mapping.
        
    Examples
    --------
    >>> transpositions_from_mapping({1: 3, 3: 0, 0: 1, 2: 2})
    [(1, 3), (3, 0)]

    """
    seen = [False] * len(mapping)
    transpositions = []

    for v in mapping:
        if seen[v]:
            continue

        seen[v] = True
        cycle = [v]
        while len(cycle) == 1 or cycle[-1] != cycle[0]:
            v = mapping[v]
            seen[v] = True
            cycle.append(v)

        cycle.pop()

        if len(cycle) > 1:
            transpositions.extend(list(couples(cycle)))

    return transpositions


def couples(iterable: Iterable[Any]):
    """
    Iterate over all the pairs: (s0, s1), (s1, s2)...
    
    Parameters
    ----------
    iterable: Iterable[Any]
        Iterable from which to take the pairs.
        
    Examples
    --------
    >>> list(couples(range(5)))
    [(0, 1), (1, 2), (2, 3), (3, 4)]

    """
    iterable = iter(iterable)
    try:
        a = next(iterable)
        while True:
            b = next(iterable)
            yield (a, b)
            a = b
    except StopIteration:
        return


def symplectic_primitive(n: int) -> np.ndarray:
    """
    Builds the 2n × 2n symplectic bilinear form matrix.
    
    Parameter
    ---------
    n: int
        Half the size of the symplectic matrix.
        
    Returns
    -------
    np.ndarray
       The 2n × 2n symplectic bilinear form matrix.

    Examples
    --------
    >>> symplectic_primitive(2)
    array([[0, 0, 1, 0],
           [0, 0, 0, 1],
           [1, 0, 0, 0],
           [0, 1, 0, 0]], dtype=int64)

    """
    lambd = np.zeros((2 * n, 2 * n), dtype=np.int64)
    lambd[:n, n:] = np.eye(n)
    lambd[n:, :n] = np.eye(n)

    return lambd


def perm_circ(perm, n):
    raise NotImplementedError


def filter_circ(circ, coords):
    raise NotImplementedError


def lift(stab, qubits):
    raise NotImplementedError


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)