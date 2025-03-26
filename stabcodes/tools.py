from stabcodes.pauli import PauliOperator, Stabilizer2D
import numpy as np


def pretty_print_operators(ops):
    """FIXME
    Pretty-print an iterable of PauliOperators

    Parameters;
    ops: Iterable of PauliOperators
    """
    print()
    for op in ops:
        print(PauliOperator.__repr__(op))
    print()


def pretty_print_stabilizers(stabs):
    """FIXME
    Pretty-print an iterable of Stabilizers

    Parameters;
    stabs: Iterable of Stabilizers
    """
    print()
    for op in stabs:
        print(Stabilizer2D.__repr__(op))
    print()


def largest(lists):
    """FIXME
    Return the largest element of the given list of elements with len method.

    Parameters:
    lists: List of element implementing the __len__ method

    Returns:
    The element of lists with biggest length
    """
    return max(lists, key=len)


def cycle_to_operator(cycle):
    """FIXME
    Convert a path of edges to a list of vertices.

    Parameters:
    cycle: A list of edges that form a cycle
    """
    return [e[0] for e in cycle]


def split_operator(operator, support):
    """FIXME
    Splits a PauliOperator into two parts, restricting it on a support
    or complementing it.

    Parameters:
    operator: PauliOperator to split
    support: A list of qubits to split along

    Returns:
    2 PauliOperators whose product equals the given one
    """
    PA = []
    PB = []

    for p in operator.paulis:
        if p.qubit in support:
            PA.append(p)
        else:
            PB.append(p)

    return PauliOperator(PA, operator.nb_qubits), PauliOperator(PB, operator.nb_qubits)


def rotate_left(liste):
    liste[:] = liste[1:] + [liste[0]]


def rotate_to_place_first(liste, element):
    indice = liste.index(element)
    liste[:indice], liste[len(liste):] = [], liste[:indice]


def symplectic_primitive(n):
    lambd = np.zeros((2 * n, 2 * n))
    lambd[:n, n:] = np.eye(n)
    lambd[n:, :n] = np.eye(n)

    return lambd


def is_intersecting(set1, to_avoid):
    """FIXME"""
    return len(set1 & to_avoid) > 1


def perm_circ(perm, n):
    raise NotImplementedError


def filter_circ(circ, coords):
    raise NotImplementedError


def lift(stab, qubits):
    raise NotImplementedError
