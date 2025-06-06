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


def reverse_dict(dic):
    """
    Unique hashable values of dic are returned as key in the new dictionnary,
    with new values being the former keys

    Fails if a value in not hashable or if a value is the same for different keys

    See tools.invert_dict if a list of keys having the same value is wanted
    """
    new_dic = dict()
    for (key, value) in dic.items():
        if isinstance(value, list):
            for v in value:
                if v not in new_dic:
                    try:
                        new_dic[v] = key
                    except Exception:
                        raise TypeError(f"Unhashable type : {type(value)}")
        else:
            if value not in new_dic:
                try:
                    new_dic[value] = key
                except Exception:
                    raise TypeError(f"Unhashable type : {type(value)}")
            else:
                raise ValueError(f"Value {repr(value)} is present multiple times")
    return new_dic


def mapping(code0, code1, modes, qb, rev=False):
    seen = set()
    seen2 = set()
    traited = set()
    while len(seen) < len(code0.stabilizers[modes[0]]):
        for stab in code0.iter_stabilizers(modes[0]):
            if stab in seen:
                continue
            if len(stab.order) == 1:
                seen.add(stab)
                continue

            if inter := set(stab.order) & (qb.keys() - traited):
                if len(inter) == 4:
                    seen.add(stab)
                    continue

                trans_inter = set(qb[i] for i in inter)
                for stab2 in code1.iter_stabilizers(modes[1]):
                    if stab2 in seen2:
                        continue

                    if set(stab2.order) & trans_inter:
                        a = next(iter(inter))
                        try:
                            rotate_to_place_first(stab.order, a)
                            rotate_to_place_first(stab2.order, qb[a])
                        except ValueError:
                            print(stab.order, stab2.order)
                            raise ValueError

                        for (q0, q1) in zip(stab.order, stab2.order):
                            qb[q0] = q1

                        seen.add(stab)
                        seen2.add(stab2)
                        break

                else:
                    raise NotImplementedError
                traited.add(a)
                break
        else:
            if len(seen) < len(code0.stabilizers[modes[0]]):
                raise NotImplementedError

    return qb


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
