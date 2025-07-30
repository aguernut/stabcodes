"""
Define the base class for stabilizer codes and their derivatives (surface codes and color codes)
"""


from typing import Optional, Union, Sequence, Mapping
from math import prod
from copy import copy, deepcopy
from itertools import count, product, dropwhile, takewhile
from stabcodes.tools import rotate_left, rotate_to_place_first
from stabcodes.pauli import Stabilizer2D, Stabilizer, PauliOperator, X, Z, I
from stabcodes.stabgen import StabGen
from stabcodes.tools import symplectic_primitive
import numpy as np
from numpy.typing import NDArray


class StabilizerCode:
    """
    Base class hosting the logic common to all the stabilizer codes.

    Notes
    -----
    While this can be instantiated, it is better to overload this class to include
    a constructor to build a whole parametrized family of codes.

    Examples
    --------
    >>> SX = [Stabilizer([X(0), X(1), X(2), X(3)], 4)]
    >>> SZ = [Stabilizer([Z(0), Z(1), Z(2), Z(3)], 4)]
    >>> LX = [PauliOperator([X(0), X(1)], 4), PauliOperator([X(0), X(2)], 4)]
    >>> LZ = [PauliOperator([Z(0), Z(2)], 4), PauliOperator([Z(0), Z(1)], 4)]
    >>> code = StabilizerCode({"X": SX, "Z": SZ}, {"X": LX, "Z": LZ}, range(4))
    >>> code.check()  # Are the commutation relations correct ?

    """
    def __init__(self, stabilizers: Union[Sequence[Stabilizer], Mapping[str, Stabilizer]],
                 logical_operators: Mapping[str, PauliOperator],
                 qubits: range,
                 stab_relations: Optional[list[tuple[list[int], list[int]]]] = None,
                 no_check: bool = False):
        """Builds a stabilizer code from two collections of :class:`PauliOperator` s.
        
        Parameters
        ----------
        stabilizers: Union[Sequence[Stabilizer], Mapping[str, Stabilizer]
            Collection of stabilizer operators of the code.
        logical_operators:
            Mapping of logical Pauli operators of the code. X and Z logical operators
            must be specified and anticommute with their counterpart.
        qubits: range
            Range of qubit indices composing this code.
        stab_relations: list[tuple[list[int], list[int]]], optional
            Collections of stabilizer indices whose multiplication lead to the same
            value.
        no_check: bool
            Whether to check the consistency of the instantiated object. Faster with True,
            but you won't be warned if something is wrong with your stabilizer code.

        Examples
        --------
        >>> SX = [Stabilizer([X(0), X(1)], 3), Stabilizer([X(0), X(2)], 3), Stabilizer([X(1), X(2)], 3)]
        >>> SZ = []
        >>> LX = [PauliOperator([X(0)], 3)]
        >>> LZ = [PauliOperator([Z(0), Z(1), Z(2)], 3)]
        >>> code = StabilizerCode({"X": SX, "Z": SZ}, {"X": LX, "Z": LZ}, range(3), [([0, 1], [2])])
        
        """

        self._stabilizers = StabGen(stabilizers)
        self._qubits = qubits
        self._logical_operators = StabGen(logical_operators)
        self._stab_relations = [] if stab_relations is None else stab_relations

        if not no_check:
            self.check()

    def shift_qubits(self, n: int, extend_to=None):
        """Shifts all the qubit indices by an integer value.

        Parameters
        ----------
        n: int
            Integer value by which the qubit indices must be offset.
        extend_to: int, optional
            Final maximum value for the qubit indices in the whole context (e.g. in other codes)

        """
        self._qubits = range(self._qubits.start + n, self._qubits.stop + n)
        if extend_to is None:
            extend_to = self._qubits.stop
        m = extend_to - self._qubits.stop

        self._stabilizers.extend(m)
        self._stabilizers.itranslate(n=n)
        self._logical_operators.extend(m)
        self._logical_operators.itranslate(n=n)

    def copy(self) -> "StabilizerCode":
        """
        Returns a copy of this code.

        """
        return type(self)(copy(self._stabilizers), copy(self._logical_operators),
                          copy(self._qubits), copy(self._stab_relations),
                          no_check=True)

    @property
    def num_stabilizers(self) -> int:
        """Number of stabilizers of the code."""
        return len(self._stabilizers)

    @property
    def num_qubits(self) -> int:
        """Number of physical qubits used by the code."""
        return len(self._qubits)

    @property
    def qubits(self) -> range:
        """Collection of physical qubit indices used by this code."""
        return range(self._qubits.start, self._qubits.stop)

    @property
    def num_logical_qubits(self) -> int:
        """Number of logical qubits of the code."""
        return self.num_qubits - self.num_stabilizers + len(self._stab_relations)

    @property
    def stabilizers(self) -> StabGen:
        """Collection of stabilizer operators used by this code."""
        return self._stabilizers

    @property
    def logical_operators(self) -> StabGen:
        """Collection of logical Pauli operators of this code."""
        return self._logical_operators

    def __getitem__(self, index):
        return self._stabilizers.__getitem__(index)

    def __setitem__(self, index, value):
        return self._stabilizers.__setitem__(index, value)

    def check(self):
        """
        Checks the soundness of this stabilizer code.

        Raises
        ------
        ValueError
            Something inconsistent appears in the definition of the stabilizer code.
        """
        if {"X", "Z"} != set(self.logical_operators.keys()):
            raise ValueError("Provided logical operators must be split into \"X\" and \"Z\" types.")
        for s in self._stabilizers:
            if s.nb_qubits != self._qubits.stop:
                raise ValueError(f"Stabilizer {repr(s)} (index {self.stabilizers.index(s)}) is applied on different qubits ({self._qubits}).")

        g = np.nonzero(self.stabilizers.gram_product(self.stabilizers))
        try:
            i, j = next(iter(zip(*g)))
            raise ValueError(f"Stabilizers {repr(self.stabilizers[int(i)])} and {repr(self.stabilizers[int(j)])} do not commute.")
        except StopIteration:
            pass

        g = np.nonzero(self.stabilizers.gram_product(self.logical_operators))
        try:
            i, j = next(iter(zip(*g)))
            raise ValueError(f"Stabilizers {repr(self.stabilizers[int(i)])} and {repr(self.logical_operators[int(j)])} do not commute.")
        except StopIteration:
            pass

        g = np.nonzero(self.logical_operators.gram_product(self.logical_operators) - symplectic_primitive(self.num_logical_qubits))
        try:
            i, j = next(iter(zip(*g)))
            raise ValueError(f"Logical operators {repr(self.logical_operators[int(i)])} and {repr(self.logical_operators[int(j)])} unexpectedly anticommuted.")
        except StopIteration:
            pass

        for relation in self._stab_relations:
            assert prod((self._stabilizers[i] for i in relation[0]),
                        start=PauliOperator(nb_qubits=len(self._qubits))) == \
                prod((self._stabilizers[i] for i in relation[1]),
                     start=PauliOperator(nb_qubits=len(self._qubits))), \
                "Stabilizer relations do not check out"

    def stab_to_matrix(self) -> NDArray[np.int64]:
        """
        Returns a symplectic view of the stabilizer of this code.

        """
        return self._stabilizers.to_matrix()

    def is_logical_operator(self, operator: PauliOperator) -> bool:
        """
        Returns whether the given Pauli operator is a valid logical operator of this code.

        Parameters
        ----------
        operator: PauliOperator
            Operator whose commutation with the stabilizers will be checked.

        """
        return all(operator.commute(s) for s in self._stabilizers)

    def iter_stabilizers(self):
        """
        Iterates over the stabilizers of this code.

        """
        return iter(self._stabilizers)

    def iter_logical_operators(self, kind=None):
        """
        Iterates of the canonical logical Pauli operators of this code.

        Parameters
        ----------
        kind: str
            Can be "X" or "Z". Filter the kind of logical operators over which to iterate.
        """
        if kind is None:
            return iter(self._logical_operators)
        else:
            return iter(self._logical_operators[kind])


class SurfaceCode(StabilizerCode):

    @classmethod
    def hex_code(self, d1: int, d2: int) -> "SurfaceCode":
        assert d1 > 1
        assert d2 > 1

        # Qubits that are horizontally placed in horizontal or vertical position
        horiz = [[] for _ in range(d2)]
        verts = [[] for _ in range(d2)]

        for i in range(d2):
            for j in range(d1):
                horiz[i].append(i * 2 * d1 + 2 * j)
                horiz[i].append(i * 2 * d1 + 2 * j + 1)
                verts[i].append(i * d1 + j + 2 * d1 * d2)

        # Stabilizers associated to faces
        stabx = [Stabilizer2D([X(horiz[i][(2 * j + i % 2 + 1) % (2 * d1)]),
                               X(horiz[i][(2 * j + i % 2) % (2 * d1)]),
                               X(verts[i][j]),
                               X(horiz[(i + 1) % d2][(2 * j + i % 2 + (i % 2 == 0 and i + 1 == d2)) % (2 * d1)]),
                               X(horiz[(i + 1) % d2][(2 * j + i % 2 + 1 + (i % 2 == 0 and i + 1 == d2)) % (2 * d1)]),
                               X(verts[i][(j + 1) % d1])],
                              3 * d1 * d2) for (i, j) in product(range(d2), range(d1))]

        # Stabilizers associated to vertices
        stabz = [Stabilizer2D([Z(horiz[i][(2 * j + i % 2) % (2 * d1)]),
                               Z(horiz[i][(2 * j + i % 2 + 1) % (2 * d1)]),
                               Z(verts[(i - 1) % d2][(j + i % 2) % d1])], 3 * d1 * d2) for (i, j) in product(range(d2), range(d1))]

        stabz += [Stabilizer2D([Z(horiz[i][(2 * j + i % 2) % (2 * d1)]),
                                Z(horiz[i][(2 * j + i % 2 - 1) % (2 * d1)]),
                                Z(verts[i % d2][(j) % d1])], 3 * d1 * d2) for (i, j) in product(range(d2), range(d1))]

        # Non trivial closed paths are logical operators
        # We keep track of one representative for each non-equivalent logical operators
        logical_operators = {"X": [PauliOperator([X(horiz[0][i]) for i in range(2 * d1)],
                                                 3 * d1 * d2),
                                   PauliOperator([X(horiz[i][0]) for i in range(d2)] + [X(verts[i][0]) for i in range(d2)],
                                                 3 * d1 * d2)],
                             "Z": [PauliOperator([Z(horiz[i][-1]) for i in range(d2)],
                                                 3 * d1 * d2),
                                   PauliOperator([Z(verts[0][i]) for i in range(d1)],
                                                 3 * d1 * d2)]}

        return SurfaceCode({"X": stabx, "Z": stabz},
                           logical_operators,
                           range(3 * d1 * d2),
                           stab_relations=[(list(range(len(stabx) - 1)), [len(stabx) - 1]),
                                           (list(range(len(stabx), len(stabx) + len(stabz) - 1)), [len(stabx) + len(stabz) - 1])])

    @classmethod
    def toric_code(cls, d1, d2):
        """
        Square lattice toric code
        """
        assert d1 > 1
        assert d2 > 1

        # Qubits that are vertical or horizontal in the drawing
        verts = [[] for _ in range(d2)]
        horiz = [[] for _ in range(d2)]

        for i in range(d2):
            for j in range(d1):
                verts[i].append(i * d1 + j)
                horiz[i].append(i * d1 + j + d1 * d2)

        # Stabilizers associated with faces
        stabx = [Stabilizer2D([X(verts[i // d1][i % d1]),
                               X(horiz[i // d1][i % d1]),
                               X(verts[(i // d1 + 1) % d2][i % d1]),
                               X(horiz[i // d1][(i + 1) % d1])],
                              2 * d1 * d2) for i in range(d1 * d2)]

        # Stabilizers associated with vertices
        stabz = [Stabilizer2D([Z(horiz[i // d1][i % d1]),
                               Z(verts[(i // d1 + 1) % d2][(i - 1) % d1]),
                               Z(horiz[(i // d1 + 1) % d2][i % d1]),
                               Z(verts[(i // d1 + 1) % d2][i % d1])],
                              2 * d1 * d2) for i in range(d1 * d2)]

        # Non trivial closed paths are logical operators
        # We keep track of one representative for each non-equivalent logical operators
        logical_operators = {"X": [PauliOperator([X(qb) for qb in verts[0]], 2 * d1 * d2),
                                   PauliOperator([X(qb) for qb in [horiz[i][0] for i in range(d2)]], 2 * d1 * d2)],
                             "Z": [PauliOperator([Z(qb) for qb in [verts[i][0] for i in range(d2)]], 2 * d1 * d2),
                                   PauliOperator([Z(qb) for qb in horiz[0]], 2 * d1 * d2)]}

        stabilizers = {"X": stabx, "Z": stabz}

        return cls(stabilizers, logical_operators, range(2 * d1 * d2),
                   stab_relations=[(list(range(len(stabx) - 1)), [len(stabx) - 1]),
                                   (list(range(len(stabx), len(stabx) + len(stabz) - 1)), [len(stabx) + len(stabz) - 1])])

    @classmethod
    def cylindrical_patch(cls, dx: int, dz: int, big_dx=None):
        """
        Cylindrical fraction of a toric code
        """
        code = cls.toric_code(dx, dz)
        if big_dx is None:
            big_dx = dx
        to_remove = []
        for stab in code.iter_stabilizers("Z"):
            if stab.support_as_set & set((dx * dz + i * dx for i in range(dz))):
                to_remove.append(stab)
        for stab in to_remove:
            code._stabilizers["Z"].remove(stab)

        for stab in code.iter_stabilizers():
            for i in (dx * dz + i * dx for i in range(dz)):
                stab[i] = I(i)
            foo = stab.support_as_set
            stab.order = [i for i in stab.order if i in foo]

        N = 2 * dx * dz
        newN = 2 * big_dx * dz
        mapping = dict((i, (i % (N // 2)) // dx * big_dx + (i % (N // 2)) % dx + (i // (N // 2)) * (newN // 2)) for i in code._qubits)

        code._qubits = range(newN)

        code._stabilizers.extend(newN - N)
        code._stabilizers.itranslate(mapping=mapping)

        s = set(mapping.values()) - set((big_dx * dz + i * big_dx for i in range(dz)))

        for j in range(newN):
            if j in s:
                continue
            code.stabilizers["Z"].append(Stabilizer2D([Z(j)], len(code._qubits)))

        code._logical_operators["X"] = code._logical_operators["X"][:1]
        code._logical_operators["Z"] = code._logical_operators["Z"][:1]

        code._logical_operators.extend(newN - N)
        code._logical_operators.itranslate(mapping=mapping)

        code._stab_relations = [(list(range(len(code.stabilizers["X"]) - 1)), [len(code.stabilizers["X"]) - 1])]

        return SurfaceCode(code._stabilizers, code._logical_operators, code._qubits, code._stab_relations)

    def iter_stabilizers(self, kind=None):
        if kind is None:
            return super().iter_stabilizers()
        else:
            return iter(self._stabilizers[kind])

    def dehn_twist(self, guide, auxiliary=None, to_avoid={}, force_cnot=[], check=True):
        """
        guide: A Z (non self-intersecting) logical operator of the code
        auxiliary: One qubit within the Xoperator parallel to the guide,
        indicating the side on which the twist is performed
        """
        if check:
            assert self.is_logical_operator(PauliOperator.from_support(guide, "Z", len(self._qubits)))
        if not guide:
            return [], [], None
        guide_as_set, ordered_guide, ordered_pairs, auxiliary = self._compute_ordered_pairs(guide, auxiliary, to_avoid)
        # CNOT operations required for the dehn twist
        CNOTs, to_move = self._compute_CNOTs(guide_as_set, ordered_pairs)

        if force_cnot:
            print("Overriding CNOTs")
            CNOTs = force_cnot

        for stabx in self.stabilizers["X"]:
            for cnot in CNOTs:
                stabx.apply_CNOT_with_caution(*cnot)

        # Orders in Stabilizers touched by CNOTs need to be manually updated
        # Only those whose support changed need update.
        faces_other_kind = set()

        for stabz in self.stabilizers["Z"]:
            to_add = False
            for cnot in CNOTs:
                if stabz.apply_CNOT_with_caution(*cnot):
                    to_add = True
            if to_add:
                faces_other_kind.add(stabz)

        # The only X stabilizers affected by the CNOTs are those of ordered_pairs
        for (stab, deb, end, news) in zip(ordered_pairs, ordered_guide, ordered_guide[1:] + [ordered_guide[0]], to_move[1:] + to_move[:1]):
            stab.order[stab.order.index(deb) + 1:stab.order.index(end)] = news

        # For the affected Z stabilizers, all is as if we pulled along the guide
        mapping_dict = dict(zip(ordered_guide, ordered_guide[-1:] + ordered_guide[:-1]))
        for stab in faces_other_kind:
            for (i, elem) in enumerate(stab.order):
                if elem in mapping_dict:
                    stab.order[i] = mapping_dict[elem]

        # Logical operators are also affected
        for log in self._logical_operators:
            for cnot in CNOTs:
                log.apply_CNOT(*cnot)

        return ordered_guide, CNOTs, auxiliary

    def _compute_ordered_pairs(self, guide, auxiliary, to_avoid={}):
        """
        Select the X stabilizers traversed by a the guide, starting
        with one containing auxiliary.
        """
        guide_as_set = set(guide)  # Speed up the next step

        # Select the X stabilizers traversed by the guide
        pairs = [stab for stab in self.stabilizers["X"]
                 if len(stab.support_as_set & guide_as_set) == 2]

        # Coherent orderings of the selected stabilizers and guide support
        ordered_pairs = []
        ordered_guide = []
        processed = set()  # Speed up the process

        if auxiliary is None:
            stab = pairs[0]
            while stab.order[0] not in guide_as_set or stab.order[1] in guide_as_set:
                rotate_left(stab.order)
            for qb in stab.order[1:]:
                if qb in to_avoid:
                    rotate_left(stab.order)
                    while stab.order[0] not in guide_as_set or stab.order[1] in guide_as_set:
                        rotate_left(stab.order)
                    for qb in stab.order[1:]:
                        if qb in guide_as_set:
                            auxiliary = stab.order[1]
                            break

                if qb in guide_as_set:
                    auxiliary = stab.order[1]
                    break

        if auxiliary is None:
            raise ValueError("Automated twist failed")

        # Used to ensure a traversal of a closed path
        first_target = guide[0]
        next_target = guide[1]
        for stab in pairs:
            # Select the first traversed stabilizer (containing the auxiliary)
            if auxiliary in stab.support:
                # Rotate the view of the traversed stabilizers
                # To place the next qubit of the guide in first place
                rotate_to_place_first(stab.order, auxiliary)
                while stab.order[0] not in guide:
                    stab.order[:0], stab.order[1:] = stab.order[-1:], stab.order[:-1]
                first_target = stab.order[0]

                # The next target is the other qubit also present in the guide
                next_target = next(dropwhile(lambda x: x not in guide,
                                             stab.order[1:]))
                ordered_guide.append(first_target)
                processed.add(stab)
                ordered_pairs.append(stab)
                break
        else:
            # No first traversed stabilizer exists, the auxiliary is wrong
            raise ValueError(f"Given guide {auxiliary} is not suitable")

        # Compute the traversal along the guide
        while next_target != first_target:
            for stab in pairs:
                if stab in processed:
                    continue

                if next_target in stab.support:
                    # Rotate the view of the traversed stabilizers
                    # To place the next qubit of the guide in first place
                    rotate_to_place_first(stab.order, next_target)
                    ordered_guide.append(next_target)

                    # The next target is the other qubit also present in the guide
                    next_target = next(dropwhile(lambda x: x not in guide, stab.order[1:]))
                    processed.add(stab)
                    ordered_pairs.append(stab)
                    break
            else:
                raise ValueError(f"Given guide {guide} is not suitable")

        return guide_as_set, ordered_guide, ordered_pairs, auxiliary

    def _compute_CNOTs(self, guide_as_set, ordered_pairs):
        """
        Compute the CNOTs required for the dehn twist
        """
        # Group of qubit that need to be moved one stabilizer further
        # along the ordered guide
        CNOTs = []
        to_move = []
        for stab in ordered_pairs:
            to_move.append([])
            # One CNOT controlled on the first qubit of the stabilizer that is
            # also in guide to all the qubit of the stabilizer until the second
            # qubit also in guide is counter-clockwisely met.
            for qb in takewhile(lambda x: x not in guide_as_set, stab.order[1:]):
                CNOTs.append((stab.order[0], qb))

                # Targeted qubits are to be moved
                to_move[-1].append(qb)
        return CNOTs, to_move


class ColorCode(StabilizerCode):

    @classmethod
    def hex_code(cls, M, N):
        raise NotImplementedError

    @classmethod
    def code488(cls, d1, d2):
        raise NotImplementedError

    def compute_unfolding_circuit(self, color):
        raise NotImplementedError

    def compute_unfolding_circuitH(self, color):
        raise NotImplementedError

    def compute_folding_circuit(self, color, perm={}):
        raise NotImplementedError

    def part_twist(self, guideZ, color, opeZA=None, opeZB=None, auxA=None, auxB=None):
        raise NotImplementedError

    def dehn_twist(self, guideZ, color="blue", steps=2):
        raise NotImplementedError

    def unfolding(self, color="blue"):
        raise NotImplementedError

    def folding(self, color="blue"):
        raise NotImplementedError

    # def logicals_as_array(self):
    #     raise NotImplementedError

    # def confront(self):
    #     raise NotImplementedError


def _orienting_stabilizer(colored_paulioperators, to_keep=None):
    raise NotImplementedError


def simplify(available, stabs):
    # TODO in StabGen ?
    raise NotImplementedError


def simplify_ops(available, operators):
    raise NotImplementedError


def build_digraph_from_stabs(stabs):
    # TODO in StabGen ? -> Requires 2D
    raise NotImplementedError


def horiz_hex_dt_protocol():
    raise NotImplementedError


def economic_DT():
    raise NotImplementedError


def economic_DT488():
    raise NotImplementedError


def square488_protocol():
    raise NotImplementedError


def horiz_hex_dt_protocol_with_perm():
    raise NotImplementedError


def vertical_hex_dt_protocol():
    raise NotImplementedError


def vertical_hex_dt_protocol_split():
    raise NotImplementedError


def rot_90_488():
    raise NotImplementedError


def H_trans():
    raise NotImplementedError


def S_trans():
    raise NotImplementedError


# def last_exp():
#      raise NotImplementedError


def test_dehn_twist1():
    c = SurfaceCode.toric_code(6, 4)
    initial_logical = deepcopy(c.logical_operators)
    c.dehn_twist([0, 6, 12, 18], 24)
    c.check()
    target = {'X': [Stabilizer2D([X(30), X(6), X(25), X(0)], 48),
                    Stabilizer2D([X(1), X(25), X(7), X(26)], 48),
                    Stabilizer2D([X(2), X(26), X(8), X(27)], 48),
                    Stabilizer2D([X(3), X(27), X(9), X(28)], 48),
                    Stabilizer2D([X(4), X(28), X(10), X(29)], 48),
                    Stabilizer2D([X(5), X(29), X(11), X(24)], 48),
                    Stabilizer2D([X(36), X(12), X(31), X(6)], 48),
                    Stabilizer2D([X(7), X(31), X(13), X(32)], 48),
                    Stabilizer2D([X(8), X(32), X(14), X(33)], 48),
                    Stabilizer2D([X(9), X(33), X(15), X(34)], 48),
                    Stabilizer2D([X(10), X(34), X(16), X(35)], 48),
                    Stabilizer2D([X(11), X(35), X(17), X(30)], 48),
                    Stabilizer2D([X(42), X(18), X(37), X(12)], 48),
                    Stabilizer2D([X(13), X(37), X(19), X(38)], 48),
                    Stabilizer2D([X(14), X(38), X(20), X(39)], 48),
                    Stabilizer2D([X(15), X(39), X(21), X(40)], 48),
                    Stabilizer2D([X(16), X(40), X(22), X(41)], 48),
                    Stabilizer2D([X(17), X(41), X(23), X(36)], 48),
                    Stabilizer2D([X(24), X(0), X(43), X(18)], 48),
                    Stabilizer2D([X(19), X(43), X(1), X(44)], 48),
                    Stabilizer2D([X(20), X(44), X(2), X(45)], 48),
                    Stabilizer2D([X(21), X(45), X(3), X(46)], 48),
                    Stabilizer2D([X(22), X(46), X(4), X(47)], 48),
                    Stabilizer2D([X(23), X(47), X(5), X(42)], 48)],
              'Z': [Stabilizer2D([Z(0), Z(24), Z(11), Z(30)], 48),
                    Stabilizer2D([Z(25), Z(6), Z(31), Z(7)], 48),
                    Stabilizer2D([Z(26), Z(7), Z(32), Z(8)], 48),
                    Stabilizer2D([Z(27), Z(8), Z(33), Z(9)], 48),
                    Stabilizer2D([Z(28), Z(9), Z(34), Z(10)], 48),
                    Stabilizer2D([Z(29), Z(10), Z(35), Z(11)], 48),
                    Stabilizer2D([Z(6), Z(30), Z(17), Z(36)], 48),
                    Stabilizer2D([Z(31), Z(12), Z(37), Z(13)], 48),
                    Stabilizer2D([Z(32), Z(13), Z(38), Z(14)], 48),
                    Stabilizer2D([Z(33), Z(14), Z(39), Z(15)], 48),
                    Stabilizer2D([Z(34), Z(15), Z(40), Z(16)], 48),
                    Stabilizer2D([Z(35), Z(16), Z(41), Z(17)], 48),
                    Stabilizer2D([Z(12), Z(36), Z(23), Z(42)], 48),
                    Stabilizer2D([Z(37), Z(18), Z(43), Z(19)], 48),
                    Stabilizer2D([Z(38), Z(19), Z(44), Z(20)], 48),
                    Stabilizer2D([Z(39), Z(20), Z(45), Z(21)], 48),
                    Stabilizer2D([Z(40), Z(21), Z(46), Z(22)], 48),
                    Stabilizer2D([Z(41), Z(22), Z(47), Z(23)], 48),
                    Stabilizer2D([Z(18), Z(42), Z(5), Z(24)], 48),
                    Stabilizer2D([Z(43), Z(0), Z(25), Z(1)], 48),
                    Stabilizer2D([Z(44), Z(1), Z(26), Z(2)], 48),
                    Stabilizer2D([Z(45), Z(2), Z(27), Z(3)], 48),
                    Stabilizer2D([Z(46), Z(3), Z(28), Z(4)], 48),
                    Stabilizer2D([Z(47), Z(4), Z(29), Z(5)], 48)]}

    target_logical = {'X': [PauliOperator([X(0), X(1), X(2), X(3), X(4), X(5), X(24)], 48),
                            PauliOperator([X(24), X(30), X(36), X(42)], 48)],
                      'Z': [PauliOperator([Z(0), Z(6), Z(12), Z(18)], 48),
                            PauliOperator([Z(0), Z(24), Z(25), Z(26), Z(27), Z(28), Z(29)], 48)]}

    for sx in c.stabilizers["X"]:
        if sx in target["X"]:
            break
    else:
        raise ValueError("Unexpected stabilizers found")

    for sz in c.stabilizers["Z"]:
        if sz in target["Z"]:
            break
    else:
        raise ValueError("Unexpected stabilizers found")

    for log in c.logical_operators["X"]:
        if log in target_logical["X"]:
            break
    else:
        raise ValueError("Unexpected stabilizers found")

    for log in c.logical_operators["Z"]:
        if log in target_logical["Z"]:
            break
    else:
        raise ValueError("Unexpected stabilizers found")

    c.dehn_twist([0, 6, 12, 18], 24)
    c.check()
    c.dehn_twist([0, 6, 12, 18], 24)
    c.check()
    c.dehn_twist([0, 6, 12, 18], 24)
    c.check()

    assert initial_logical["X"][0] * initial_logical["X"][1] == c.logical_operators["X"][0]
    assert initial_logical["Z"][0] * initial_logical["Z"][1] == c.logical_operators["Z"][1]

    c = SurfaceCode.toric_code(6, 4)
    c.dehn_twist([24, 25, 26, 27, 28, 29], 0)
    c.check()
    c.dehn_twist([24, 25, 26, 27, 28, 29], 0)
    c.check()
    c.dehn_twist([24, 25, 26, 27, 28, 29], 0)
    c.check()
    c.dehn_twist([24, 25, 26, 27, 28, 29], 0)
    c.check()
    c.dehn_twist([24, 25, 26, 27, 28, 29], 0)
    c.check()
    c.dehn_twist([24, 25, 26, 27, 28, 29], 0)
    c.check()

    assert initial_logical["X"][0] * initial_logical["X"][1] == c.logical_operators["X"][1]
    assert initial_logical["Z"][0] * initial_logical["Z"][1] == c.logical_operators["Z"][0]


def test_dehn_twist2():
    c = SurfaceCode.hex_code(4, 2)
    initial_logical = deepcopy(c.logical_operators)
    print(c.logical_operators)

    c.dehn_twist(list(range(16, 20)), 0)
    c.check()
    c.dehn_twist(list(range(16, 20)), 0)
    c.check()
    c.dehn_twist(list(range(16, 20)), 0)
    c.check()
    c.dehn_twist(list(range(16, 20)), 0)
    c.check()
    assert initial_logical["X"][0] * initial_logical["X"][1] == c.logical_operators["X"][1]
    assert initial_logical["Z"][0] * initial_logical["Z"][1] == c.logical_operators["Z"][0]

    c = SurfaceCode.hex_code(4, 2)
    c.dehn_twist([7, 15], 0)
    c.check()
    c.dehn_twist([7, 15], 0)

    assert initial_logical["X"][0] * initial_logical["X"][1] == c.logical_operators["X"][0]
    assert initial_logical["Z"][0] * initial_logical["Z"][1] == c.logical_operators["Z"][1]
    c.check()


def test_dehn_twist3():
    c = SurfaceCode.toric_code(6, 4)
    initial_logical = deepcopy(c.logical_operators)
    print(c.logical_operators)

    c.dehn_twist([3, 28, 29, 11, 17, 23, 47, 46], 35)
    c.check()
    c.dehn_twist([3, 28, 29, 11, 17, 23, 47, 46], 35)
    c.check()
    c.dehn_twist([3, 28, 29, 11, 17, 23, 47, 46], 35)
    c.check()
    c.dehn_twist([3, 28, 29, 11, 17, 23, 47, 46], 35)
    c.check()
    c.dehn_twist([3, 28, 29, 11, 17, 23, 47, 46], 35)
    c.check()
    c.dehn_twist([3, 28, 29, 11, 17, 23, 47, 46], 35)
    c.check()
    c.dehn_twist([3, 28, 29, 11, 17, 23, 47, 46], 35)
    c.check()
    c.dehn_twist([3, 28, 29, 11, 17, 23, 47, 46], 35)
    c.check()

    assert initial_logical["Z"][1] * PauliOperator([Z(3), Z(28), Z(29), Z(11), Z(17), Z(23), Z(47), Z(46)], 48) == c.logical_operators["Z"][1]
    assert initial_logical["X"][0] * PauliOperator([X(27), X(9), X(10), X(35), X(41), X(22), X(21), X(45)], 48) == c.logical_operators["X"][0]


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)

    stabs = [PauliOperator.from_str("+IXZZXI"),
             PauliOperator.from_str("+XZZXII"),
             PauliOperator.from_str("+ZZXIXI"),
             PauliOperator.from_str("+ZXIXZI"),
             PauliOperator.from_str("+XIXZZI")]

    log = {"X": [PauliOperator.from_str("+XXXXXI"), PauliOperator.from_str("+IIIIIX")],
           "Z": [PauliOperator.from_str("+ZZZZZI"), PauliOperator.from_str("+IIIIIZ")]}

    code = StabilizerCode(stabs, log, range(6), [([0, 1, 2, 3], [4])])
    code.is_logical_operator(PauliOperator.from_support([1, 2, 3, 4, 5], "Y", 6))

    a = SurfaceCode.hex_code(3, 3)
    b = SurfaceCode.toric_code(3, 3)
    c = SurfaceCode.cylindrical_patch(3, 3)
    d = SurfaceCode.cylindrical_patch(2, 3, 3)

    test_dehn_twist1()
    test_dehn_twist2()
    test_dehn_twist3()
