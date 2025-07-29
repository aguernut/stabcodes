"""
Implements Pauli matrix and Pauli operator logic.
"""

from typing import Optional, Iterator, Union
import numpy as np
from numpy.typing import NDArray
from itertools import chain
import qiskit
import warnings


_PAULIS: set[str] = {"I", "X", "Y", "Z"}

_CNOT_TABLEAU: dict[str, str] = {"II": "II",
                                 "XI": "XX",
                                 "YI": "YX",
                                 "ZI": "ZI",
                                 "IX": "IX",
                                 "XX": "XI",
                                 "YX": "YI",
                                 "ZX": "ZX",
                                 "IZ": "ZZ",
                                 "XZ": "YY",
                                 "YZ": "XY",
                                 "ZZ": "IZ",
                                 "IY": "ZY",
                                 "XY": "YZ",
                                 "YY": "XZ",
                                 "ZY": "IY"}

_CZ_TABLEAU: dict[str, str] = {"II": "II",
                               "XI": "XZ",
                               "YI": "YZ",
                               "ZI": "ZI",
                               "IX": "ZX",
                               "XX": "YY",
                               "YX": "XY",
                               "ZX": "IX",
                               "IZ": "IZ",
                               "XZ": "XI",
                               "YZ": "YI",
                               "ZZ": "ZZ",
                               "IY": "ZY",
                               "XY": "YX",
                               "YY": "XX",
                               "ZY": "IY"}

_SWAP_TABLEAU: dict[str, str] = {"II": "II",
                                 "XI": "IX",
                                 "YI": "IY",
                                 "ZI": "IZ",
                                 "IX": "XI",
                                 "XX": "XX",
                                 "YX": "XY",
                                 "ZX": "XZ",
                                 "IZ": "ZI",
                                 "XZ": "ZX",
                                 "YZ": "ZY",
                                 "ZZ": "ZZ",
                                 "IY": "YI",
                                 "XY": "YX",
                                 "YY": "YY",
                                 "ZY": "YZ"}

_S_TABLEAU: dict[str, str] = {"I": "I",
                              "X": "Y",
                              "Y": "X",
                              "Z": "Z"}

_SDG_TABLEAU: dict[str, str] = {"I": "I",
                                "X": "Y",
                                "Y": "X",
                                "Z": "Z"}

_H_TABLEAU: dict[str, str] = {"I": "I",
                              "X": "Z",
                              "Y": "Y",
                              "Z": "X"}

_COST: dict[str, float] = {"II": 0, "IX": 1, "IY": 1.5, "IZ": 1,
                           "XI": 0, "XX": -1, "XY": 0, "XZ": 0.5,
                           "YI": 0, "YX": -0.5, "YY": -1.5, "YZ": -0.5,
                           "ZI": 0, "ZX": 0.5, "ZY": 0, "ZZ": -1}


class Pauli:
    """Class implementing the logic of Pauli matrices.

    Its main purpose is to serve as building blocks for
    :class:`~stabcodes.Pauli.PauliOperator` and
    :class:`Stabilizer`.

    Attributes
    ----------

    kind : str
        String describe the kind of Pauli matrix. Must be either
        "I", "X", "Y" or "Z".
    qubit : int
        Qubit on which the Pauli matrix is applied. Relevant in
        broader contexts such as Pauli operators. Must be positive.

    Notes
    -----
    These matrices form a multiplicative group. Thus, elements of this
    class can be multiplied if they act on the same qubit.

    Examples
    --------
    >>> Pauli("X", 0) * Pauli("Z", 0)
    Y(0)
    >>> p, q = Pauli("X"), Pauli("Y")
    >>> p *= q
    >>> p
    Z(0)
    >>> Pauli("I", 0) * Pauli("X", 1)
    Traceback (most recent call last):
        ...
    ValueError: Paulis I(0) and X(1) must be applied on same qubit.
    """

    def __init__(self, kind: str = "I", qubit: int = 0):
        """Default constructor of the Pauli class

        Parameters
        ----------
        kind : str
            String describe the kind of Pauli matrix. Must be either
            "I", "X", "Y" or "Z".
        qubit : int
            Qubit on which the Pauli matrix is applied. Relevant in
            broader contexts such as Pauli operators. Must be positive.

        Raises
        ------
        ValueError
            Raised when an improper kind is provided, or a negative
            qubit.

        Examples
        --------
        >>> Pauli("X", 0)
        X(0)
        >>> Pauli("S")
        Traceback (most recent call last):
            ...
        ValueError: Only {"I", "X", "Y", "Z"} kinds are supported, not S.
        >>> Pauli("X", -1)
        Traceback (most recent call last):
            ...
        ValueError: Given qubit must be a positive integer (received -1).

        """

        if kind not in _PAULIS:
            raise ValueError(f'Only {{"I", "X", "Y", "Z"}} kinds are supported, not {kind}.')
        self.__kind = kind
        if qubit < 0:
            raise ValueError(f"Given qubit must be a positive integer (received {qubit}).")
        self.__qubit = qubit

    @property
    def kind(self) -> str:
        """Kind of Pauli matrix represented."""
        return self.__kind

    @kind.setter
    def kind(self, value: str):
        if value not in _PAULIS:
            raise ValueError(f'Only {{"I", "X", "Y", "Z"}} kinds are supported, not {value}.')
        self.__kind = value

    @property
    def qubit(self) -> int:
        """Qubit acted upon by the Pauli matrix"""
        return self.__qubit

    @qubit.setter
    def qubit(self, value: int):
        if value < 0:
            raise ValueError(f"Given qubit must be a positive integer (received {value}).")
        self.__qubit = value

    def __repr__(self) -> str:
        return f"{self.kind}({self.qubit})"

    def __copy__(self) -> "Pauli":
        return Pauli(self.kind, self.qubit)

    def copy(self) -> "Pauli":
        """Returns a copy of this Pauli matrix.

        Examples
        --------
        >>> a = Pauli("X", 0)
        >>> b = a.copy()
        >>> a = a * b
        >>> a.kind
        'I'
        >>> b.kind
        'X'

        """
        return self.__copy__()

    def commute(self, other: "Pauli") -> bool:
        """Checks if the provided Pauli matrix commutes with this Pauli matrix.

        Matrices acting on different qubits always
        commutes. Otherwise, matrices commutes if they are of the same
        kind or either is an identity matrix.

        Parameters
        ----------
        other : :class:`~stabcodes.Pauli.Pauli`
            Pauli matrix to check commutation against.

        Raises
        ------
        :obj:`TypeError`
            Raised if commutation is checked against a
            non-:class:`~stabcodes.Pauli.Pauli`.

        Examples
        --------
        >>> Pauli("X").commute(Pauli("I"))
        True
        >>> Pauli("X").commute(Pauli("Z"))
        False
        >>> Pauli("Y", 1).commute(Pauli("X", 0))
        True

        """
        if not isinstance(other, Pauli):
            raise TypeError(f"Commutation can only be tested w.r.t. other Paulis (received {type(other).__name__}).")

        if (self.qubit != other.qubit) or \
           (self.kind == "I") or \
           (other.kind == "I"):
            return True
        return self.kind == other.kind

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Pauli):
            return self.kind == other.kind and self.qubit == other.qubit
        return False

    def __mul__(self, other: "Pauli") -> "Pauli":
        if not isinstance(other, Pauli):
            raise TypeError(
                f"Only Paulis can be multiplied to Paulis, not {type(other).__name__}.")

        if self.qubit != other.qubit:
            raise ValueError(
                f"Paulis {self} and {other} must be applied on same qubit.")

        if self.kind == "I":
            return Pauli(other.kind, self.qubit)

        if other.kind == "I":
            return Pauli(self.kind, self.qubit)

        non_id = {"X", "Y", "Z"}

        non_id -= {self.kind}
        non_id -= {other.kind}

        if len(non_id) == 2:
            return Pauli("I", self.qubit)

        return Pauli(next(iter(non_id)), self.qubit)

    def __imul__(self, other: "Pauli"):
        foo = self * other
        self.kind = foo.kind
        return self

    def __hash__(self) -> int:
        return hash(self.kind + "(" + str(self.qubit) + ")")


class PauliOperator:
    """
    Class implementing the logic of Pauli operators.

    Attributes
    ----------

    paulis : list[Pauli]
        List of :class:`Pauli` matrices composing
        the Pauli operator.
    nb_qubits : :obj:`int`
        Number of qubits upon which the Pauli operator acts.
        Must be positive.
    support : list[int]
        List of qubits upon which the Pauli operator acts non-trivially.
    support_as_set : set[int]
        Set of qubits upon which the Pauli operator acts non-trivially.
    last_measure_time : list[Optional[int]]
        Time of last measurement of this operator in a broader context of circuit experiment.

    Notes
    -----
    Pauli operators on a given number of qubits form a multiplicative group.
    Thus, elements of this class can be multiplied if they act on the same
    number of qubits.

    Examples
    --------
    >>> o1 = PauliOperator([X(0), Y(1)], 3)
    >>> o2 = PauliOperator([X(1), Z(2)], 3)
    >>> o1 * o2
    PauliOperator([X(0), Z(1), Z(2)], 3)
    >>> o1 *= o1
    >>> o1
    PauliOperator([], 3)
    >>> o1.extend(1)
    >>> o2 * o1
    Traceback (most recent call last):
        ...
    ValueError: Cannot multiply two PauliOperators acting on different number of qubits (3 != 4).
    """

    def __init__(self, paulis: Optional[list[Pauli]] = None, nb_qubits: int = 1, last_measure_time: Optional[list[Optional[int]]] = None):
        """Default constructor of the PauliOperator class.

        Parameters
        ----------
        paulis : list[Pauli]
            List of :class:`~stabcodes.Pauli.Pauli` composing the Pauli
            operator. Set to `nb_qubits` identity matrices if None.
        nb_qubits : :obj:`int`
            Number of qubits upon which the Pauli operator will act.
        last_measure_time : list[Optional[int]], optional
            List of time indices where this operator was measured.

        Raises
        ------
        ValueError
            Raised if a provided Pauli acts on a qubit not covered by
            this operator (e.g. if Pauli.qubit >= `nb_qubits`).
        TypeError
            Raised if a non-:class:`~stabcodes.Pauli.Pauli` is provided.

        Examples
        --------
        >>> PauliOperator(nb_qubits=2)
        PauliOperator([], 2)
        >>> PauliOperator([X(0), Y(3), Z(1), Z(0)], 4)
        PauliOperator([Y(0), Z(1), Y(3)], 4)
        >>> PauliOperator([X(1)], 1)
        Traceback (most recent call last):
            ...
        ValueError: Given Pauli (X(1)) is applied on an out-of-range qubit (range: (0, 0)).
        """
        self.__nb_qubits = nb_qubits

        if paulis is None:
            paulis = []

        if last_measure_time is None:
            last_measure_time = [None]

        self.__paulis = [Pauli("I", i) for i in range(nb_qubits)]

        for pauli in paulis:
            if not isinstance(pauli, Pauli):
                raise TypeError("PauliOperator must be given Paulis, not {type(pauli)}.")
            if pauli.qubit < 0 or pauli.qubit >= nb_qubits:
                raise ValueError(f"Given Pauli ({pauli}) is applied on an out-of-range qubit (range: (0, {nb_qubits-1})).")

            self.__paulis[pauli.qubit] *= pauli

        self.__support = None
        self.__support_as_set = None
        self.__last_measure_time = last_measure_time

    @classmethod
    def from_support(cls, support: list[int], kind: str = "Z", nb_qubits: int = 1):
        """
        Creates a :class:`PauliOperator` with the same Pauli on given support.

        Parameters
        ----------

        support: list[int]
            The support acted upon not trivially.
        kind: str
            The kind of Pauli wanted. Can take only the value "X" | "Y" | "Z".
        nb_qubits: int
            Number of qubits in context

        Returns
        -------
        PauliOperator
            Pauli operator acting as the given kind on the given support

        Examples
        --------
        >>> PauliOperator.from_support([1,3,5], "Z", 7)
        PauliOperator([Z(1), Z(3), Z(5)], 7)
        >>> PauliOperator.from_support([1, 2], "X", 7)
        PauliOperator([X(1), X(2)], 7)
        """
        func = X if kind == "X" else Z if kind == "Z" else Y if kind == "Y" else None
        return cls([func(qb) for qb in support], nb_qubits)

    @property
    def paulis(self) -> list[Pauli]:
        """Pauli matrices composing the Pauli operator."""
        return [p for p in self.__paulis]

    @property
    def nb_qubits(self) -> int:
        """Number of qubits acted upon by the Pauli operator."""
        return self.__nb_qubits

    @property
    def last_measure_time(self) -> Optional[int]:
        """Last time this operator was measured, if any.

        Must be updated through a call to the :meth:`~stabcodes.Pauli.PauliOperator.measure` method or the :meth:`~stabcodes.Pauli.PauliOperator.reset` method.

        """
        return self.__last_measure_time

    def __copy__(self) -> "PauliOperator":
        return PauliOperator([p.copy() for p in self.paulis], self.nb_qubits, self.last_measure_time)

    def copy(self) -> "PauliOperator":
        """Returns a copy of this Pauli operator.

        Examples
        --------
        >>> a = PauliOperator([X(0), Z(1)], 2)
        >>> b = a.copy()
        >>> a *= b
        >>> a
        PauliOperator([], 2)
        >>> b
        PauliOperator([X(0), Z(1)], 2)

        """
        return self.__copy__()

    def __repr__(self) -> str:
        return f"PauliOperator({[pauli for pauli in self.paulis if pauli.kind != 'I']}, {self.nb_qubits})"

    def __str__(self) -> str:
        return "+" + "".join(pauli.kind for pauli in self.paulis[::-1])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PauliOperator):
            return False
        return str(self) == str(other)

    def __bool__(self) -> bool:
        return bool(self.support)

    def __hash__(self) -> int:
        return hash(str(self))

    def apply_Clifford(self, tableau: dict[str, str], *qubits: int):
        """Applies a Clifford Tableau in-place to the Pauli operator.

        The Tableau of a :math:`n`-qubit Clifford gate is a dictionary that
        indicates how the :math:`4^{n}` Pauli operators over :math:`n` qubits
        change when conjugated by it.
        For example the Tableau of the CNOT gate is:

        .. code-block:: python

            _CNOT_TABLEAU = {"II": "II", "XI": "XX", "YI": "YX", "ZI": "ZI",
                             "IX": "IX", "XX": "XI", "YX": "YI", "ZX": "ZX",
                             "IZ": "ZZ", "XZ": "YY", "YZ": "XY", "ZZ": "IZ",
                             "IY": "ZY", "XY": "YZ", "YY": "XZ", "ZY": "IY"}

        The Tableau of usual Clifford gates can be
        imported as :code:`_CNOT_TABLEAU`, :code:`_CZ_TABLEAU`, :code:`_SWAP_TABLEAU`,
        :code:`_S_TABLEAU`, :code:`_SDG_TABLEAU`, :code:`_H_TABLEAU`.

        Parameters
        ----------
        tableau : dict[str, str]
            Tableau corresponding to the Clifford to apply
        *qubits : :obj:`int`
            Ordered qubit indices on which the Clifford is
            applied.

        Raises
        ------
        ValueError
            The provided Tableau does not match the number of
            specified qubits.

        Examples
        --------
        >>> P = PauliOperator([Z(0), Y(1)], 2)
        >>> Q = P.copy()
        >>> P.apply_Clifford(_CNOT_TABLEAU, 0, 1)
        >>> P.apply_Clifford(_CNOT_TABLEAU, 1, 0)
        >>> P.apply_Clifford(_CNOT_TABLEAU, 0, 1)
        >>> P
        PauliOperator([Y(0), Z(1)], 2)
        >>> P.apply_Clifford(_SWAP_TABLEAU, 0, 1)
        >>> P == Q
        True
        """
        if len(next(iter(tableau))) != len(qubits):
            raise ValueError(
                f"The given tableau does not affect {len(qubits)} at once.")

        now = "".join(self.paulis[qb].kind for qb in qubits)
        future = tableau[now]

        for (kind, qb) in zip(future, qubits):
            self.__paulis[qb] = Pauli(kind, qb)
        self.__support = None
        self.__support_as_set = None

    def apply_circuit(self, circuit: qiskit.QuantumCircuit):
        """Applies a :obj:`qiskit.circuit.QuantumCircuit` to the Pauli operator (by conjugation) :mod:`string`.

        This function iterates over the gates of the circuit, applying
        each of them in turn.

        Parameters
        ----------
        circuit : :class:`~qiskit.circuit.QuantumCircuit`
            Quantum circuit to apply.

        Raises
        ------
        ValueError
            A gate of the circuit is not among the defined Clifford
            gate.

        Examples
        --------
        >>> P = PauliOperator([Z(0)], 2)
        >>> circuit = qiskit.QuantumCircuit(2)
        >>> circuit.h(0)
        <...
        >>> circuit.cx(0, 1)
        <...
        >>> P.apply_circuit(circuit)
        >>> P == PauliOperator([X(0), X(1)], 2)
        True
        >>> circuit.t(0)
        <...
        >>> P.apply_circuit(circuit)
        Traceback (most recent call last):
            ...
        ValueError: Unknown Clifford gate: t.

        """
        backup = self.copy()
        try:
            for gate in circuit:
                name = gate.operation.name
                qubits = gate.qubits

                if name == "cx":
                    self.apply_CNOT(qubits[0]._index,
                                    qubits[1]._index)
                elif name == "cz":
                    self.apply_CZ(qubits[0]._index,
                                  qubits[1]._index)
                elif name == "s":
                    self.apply_S(qubits[0]._index)
                elif name == "sdg":
                    self.apply_SDG(qubits[0]._index)
                elif name == "h":
                    self.apply_H(qubits[0]._index)
                elif name == "swap":
                    self.apply_SWAP(qubits[0]._index,
                                    qubits[1]._index)
                elif name == "id":
                    pass
                elif name == "x":
                    pass
                elif name == "y":
                    pass
                elif name == "z":
                    pass
                else:
                    raise ValueError(f"Unknown Clifford gate: {name}.")

        except Exception as e:
            self.__paulis = backup.__paulis
            raise e

    def apply_CNOT(self, control: int, target: int):
        """Applies a CNOT gate between the `control` qubit and the `target` qubit.

        Parameters
        ----------
        control : :obj:`int`
            Control qubit of the CNOT.
        target : :obj:`int`
            Target qubit of the CNOT.

        Examples
        --------
        >>> P = PauliOperator([Y(0), Z(1), X(2)], 3)
        >>> P.apply_CNOT(2, 0)
        >>> P
        PauliOperator([Z(0), Z(1), Y(2)], 3)

        """
        self.apply_Clifford(_CNOT_TABLEAU, control, target)
        self.__support = None
        self.__support_as_set = None

    def apply_CZ(self, qubit1: int, qubit2: int):
        """Applies a CZ gate between the two qubits.

        Parameters
        ----------
        qubit1 : :obj:`int`
            First qubit on which to apply the CZ gate.
        qubit2 : :obj:`int`
            Second qubit on which to apply the CZ gate.

        Examples
        --------
        >>> P = PauliOperator([Y(0), Z(1), X(2)], 3)
        >>> P.apply_CZ(2, 0)
        >>> P
        PauliOperator([X(0), Z(1), Y(2)], 3)

        """
        self.apply_Clifford(_CZ_TABLEAU, qubit1, qubit2)
        self.__support = None
        self.__support_as_set = None

    def apply_H(self, qubit: int):
        """Applies a Hadamard gate on the given qubit.

        Parameters
        ----------
        qubit : :obj:`int`
            Qubit on which the Hadamard gate is applied.

        Examples
        --------
        >>> P = PauliOperator([Y(0), Z(1), X(2)], 3)
        >>> P.apply_H(0)
        >>> P.apply_H(1)
        >>> P
        PauliOperator([Y(0), X(1), X(2)], 3)

        """
        self.apply_Clifford(_H_TABLEAU, qubit)

    def apply_S(self, qubit: int):
        """Applies a S gate on the given qubit.

        Parameters
        ----------
        qubit : :obj:`int`
            Qubit on which the S gate is applied.

        Examples
        --------
        >>> P = PauliOperator([Y(0), Z(1), X(2)], 3)
        >>> P.apply_S(0)
        >>> P.apply_S(1)
        >>> P.apply_S(2)
        >>> P
        PauliOperator([X(0), Z(1), Y(2)], 3)

        """
        self.apply_Clifford(_S_TABLEAU, qubit)

    def apply_SDG(self, qubit: int):
        """Applies a S :math:`{\\!}^{†}` gate on the given qubit.

        Parameters
        ----------
        qubit : :obj:`int`
            Qubit on which the S :math:`{\\!}^{†}` gate is applied.

        Examples
        --------
        >>> P = PauliOperator([Y(0), Z(1), X(2)], 3)
        >>> P.apply_SDG(0)
        >>> P.apply_SDG(1)
        >>> P.apply_SDG(2)
        >>> P
        PauliOperator([X(0), Z(1), Y(2)], 3)

        """
        self.apply_Clifford(_SDG_TABLEAU, qubit)

    def apply_SWAP(self, qubit1: int, qubit2: int):
        """Applies a SWAP gate between the two qubits.

        Parameters
        ----------
        qubit1 : :obj:`int`
            First qubit on which to apply the SWAP gate.
        qubit2 : :obj:`int`
            Second qubit on which to apply the SWAP gate.

        Examples
        --------
        >>> P = PauliOperator([Y(0), Z(1), X(2)], 4)
        >>> P.apply_SWAP(2, 0)
        >>> P.apply_SWAP(3, 1)
        >>> P
        PauliOperator([X(0), Y(2), Z(3)], 4)

        """
        self.apply_Clifford(_SWAP_TABLEAU, qubit1, qubit2)

    def commute(self, other: "PauliOperator") -> bool:
        """Checks for commutation with the given Pauli operator.

        Parameters
        ----------
        other : :class:`PauliOperator`
            Pauli operator w.r.t. which the commutation is checked.

        Returns
        -------
        :obj:`bool`
            :obj:`True` if the two operators commute, otherwise
            :obj:`False`.

        Raises
        ------
        :obj:`TypeError`
            Raised when given a non-:class:`PauliOperator`.

        :obj:`ValueError`
            Raised when the two Pauli operators do not involve the same number of qubits.

        Examples
        --------
        >>> P = PauliOperator([X(0), Z(2)], 3)
        >>> Q = PauliOperator([Y(0), Y(1), Y(2)], 3)
        >>> P.commute(Q)
        True
        >>> P.commute(PauliOperator([X(1)], 2))
        Traceback (most recent call last):
            ...
        ValueError: A Pauli operator over 3 cannot be checked for commutation with a PauliOperator over 2.
        >>> P.commute(PauliOperator([Z(0)], 3))
        False

        """
        if not isinstance(other, PauliOperator):
            raise TypeError(f"Commutation can only be checked against a PauliOperator, found {type(other).__name__}.")

        if self.nb_qubits != other.nb_qubits:
            raise ValueError(f"A Pauli operator over {self.nb_qubits} cannot be checked for commutation with a PauliOperator over {other.nb_qubits}.")

        return sum(not self.paulis[i].commute(other.paulis[i]) for i in range(self.nb_qubits)) % 2 == 0

    def simplify(self, available: list["PauliOperator"]) -> "PauliOperator":
        """Returns a simplified version of this operator.

        This function computes a simplified version of this operator
        by greedily trying to reduce its weight by multiplying it with
        given available operators.

        The typical use case is the simplification of a Pauli operator
        by the stabilizers of a stabilizer code.

        This is a greedy heuristic to a hard optimization problem, so
        no guarantee of optimality are given.

        Parameters
        ----------
        available : list[PauliOperator]
            List of Paulioperators that are available to try reducing
            the weight.

        Returns
        -------
        PauliOperator
            Simplified version of this operator [modulo available
            operators].

        Examples
        --------
        >>> P = PauliOperator([Z(0), X(1), Y(3), Y(4)], 5)
        >>> available = [PauliOperator([Z(0), Z(1)], 5), PauliOperator([Z(1), Z(2)], 5), PauliOperator([Z(2), Z(3)], 5), PauliOperator([Z(3), Z(4)], 5)]
        >>> P.simplify(available)
        PauliOperator([Y(1), X(3), X(4)], 5)

        """
        flag = True
        simplified = self.copy()
        while flag:
            flag = False
            for s in available:
                if simplified._simplification_potential(s) < 0:
                    simplified = simplified * s
                    flag = True
        return simplified

    def isimplify(self, available: list["PauliOperator"]) -> "PauliOperator":
        """Simplifies this operator in-place.

        This function computes a simplified version of this operator
        by greedily trying to reduce its weight by multiplying it with
        given available operators.

        The typical use case is the simplification of a Pauli operator
        by the stabilizers of a stabilizer code.

        This is a greedy heuristic to a hard optimization problem, so
        no guarantee of optimality are given.

        Parameters
        ----------
        available : list[PauliOperator]
            List of Paulioperators that are available to try reducing
            the weight.

        Returns
        -------
        PauliOperator
            This operator after being simplified.

        Examples
        --------
        >>> P = PauliOperator([Z(0), X(1), Y(3), Y(4)], 5)
        >>> available = [PauliOperator([Z(0), Z(1)], 5), PauliOperator([Z(1), Z(2)], 5), PauliOperator([Z(2), Z(3)], 5), PauliOperator([Z(3), Z(4)], 5)]
        >>> P.isimplify(available)
        PauliOperator([Y(1), X(3), X(4)], 5)

        """
        flag = True
        while flag:
            flag = False
            for s in available:
                if self._simplification_potential(s) < 0:
                    self *= s
                    flag = True
        self.__support = None
        self.__support_as_set = None
        return self

    @classmethod
    def from_str(cls, strg: Union[str, "PauliOperator"], bigendian: bool = False) -> "PauliOperator":
        """Constructor of PauliOperator using a string under a Qiskit little endian format.

        Parameters
        ----------
        strg : Union[str, PauliOperator]
            String of the form '+PPP...P' with Ps being multiple kinds of Pauli
            matrix. The __str__ method will first be used if
            a :class:`PauliOperator` is given instead.

        bigendian : bool, optional
            If :obj:`True`, the opposite of qiskit endianness is used (ie big-endian).
            Beware when toggling this option and not giving a :obj:`str` input.

        Returns
        -------
        PauliOperator
            A new instance of :class:`PauliOperator`.

        Raises
        ------
        ValueError
            Provided string does not conform to the expected format.
        TypeError
            Provided argument is not of type :obj:`str` (nor a
            :class:`PauliOperator`).

        Notes
        -----
        This function uses qiskit's endianess to better interface with it.
        You can use it to copy a PauliOperator by giving it as the argument.

        Examples
        --------
        >>> PauliOperator.from_str("+XYZI")
        PauliOperator([Z(1), Y(2), X(3)], 4)
        >>> PauliOperator.from_str("+XYZI", bigendian=True)
        PauliOperator([X(0), Y(1), Z(2)], 4)
        >>> PauliOperator.from_str(PauliOperator([Z(1), Y(2), X(3)], 4))
        PauliOperator([Z(1), Y(2), X(3)], 4)
        >>> PauliOperator.from_str("XYZI")
        Traceback (most recent call last):
            ...
        ValueError: Given argument must be a string of the form '+PPP...P' with Ps being multiple kinds of Pauli matrix, got XYZI.
        """
        if isinstance(strg, PauliOperator):
            strg = str(strg)

        if not isinstance(strg, str):
            raise TypeError(f"Given argument must be a string of the form '+PPP...P' with Ps being multiple kinds of Pauli matrix, got {strg}.")

        if strg[0] != '+' or not set(strg[1:]) <= _PAULIS:
            raise ValueError(f"Given argument must be a string of the form '+PPP...P' with Ps being multiple kinds of Pauli matrix, got {strg}.")

        if bigendian:
            return cls([Pauli(s, i) for (i, s) in enumerate(strg[1:])], len(strg) - 1)
        return cls([Pauli(s, i) for (i, s) in enumerate(strg[1:][::-1])], len(strg) - 1)

    def translate(self, mapping: Optional[dict[int, int]] = None, n: Optional[int] = None) -> "PauliOperator":
        """Returns a translated version of this operator.

        A translation of the operator to another context, being it a
        second similar quantum system such as a copy of a stabilizer
        code, or a remapping of the qubits.

        Parameters
        ----------
        mapping : dict[int, int], optional
            Remapping of the qubits
        n : int, optional
            "Space" translation of the qubit indices.

        Notes
        -----
        Both arguments cannot be specified at the same time.
        The mapping argument will take priority. This method
        should be combined to :meth:`extend` if one wants both
        functionalities.

        Returns
        -------
        PauliOperator
            A translated version of the operator.

        Examples
        --------
        >>> P = PauliOperator.from_str("+XZX")
        >>> P.translate(n=3)
        PauliOperator([X(3), Z(4), X(5)], 6)
        >>> P.translate(mapping={0: 1, 1: 2, 2: 0})
        PauliOperator([X(0), X(1), Z(2)], 3)
        >>> P.translate(n=3, mapping={0: 1, 1: 2, 2: 0}) # Warns you that a parameter was ignored.
        PauliOperator([X(0), X(1), Z(2)], 3)

        """
        if mapping is not None and n is not None:
            warnings.warn("Both parameter specified in call to translate, ignoring the n parameter", RuntimeWarning)

        if mapping is None:
            paulis = [Pauli(p.kind, p.qubit + n) for p in self.paulis]
            return PauliOperator(paulis, self.nb_qubits + n)

        paulis = [Pauli(p.kind, mapping[p.qubit] if p.qubit in mapping else p.qubit)
                  for p in self.paulis if p.kind != "I"]
        return PauliOperator(paulis, self.nb_qubits)

    def extend(self, n: int):
        """Extends the domain of the operator.

        Broaden the context of a Pauli operator without changing the
        qubits it affects.

        Parameters
        ----------
        n : int
            Number of additional qubits to consider. Must be positive.

        Raises
        ------
        ValueError
            Raised when provided number is negative.

        TypeError
            Raised when provided parameter is not of type :obj:`int`.

        Examples
        --------
        >>> P = PauliOperator.from_str("+XZX")
        >>> P.extend(3)
        >>> P
        PauliOperator([X(0), Z(1), X(2)], 6)

        """
        if not isinstance(n, int):
            raise TypeError(f"Number of qubits should be of type int, got {type(n).__name__}.")
        if n < 0:
            raise ValueError(f"Number of additional qubits should be positive, got {n}.")

        self.__paulis += [Pauli("I", i)
                          for i in range(self.nb_qubits, self.nb_qubits + n)]
        self.__nb_qubits += n
        self.__support = None
        self.__support_as_set = None

    def __mul__(self, other: "PauliOperator") -> "PauliOperator":
        """Special function for the multiplication of Pauli operators.

        Parameters
        ----------
        other : PauliOperator
            Pauli operator to use for multiplication.

        Returns
        -------
        PauliOperator
            Operation result.

        Examples
        --------
        >>> o1 = PauliOperator([X(0), Y(1)], 3)
        >>> o2 = PauliOperator([X(1), Z(2)], 3)
        >>> o1 * o2
        PauliOperator([X(0), Z(1), Z(2)], 3)
        >>> o1 *= o1
        >>> o1
        PauliOperator([], 3)

        """
        if not isinstance(other, PauliOperator):
            raise TypeError(f"Cannot multiply a PauliOperator to a {type(other).__name__}.")

        if self.nb_qubits != other.nb_qubits:
            raise ValueError(f"Cannot multiply two PauliOperators acting on different number of qubits ({self.nb_qubits} != {other.nb_qubits}).")

        foo = self.copy()
        for (p, q) in zip(foo.paulis, other.paulis):
            p *= q

        return foo

    def to_symplectic_array(self) -> NDArray[np.int64]:
        """
        Alias for :meth:`to_symplectic_vector` with parameter `zfirst` set to :obj:`True`.
        """
        return self.to_symplectic_vector(zfirst=True)

    def to_symplectic_vector(self, zfirst: bool = False) -> NDArray[np.int64]:
        """Returns the expression of this Pauli operator as a symplectic vector.

        A :math:`n`-qubit Pauli operator can be expressed by a :math:`2n`-vector
        over :math:`\\mathbb{Z}_2` by putting a 1 in its :math:`i`-th coordinate if
        it acts as X or Y on qubit :math:`i` or 0 otherwise and by putting a
        1 in its :math:`i`-th coordinate if it acts as Z or Y on qubit :math:`i` or
        0 otherwise. The role of X and Z can be exchanged by toggling
        the `zfirst` parameter.

        Parameters
        ----------
        zfirst : bool
            If :obj:`True`, the vector will described the Z part
            first. Default to :obj:`False`

        Returns
        -------
        NDArray[np.int64]
            Symplectic vector expression of the Pauli operator.

        Examples
        --------
        >>> P = PauliOperator([X(0), Y(1), Z(2)], 3)
        >>> P.to_symplectic_vector()
        array([1, 1, 0, 0, 1, 1])
        >>> P.to_symplectic_vector(True)
        array([0, 1, 1, 1, 1, 0])
        >>> P.to_symplectic_array() # Alias for the former
        array([0, 1, 1, 1, 1, 0])

        """
        x, z = ("Z", "X") if zfirst else ("X", "Z")

        return np.array(list(chain((int(pauli.kind in (x, "Y")) for pauli in self.paulis),
                                   (int(pauli.kind in (z, "Y")) for pauli in self.paulis))))

    def _simplification_potential(self, other: "PauliOperator") -> int:
        return sum(_COST[p.kind + q.kind] for p, q in zip(self.paulis, other.paulis))

    def __getitem__(self, index):
        return self.__paulis[index]

    def __setitem__(self, index, pauli):
        self.__paulis[index] = pauli
        self.__support = None
        self.__support_as_set = None

    @property
    def support(self) -> list[int]:
        """Support of this Pauli operator."""
        if self.__support is None:
            self.__support = sorted(p.qubit for p in self.__paulis if p.kind != "I")
        return self.__support

    @property
    def support_as_set(self) -> set[int]:
        """Support of this Pauli operator as a set."""
        if self.__support_as_set is None:
            self.__support_as_set = set(self.support)
        return self.__support_as_set

    @property
    def isX(self) -> bool:
        """
        Returns :obj:`True` when the non-trivial paulis are all Xs.
        """
        return all(p.kind in ("I", "X") for p in self.paulis)

    @property
    def isZ(self) -> bool:
        """
        Returns :obj:`True` when the non-trivial paulis are all Zs.
        """
        return all(p.kind in ("I", "Z") for p in self.paulis)

    def strip(self, kind: str) -> bool:
        """Strips this operator from its expressed X or Z Pauli matrices.

        The operator will projected to an all-X form or an all-Z form,
        modifying its `paulis` attribute to reflect it.

        Parameters
        ----------
        kind : str
            Kind of Pauli matrices to remove

        Returns
        -------
        bool
            Indicates whether some Pauli matrices were removed.

        Raises
        ------
        ValueError
            Raised when a wrong value of `kind` is provided.

        TypeError
            Raised when given a non-:obj:`str` as parameter.

        Examples
        --------
        >>> P = PauliOperator.from_str("+ZYX")
        >>> P.strip("X")
        True
        >>> P
        PauliOperator([Z(1), Z(2)], 3)
        >>> P.strip("Z")
        True
        >>> P
        PauliOperator([], 3)
        >>> P.strip("X")
        False

        """
        if not isinstance(kind, str):
            raise TypeError(f"Expected a kind parameter of type str, found {type(kind).__name__}.")

        if kind not in ("X", "Z"):
            raise ValueError(f"Allowed values for kind parameter are 'X' and 'Z', got {kind}.")

        flag = False
        for p in self.__paulis:
            if p.kind == kind:
                flag = True
                p.kind = "I"
            elif p.kind == "Y":
                flag = True
                p.kind = "X" if kind == "Z" else "Z"

        self.__support = None
        self.__support_as_set = None

        return flag

    def measure(self, measure_count: Iterator[int]) -> Optional[int]:
        """Measures the Pauli operator at the next time step.

        The next value of the iterator over increasing measurement
        time indices is added to the list of measurement time of this
        operator. If the same iterator is used for several Pauli
        operators, this ensures that no measurement happens
        concurrently, thus guaranteeing coherence.

        Parameters
        ----------
        measure_count : Iterator[int]
            Increasing iterator over time indices

        Returns
        -------
        :obj:`int` | :obj:`None`
            Time of the previous measurement of this operator, if any

        Raises
        ------
        RuntimeError
            Raised if the measurement protocol was incorrectly
            started.

        Examples
        --------
        >>> P = PauliOperator([Z(0)], 2)
        >>> Q = PauliOperator([Z(1)], 2)
        >>> measure_schedule = iter(range(6))
        >>> P.measure(measure_schedule)
        >>> Q.measure(measure_schedule)
        >>> P.measure(measure_schedule)
        0
        >>> Q.measure(measure_schedule)
        1
        >>> P.measure(measure_schedule)
        2
        >>> Q.measure(measure_schedule)
        3
        >>> P.last_measure_time
        [None, 0, 2, 4]
        >>> Q.last_measure_time
        [None, 1, 3, 5]

        """
        try:
            foo = next(measure_count)
            if self.last_measure_time[-1] is None or self.last_measure_time[-1] < foo:
                self.last_measure_time.append(foo)
                return self.last_measure_time[-2]
            if self.last_measure_time[-1] >= foo:
                raise RuntimeError("Measurement protocol not started properly")
        except AttributeError:
            raise RuntimeError("Measurement protocol not started properly")

    def reset(self):
        """Resets the measurement times of this operator.

        This method resets the `last_measure_time` attribute to its
        default value :code:`[None]`.

        Examples
        --------
        >>> P = PauliOperator([Z(0)], 2)
        >>> measure_schedule = iter(range(6))
        >>> P.measure(measure_schedule)
        >>> P.measure(measure_schedule)
        0
        >>> P.measure(measure_schedule)
        1
        >>> P.last_measure_time
        [None, 0, 1, 2]
        >>> P.reset()
        >>> P.last_measure_time
        [None]

        """

        self.__last_measure_time = [None]


class Stabilizer(PauliOperator):
    """
    Dummy class to serve as alias to distinguish stabilizer operators
    from other Pauli operators in a stabilizer code context.
    """

    def __init__(self, paulis: Optional[list[Pauli]] = None, nb_qubits: int = 1, last_measure_time: Optional[list[Optional[int]]] = None):
        """Default constructor of the Stabilizer class.

        Parameters
        ----------
        paulis : list[Pauli]
            List of :class:`~stabcodes.Pauli.Pauli` composing the Pauli
            operator. Set to `nb_qubits` identity matrices if None.
        nb_qubits : :obj:`int`
            Number of qubits upon which the Pauli operator will act.
        last_measure_time : list[Optional[int]], optional
            List of time indices where this operator was measured.

        Raises
        ------
        ValueError
            Raised if a provided Pauli acts on a qubit not covered by
            this operator (e.g. if Pauli.qubit >= `nb_qubits`).
        TypeError
            Raised if a non-:class:`~stabcodes.Pauli.Pauli` is provided.

        Examples
        --------
        >>> Stabilizer(nb_qubits=2)
        Stabilizer([], 2)
        >>> Stabilizer([X(0), Y(3), Z(1), Z(0)], 4)
        Stabilizer([Y(0), Z(1), Y(3)], 4)
        >>> Stabilizer([X(1)], 1)
        Traceback (most recent call last):
            ...
        ValueError: Given Pauli (X(1)) is applied on an out-of-range qubit (range: (0, 0)).
        """
        super().__init__(paulis, nb_qubits, last_measure_time)

    def __repr__(self) -> str:
        return f"Stabilizer({[pauli for pauli in self.paulis if pauli.kind != 'I']}, {self.nb_qubits})"

    def copy(self) -> "Stabilizer":
        """Returns a copy of this stabilizer operator.

        Returns
        -------
        Stabilizer
            An independent copy of this stabilizer operator.

        Examples
        --------
        >>> a = Stabilizer([X(0), Z(1)], 2)
        >>> b = a.copy()
        >>> a *= b
        >>> a
        Stabilizer([], 2)
        >>> b
        Stabilizer([X(0), Z(1)], 2)

        """
        return self.__copy__()

    def __copy__(self) -> "Stabilizer":
        return Stabilizer(paulis=[p.copy() for p in self.paulis],
                          nb_qubits=self.nb_qubits,
                          last_measure_time=list(self.last_measure_time))


class Stabilizer2D(Stabilizer):
    """
    Class implementing the logic of stabilizer operators of a stabilizer code
    over a two-dimensional space.

    Attributes
    ----------

    paulis : list[Pauli]
        List of :class:`Pauli` matrices composing
        the stabilizer operator.
    nb_qubits : :obj:`int`
        Number of qubits upon which the stabilizer operator acts.
        Must be positive.
    order: list[int]
        Ordered list of the qubits acted upon by the stabilizer. The order describes
        the vertices of the stabilizer in cyclic trigonometric order.
    support : list[int]
        List of qubits upon which the stabilizer operator acts non-trivially.
    support_as_set : set[int]
        Set of qubits upon which the stabilizer operator acts non-trivially.
    last_measure_time : int, optional
        Time of last measurement of this operator in a broader context of circuit experiment.

    Notes
    -----
    Stabilizer operators on a given number of qubits form a multiplicative group.
    Thus, elements of this class can be multiplied if they act on the same
    number of qubits.

    The `order` field need to be kept updated by the user, as there is no way to automatically
    decide of a new ordering for a stabilizer whose support as changed. Thus, if your
    stabilizer code is not 2D local code over an orientable surface, or if you do not care much
    about the stabilizer supports, you might prefer using :class:`Stabilizer`
    for your stabilizer operators.

    Examples
    --------
    >>> s1 = Stabilizer2D([X(0), Z(1), Y(2)], 3)
    >>> s2 = Stabilizer2D([Z(1), X(0), Y(2)], 3)
    >>> s1 == s2
    False
    >>> s3 = Stabilizer2D([X(0), Z(1), Y(2)], 3, order=[2, 0, 1])
    >>> s3
    Stabilizer2D([Y(2), X(0), Z(1)], 3)
    >>> s3 == s1
    True

    """

    def __init__(self, paulis: Optional[list["Pauli"]] = None, nb_qubits: int = 1, order: Optional[list[int]] = None, last_measure_time: Optional[list[Optional[int]]] = None):
        """Default constructor of the Stabilizer2D class

        Parameters
        ----------
        paulis : list[Pauli]
            List of :class:`~stabcodes.Pauli.Pauli` composing the Pauli
            operator. Set to `nb_qubits` identity matrices if None.
        nb_qubits : :obj:`int`
            Number of qubits upon which the Pauli operator will act.
        order : list[int], optional
            Ordering of the qubit that related to the geometrical aspect
            of the stabilizer.
        last_measure_time : list[Optional[int]], optional
            List of time indices where this operator was measured.

        Raises
        ------
        ValueError
            Raised if a provided Pauli acts on a qubit not covered by
            this operator (e.g. if Pauli.qubit >= `nb_qubits`) or if the
            order parameter does not exactly cover the support.
        TypeError
            Raised if a non-:class:`~stabcodes.Pauli.Pauli` is provided.

        Examples
        --------
        >>> Stabilizer2D(nb_qubits=2)
        Stabilizer2D([], 2)
        >>> Stabilizer2D([X(0), Y(3), Z(1), Z(0)], 4, [3, 0, 1])
        Stabilizer2D([Y(3), Y(0), Z(1)], 4)
        >>> Stabilizer2D([X(0), Y(3), Z(1), Z(0)], 4, [3, 0, 1, 2])
        Traceback (most recent call last):
            ...
        ValueError: Provided order does not match the given support ...
        >>> Stabilizer2D([X(1)], 1)
        Traceback (most recent call last):
            ...
        ValueError: Given Pauli (X(1)) is applied on an out-of-range qubit (range: (0, 0)).
        """
        super().__init__(paulis, nb_qubits, last_measure_time)
        if order is not None and set(order) != self.support_as_set:
            raise ValueError(f"Provided order does not match the given support ({set(order)} != {self.support_as_set}.")
        if paulis:
            self.order = [p.qubit for p in paulis if p.kind != "I"] if order is None else order
        else:
            self.order = []

    @classmethod
    def from_str(cls, strg: Union[str, "PauliOperator"], bigendian: bool = False, order: Optional[list[int]] = None) -> "PauliOperator":
        """Constructor of :class:`Stabilizer2D` using a string under a Qiskit little endian format.

        Parameters
        ----------
        strg : Union[str, PauliOperator]
            String of the form '+PPP...P' with Ps being multiple kinds of Pauli
            matrix. The __str__ method will first be used if
            a :class:`PauliOperator` is given instead.

        bigendian : bool, optional
            If :obj:`True`, the opposite of qiskit endianness is used (ie big-endian).
            Beware when toggling this option and not giving a :obj:`str` input.

        order : list[int], optional
            Ordering of the qubit that related to the geometrical aspect
            of the stabilizer.

        Returns
        -------
        PauliOperator
            A new instance of :class:`PauliOperator`.

        Raises
        ------
        ValueError
            Provided string does not conform to the expected format.
        TypeError
            Provided argument is not of type :obj:`str` (nor a
            :class:`PauliOperator`).

        Notes
        -----
        This function uses qiskit's endianess to better interface with it.
        You can use it to copy a :class:`PauliOperator` by giving it as the argument.

        Examples
        --------
        >>> Stabilizer2D.from_str("+XYZI", order=[2, 1, 3])
        Stabilizer2D([Y(2), Z(1), X(3)], 4)
        >>> Stabilizer2D.from_str("+XYZI", bigendian=True, order=[1, 0, 2])
        Stabilizer2D([Y(1), X(0), Z(2)], 4)
        >>> Stabilizer2D.from_str(Stabilizer2D([Z(1), Y(2), X(3)], 4))
        Stabilizer2D([Z(1), Y(2), X(3)], 4)
        >>> Stabilizer2D.from_str("XYZI")
        Traceback (most recent call last):
            ...
        ValueError: Given argument must be a string of the form '+PPP...P' with Ps being multiple kinds of Pauli matrix, got XYZI.
        """
        if isinstance(strg, PauliOperator):
            strg = str(strg)

        if not isinstance(strg, str):
            raise TypeError(f"Given argument must be a string of the form '+PPP...P' with Ps being multiple kinds of Pauli matrix, got {strg}.")

        if strg[0] != '+' or not set(strg[1:]) <= _PAULIS:
            raise ValueError(f"Given argument must be a string of the form '+PPP...P' with Ps being multiple kinds of Pauli matrix, got {strg}.")

        if bigendian:
            return cls([Pauli(s, i) for (i, s) in enumerate(strg[1:])], len(strg) - 1, order=order)
        return cls([Pauli(s, i) for (i, s) in enumerate(strg[1:][::-1])], len(strg) - 1, order=order)

    @classmethod
    def from_support(cls, support: list[int], kind: str = "Z", nb_qubits: int = 1, order: Optional[list[int]] = None):
        """
        Creates a :class:`PauliOperator` with the same Pauli on given support

        Parameters
        ----------

        support: list[int]
            The support acted upon not trivially.
        kind: str
            The kind of Pauli wanted. Can take only the value "X" | "Y" | "Z".
        nb_qubits: int
            Number of qubits in context
        order: list[int], optional
            Ordering of the qubit that related to the geometrical aspect
            of the stabilizer.

        Returns
        -------
        Stabilizer2D
            Pauli operator acting as the given kind on the given support

        Examples
        --------
        >>> Stabilizer2D.from_support([1,3,5], "Z", 7)
        Stabilizer2D([Z(1), Z(3), Z(5)], 7)
        >>> Stabilizer2D.from_support([1, 2], "X", 7)
        Stabilizer2D([X(1), X(2)], 7)
        """
        func = X if kind == "X" else Z if kind == "Z" else Y if kind == "Y" else None
        return cls([func(qb) for qb in support], nb_qubits)

    def apply_CNOT(self, control: int, target: int):
        """Overloads the inherited method to disable it, use :meth:`apply_CNOT_with_caution` instead.

        As the order parameter cannot be updated automatically, this
        inherited method is overloaded to raise an error if used. The
        explicit change of name to the similar method
        :meth:`apply_CNOT_with_caution` is done to avoid unknowingly
        modifying the support without changing the order field.

        Parameters
        ----------
        control : int
            Controlled qubit
        target : int
            Targeted qubit

        Raises
        ------
        AttributeError
            Raised when called. Use :meth:`apply_CNOT_with_caution` instead.

        """
        raise AttributeError(
            "CNOT can't be applied directly to a Stabilizer2D, use apply_CNOT_with_caution")

    def apply_CZ(self, qubit1: int, qubit2: int):
        """Overloads the inherited method to disable it, use :meth:`apply_CZ_with_caution` instead.

        As the order parameter cannot be updated automatically, this
        inherited method is overloaded to raise an error if used. The
        explicit change of name to the similar method
        :meth:`apply_CZ_with_caution` is done to avoid unknowingly
        modifying the support without changing the order field.

        Parameters
        ----------
        qubit1 : int
            First qubit acted upon.
        qubit2 : int
            Second qubit acted upon.

        Raises
        ------
        AttributeError
            Raised when called. Use :meth:`apply_CZ_with_caution` instead.

        """
        raise AttributeError(
            "CNOT can't be applied directly to a Stabilizer2D, use apply_CZ_with_caution")

    def apply_CNOT_with_caution(self, control: int, target: int) -> bool:
        """Applies a CNOT gate between the `control` qubit and the `target` qubit,
           while reminding the user to update the order field.

        Parameters
        ----------
        control : :obj:`int`
            Control qubit of the CNOT.
        target : :obj:`int`
            Target qubit of the CNOT.

        Returns
        -------
        bool
            :obj:`True` if the support of the stabilizer has changed, otherwise :obj:`False`.

        Examples
        --------
        >>> P = Stabilizer2D([Y(0), Z(1), X(2)], 3)
        >>> P.apply_CNOT_with_caution(2, 0)
        False
        >>> P
        Stabilizer2D([Z(0), Z(1), Y(2)], 3)
        >>> P.apply_CNOT_with_caution(0, 1)
        True
        >>> P.order.remove(0)
        >>> P
        Stabilizer2D([Z(1), Y(2)], 3)

        """
        old_support = self.support
        PauliOperator.apply_CNOT(self, control, target)
        return old_support != self.support

    def apply_CZ_with_caution(self, qubit1: int, qubit2: int) -> bool:
        """Applies a CZ gate between the two qubits, while reminding the user
           to update the order field.

        Parameters
        ----------
        qubit1 : :obj:`int`
            First qubit on which to apply the CZ gate.
        qubit2 : :obj:`int`
            Second qubit on which to apply the CZ gate.

        Returns
        -------
        bool
            :obj:`True` if the support of the stabilizer has changed, otherwise :obj:`False`.

        Examples
        --------
        >>> P = Stabilizer2D([Y(0), X(2), Z(1)], 3)
        >>> P.apply_CZ_with_caution(2, 0)
        False
        >>> P
        Stabilizer2D([X(0), Y(2), Z(1)], 3)
        >>> P.apply_CZ_with_caution(0, 1)
        True
        >>> P.order.remove(1)
        >>> P
        Stabilizer2D([X(0), Y(2)], 3)

        """
        old_support = self.support
        PauliOperator.apply_CZ(self, qubit1, qubit2)
        return old_support != self.support

    def apply_SWAP(self, qubit1: int, qubit2: int):
        """Applies a SWAP gate between the two qubits.

        This will update the order field, use :meth:`PauliOperator.apply_SWAP`
        instead if you want to manually change the support to your needs.

        Parameters
        ----------
        qubit1 : :obj:`int`
            First qubit on which to apply the SWAP gate.
        qubit2 : :obj:`int`
            Second qubit on which to apply the SWAP gate.

        Examples
        --------
        >>> P = Stabilizer2D([Y(0), X(2), Z(1)], 4)
        >>> P.apply_SWAP(2, 0)
        >>> P.apply_SWAP(3, 1)
        >>> P
        Stabilizer2D([Y(2), X(0), Z(3)], 4)

        """
        PauliOperator.apply_SWAP(self, qubit1, qubit2)
        indc = None
        indt = None
        if qubit1 in self.order:
            indc = self.order.index(qubit1)
        if qubit2 in self.order:
            indt = self.order.index(qubit2)

        if indc is not None:
            self.order[indc] = qubit2
        if indt is not None:
            self.order[indt] = qubit1

    def __repr__(self) -> str:
        return f"Stabilizer2D({[self.paulis[qb] for qb in self.order if self.paulis[qb].kind != 'I']}, {self.nb_qubits})"

    def __eq__(self, other: object) -> bool:
        """Special function checking for the equality of two :class:`Stabilizer2D`.

        The order field must be the same modulo cyclic permutation of
        the list for equality to hold.

        Parameters
        ----------
        other : object
            Object to check equality against.

        Returns
        -------
        bool
            Whether the two objects are equal.

        Examples
        --------
        >>> s1 = Stabilizer2D([X(0), Z(1), Y(2)], 3)
        >>> s2 = Stabilizer2D([Z(1), X(0), Y(2)], 3)
        >>> s1 == s2
        False
        >>> s3 = Stabilizer2D([X(0), Z(1), Y(2)], 3, order=[2, 0, 1])
        >>> s3
        Stabilizer2D([Y(2), X(0), Z(1)], 3)
        >>> s3 == s1
        True

        """
        if not isinstance(other, Stabilizer2D):
            return False
        if PauliOperator.__eq__(self, other):
            a = self.order.index(self.support[0])
            b = other.order.index(self.support[0])
            for (elem1, elem2) in zip(self.order[a:] + self.order[:a], other.order[b:] + other.order[:b]):
                if elem1 != elem2:
                    break
            else:
                return True
        return False

    def __hash__(self) -> int:
        if self.order:
            mini = self.order.index(min(self.order))
        else:
            mini = 0
        return hash(str(self) + "/".join(str(qb) for qb in self.order[mini:] + self.order[:mini]))

    def __mul__(self, other: "PauliOperator"):
        """Overloads the inherited method to disable it, use :meth:`multiply_with_caution` or :meth:`imultiply_with_caution` instead.

        As the order parameter cannot be updated automatically, this
        inherited method is overloaded to raise an error if used. The
        explicit change of name to the similar method
        use :meth:`multiply_with_caution` or :meth:`imultiply_with_caution` is done
        to avoid unknowingly modifying the support without changing the order field.

        Parameters
        ----------
        control : int
            Controlled qubit
        target : int
            Targeted qubit

        Raises
        ------
        AttributeError
            Raised when called. Use :meth:`multiply_with_caution` or :meth:`multiply_with_caution` instead.

        """
        raise AttributeError("Stabilizer2D cannot be directly multiplied, use multiply_with_caution or imultiply_with_caution instead.")

    def multiply_with_caution(self, other: PauliOperator) -> "Stabilizer2D":
        """Multiplies this stabilizer operator by a Pauli operator.

        This function will yield a :class:`Stabilizer2D` object,
        despite it not necessarily representing a stabilizer
        anymore. :meth:`PauliOperator.__mul__` can be explicitly
        called if this is a problem.

        Parameters
        ----------
        other : PauliOperator
            Pauli operator by which to multiply the stabilizer.

        Returns
        -------
        Stabilizer2D
            Operation result

        Examples
        --------
        >>> S = Stabilizer2D([Y(0), Z(1), X(2)], 3, [0, 2, 1])
        >>> T = Stabilizer.from_str("+XXX")
        >>> U = S.multiply_with_caution(T)
        >>> U.order.remove(2)
        >>> U
        Stabilizer2D([Z(0), Y(1)], 3)

        """
        foo = self.copy()
        foo.imultiply_with_caution(other)
        return foo

    def imultiply_with_caution(self, other: PauliOperator):
        """Multiplies in-place this stabilizer operator by a Pauli operaotr.

        Parameters
        ----------
        other : PauliOperator
            Pauli operator by which to multiply the stabilizer.

        Examples
        --------
        >>> S = Stabilizer2D([Y(0), Z(1), X(2)], 3, [0, 2, 1])
        >>> T = Stabilizer.from_str("+XXX")
        >>> S.imultiply_with_caution(T)
        >>> S.order.remove(2)
        >>> S
        Stabilizer2D([Z(0), Y(1)], 3)

        """
        for (p, q) in zip(self.paulis, other.paulis):
            p *= q
        self.__support = None
        self.__support_as_set = None

    def copy(self) -> "Stabilizer2D":
        """Returns a copy of this stabilizer operator.

        Returns
        -------
        Stabilizer2D
            An independent copy of this stabilizer operator.

        Examples
        --------
        >>> a = Stabilizer2D([X(0), Z(1)], 2)
        >>> b = a.copy()
        >>> a.imultiply_with_caution(b)
        >>> a.order.remove(0)
        >>> a.order.remove(1)
        >>> a
        Stabilizer2D([], 2)
        >>> b
        Stabilizer2D([X(0), Z(1)], 2)

        """
        return self.__copy__()

    def __copy__(self) -> "Stabilizer2D":
        return Stabilizer2D(paulis=[p.copy() for p in self.paulis],
                            nb_qubits=self.nb_qubits,
                            order=list(self.order),
                            last_measure_time=list(self.last_measure_time))

    def translate(self, n: Optional[int] = None, mapping: Optional[dict[int, int]] = None, _m: int = 0) -> "Stabilizer2D":
        """Returns a translated version of this stabilizer.

        A translation of the operator to another context, being it a
        second similar quantum system such as a copy of a stabilizer
        code, or a remapping of the qubits.

        Parameters
        ----------
        mapping : dict[int, int], optional
            Remapping of the qubits
        n : int, optional
            "Space" translation of the qubit indices.
        _m : int
            Negative "space" translation in the indices but not the order.
            Do not use unless you know what you do.

        Notes
        -----
        Both arguments cannot be specified at the same time.
        The mapping argument will take priority. This method
        should be combined to :meth:`extend` if one wants both
        functionalities.

        Returns
        -------
        Stabilizer2D
            A translated version of the operator.

        Examples
        --------
        >>> P = Stabilizer2D.from_str("+XZX")
        >>> P.translate(n=3)
        Stabilizer2D([X(3), Z(4), X(5)], 6)
        >>> P.translate(mapping={0: 2, 1: 1, 2: 0})
        Stabilizer2D([X(2), Z(1), X(0)], 3)
        >>> P.translate(n=3, mapping={0: 1, 1: 2, 2: 0})
        Stabilizer2D([X(1), Z(2), X(0)], 3)

        """
        if mapping is not None and n is not None:
            warnings.warn("Both parameter specified in call to translate, ignoring the n parameter", RuntimeWarning)

        if mapping is None:
            paulis = [Pauli(self.paulis[i].kind, self.paulis[i].qubit + n)
                      for i in self.order]
            return Stabilizer2D(paulis, self.nb_qubits + n - _m, order=[i + n for i in self.order])

        paulis = [Pauli(self.paulis[i].kind, mapping[i] if i in mapping else i)
                  for i in self.order]
        return Stabilizer2D(paulis, self.nb_qubits - _m, order=[mapping[i] if i in mapping else i for i in self.order])

    def simplify(self, available: list[PauliOperator]) -> "Stabilizer2D":
        """Simplifies this operator in-place.

        This function computes a simplified version of this operator
        by greedily trying to reduce its weight by multiplying it with
        given available operators.

        The typical use case is the simplification of a Pauli operator
        by the stabilizers of a stabilizer code.

        This is a greedy heuristic to a hard optimization problem, so
        no guarantee of optimality are given.

        Parameters
        ----------
        available : list[PauliOperator]
            List of Paulioperators that are available to try reducing
            the weight.

        Returns
        -------
        PauliOperator
            This operator after being simplified.

        Examples
        --------
        >>> P = Stabilizer2D([Y(3), Y(4), Z(0), X(1)], 5)
        >>> available = [Stabilizer2D([Z(0), Z(1)], 5), Stabilizer2D([Z(1), Z(2)], 5), Stabilizer2D([Z(2), Z(3)], 5), Stabilizer2D([Z(3), Z(4)], 5)]
        >>> S = P.simplify(available)
        >>> S.order.remove(0)
        >>> S
        Stabilizer2D([X(3), X(4), Y(1)], 5)

        """
        simplified = self.copy()
        simplified.isimplify(available)
        return simplified

    def isimplify(self, available: list[PauliOperator]):
        """Simplifies this operator in-place.

        This function computes a simplified version of this operator
        by greedily trying to reduce its weight by multiplying it with
        given available operators.

        The typical use case is the simplification of a Pauli operator
        by the stabilizers of a stabilizer code.

        This is a greedy heuristic to a hard optimization problem, so
        no guarantee of optimality are given.

        Parameters
        ----------
        available : list[PauliOperator]
            List of Paulioperators that are available to try reducing
            the weight.

        Returns
        -------
        PauliOperator
            This operator after being simplified.

        Examples
        --------
        >>> P = Stabilizer2D([Y(3), Y(4), Z(0), X(1)], 5)
        >>> available = [Stabilizer2D([Z(0), Z(1)], 5), Stabilizer2D([Z(1), Z(2)], 5), Stabilizer2D([Z(2), Z(3)], 5), Stabilizer2D([Z(3), Z(4)], 5)]
        >>> P.isimplify(available)
        >>> P.order.remove(0)
        >>> P
        Stabilizer2D([X(3), X(4), Y(1)], 5)

        """
        flag = True
        while flag:
            flag = False
            for s in available:
                if self._simplification_potential(s) < 0:
                    self.imultiply_with_caution(s)
                    flag = True

        self.__support = None
        self.__support_as_set = None

    def apply_circuit(self, circuit: qiskit.QuantumCircuit, _override_safeguard: Optional[bool] = False):
        """Applies a :obj:`qiskit.circuit.QuantumCircuit` to the Pauli operator (by conjugation) :mod:`string`.

        This function iterates over the gates of the circuit, applying
        each of them in turn.

        Parameters
        ----------
        circuit : :class:`~qiskit.circuit.QuantumCircuit`
            Quantum circuit to apply.
        _override_safeguard : bool, optional
            Whether to bypass the safeguard ensuring that an entangling two-qubit cannot silently be applied.

        Notes
        -----
        If `_override_safeguard` is set to True, the :obj:`Stabilizer2D` will not be
        in a coherent state and will have to be manually updated.

        Raises
        ------
        ValueError
            A gate of the circuit is not among the defined Clifford gate.

        AttributeError
            Raised when called with a circuit containing an entangling gate.
            Use the `_override_safeguard` to bypass, but remember to manually
            update the `order` field.

        Examples
        --------
        >>> P = Stabilizer2D([Z(0)], 2)
        >>> circuit = qiskit.QuantumCircuit(2)
        >>> circuit.h(0)
        <...
        >>> circuit.cx(0, 1)
        <...
        >>> P.apply_circuit(circuit)
        Traceback (most recent call last):
            ...
        AttributeError: CNOT can't be applied directly to a Stabilizer2D, use apply_CNOT_with_caution
        >>> P == Stabilizer2D([Z(0)], 2)
        True
        >>> P.apply_circuit(circuit, True)
        >>> P.order = [0, 1]
        >>> P == Stabilizer2D([X(0), X(1)], 2)
        True
        >>> circuit.t(0)
        <...
        >>> P.apply_circuit(circuit, True)
        Traceback (most recent call last):
            ...
        ValueError: Unknown Clifford gate: t.

        """

        backup = self.copy()
        try:
            for gate in circuit:
                name = gate.operation.name
                qubits = gate.qubits

                if name == "cx":
                    if _override_safeguard:
                        self.apply_CNOT_with_caution(qubits[0]._index,
                                                     qubits[1]._index)
                    else:
                        self.apply_CNOT(qubits[0]._index,
                                        qubits[1]._index)
                elif name == "cz":
                    if _override_safeguard:
                        self.apply_CZ_with_caution(qubits[0]._index,
                                                   qubits[1]._index)
                    else:
                        self.apply_CZ(qubits[0]._index,
                                      qubits[1]._index)
                elif name == "s":
                    self.apply_S(qubits[0]._index)
                elif name == "sdg":
                    self.apply_SDG(qubits[0]._index)
                elif name == "h":
                    self.apply_H(qubits[0]._index)
                elif name == "swap":
                    self.apply_SWAP(qubits[0]._index,
                                    qubits[1]._index)
                elif name == "id":
                    pass
                elif name == "x":
                    pass
                elif name == "y":
                    pass
                elif name == "z":
                    pass
                else:
                    raise ValueError(f"Unknown Clifford gate: {name}.")

        except Exception as e:
            self._PauliOperator__paulis = backup._PauliOperator__paulis
            self.order = backup.order
            raise e


def I(qb: int) -> Pauli:  # noqa: E741, E743
    """Constructor for an Identity Pauli matrix.

    Parameters
    ----------
    qb : :obj:`int`
        Qubit on which the matrix is applied.

    Returns
    -------
    :class:`~stabcodes.Pauli.Pauli`
        Identity on qubit `qb`.

    Examples
    --------
    >>> I(0)
    I(0)

    """

    return Pauli("I", qb)


def X(qb: int) -> Pauli:
    """Constructor for a X-Pauli matrix.

    Parameters
    ----------
    qb : :obj:`int`
        Qubit on which the matrix is applied.

    Returns
    -------
    :class:`~stabcodes.Pauli`
        Pauli X on qubit `qb`.

    Examples
    --------
    >>> X(0)
    X(0)

    """

    return Pauli("X", qb)


def Y(qb: int) -> Pauli:
    """Constructor for a Y-Pauli matrix.

    Parameters
    ----------
    qb : :obj:`int`
        Qubit on which the matrix is applied.

    Returns
    -------
    :class:`~stabcodes.Pauli.Pauli`
        Pauli Y on qubit `qb`.

    Examples
    --------
    >>> Y(0)
    Y(0)

    """

    return Pauli("Y", qb)


def Z(qb: int) -> Pauli:
    """Constructor for a Z-Pauli matrix.

    Parameters
    ----------
    qb : :obj:`int`
        Qubit on which the matrix is applied.

    Returns
    -------
    :class:`~stabcodes.Pauli.Pauli`
        Pauli Z on qubit `qb`.

    Examples
    --------
    >>> Z(0)
    Z(0)

    """
    return Pauli("Z", qb)


if __name__ == "__main__":
    import doctest
    warnings.simplefilter("ignore")
    doctest.testmod(optionflags=doctest.ELLIPSIS)
