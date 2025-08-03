"""
Specialization of the :class:`RecursiveMapping` data structure to hold stabilizers and Pauli operators.
The intent is to have the stabilizers accessible both through indexing or filtering their Pauli type or possibly their color.
"""

from typing import Optional, Union
from collections.abc import Sequence, Mapping
from stabcodes.recursivemapping import RecursiveMapping
from stabcodes.pauli import PauliOperator, Stabilizer, Stabilizer2D, X, Z
from qiskit import QuantumCircuit
import numpy as np
from numpy.typing import NDArray


class StabGen(RecursiveMapping):
    """
    A :class:`RecursiveMapping` designed to hold :class:`PauliOperator` and :class:`Stabilizer`
    in the context of a stabilizer code.

    Elements are stored hierarchically with string labels characterizing them
    (such as "X" or "Z" for CSS codes, "red" or "blue" or "green" as a second
    level indexing for color code, etc).

    Notes
    -----
    As the labels are lexicographically comparable, the stabilizers are always iterated
    over in the same order, as long as no label change and no element is added.

    Examples
    --------
    >>> stabs = StabGen({"X": [Stabilizer.from_support([0, 1], "X", 3), Stabilizer.from_support([1, 2], "X", 3)], "Z": [Stabilizer.from_support([0, 1, 2], "Z", 3)]})
    >>> stabs
    StabGen({'X': StabGen([Stabilizer([X(0), X(1)], 3), Stabilizer([X(1), X(2)], 3)]), 'Z': StabGen([Stabilizer([Z(0), Z(1), Z(2)], 3)])})
    >>> colored_stabs = StabGen({"red": stabs, "blue": {"X": [Stabilizer.from_support([0, 2], "X", 3)]}})
    >>> colored_stabs
    StabGen({'blue': StabGen({'X': StabGen([Stabilizer([X(0), X(2)], 3)])}), 'red': StabGen({'X': StabGen([Stabilizer([X(0), X(1)], 3), Stabilizer([X(1), X(2)], 3)]), 'Z': StabGen([Stabilizer([Z(0), Z(1), Z(2)], 3)])})})

    """

    def __init__(self, value: Optional[Union[Sequence[PauliOperator], Mapping[str, PauliOperator]]] = None):
        """Builds a StabGen from a sequence or a possibly recursive uniform mapping, ending with sequences of :class:`Paulioperator`.

        Parameters
        ----------
        value: Union[Sequence[PauliOperator], Mapping[str, PauliOperator]], optional
            When :obj:`None`, builds the empty :class:`StabGen`.
            When given a sequence of :class:`PauliOperator`, builds a base level :class:`StabGen` containing its elements.
            When given a :class:`StabGen`, performs a shallow copy.
            When given a mapping, recursively calls itself to build the :class:`StabGen`.

        Raises
        ------
        TypeError
            Raised when non-uniform keys would be used, or a non-:class:`PauliOperator` element is given.
        ValueError
            Raised if some elements do not affect the same number of qubits.

        Examples
        --------
        >>> stabs = StabGen({"X": [Stabilizer.from_support([0, 1], "X", 3), Stabilizer.from_support([1, 2], "X", 3)], "Z": [Stabilizer.from_support([0, 1, 2], "Z", 3)]})
        >>> stabs
        StabGen({'X': StabGen([Stabilizer([X(0), X(1)], 3), Stabilizer([X(1), X(2)], 3)]), 'Z': StabGen([Stabilizer([Z(0), Z(1), Z(2)], 3)])})
        >>> colored_stabs = StabGen({"red": stabs, "blue": {"X": [Stabilizer.from_support([0, 2], "X", 3)]}})
        >>> colored_stabs
        StabGen({'blue': StabGen({'X': StabGen([Stabilizer([X(0), X(2)], 3)])}), 'red': StabGen({'X': StabGen([Stabilizer([X(0), X(1)], 3), Stabilizer([X(1), X(2)], 3)]), 'Z': StabGen([Stabilizer([Z(0), Z(1), Z(2)], 3)])})})

        """
        super().__init__(value)

        nb_qubits = None
        for s in self:
            if not isinstance(s, PauliOperator):
                raise TypeError(f"Expected PauliOperator, got {s} ({type(s).__name__})")
            if nb_qubits is None:
                nb_qubits = s.nb_qubits
            elif nb_qubits != s.nb_qubits:
                raise ValueError(f"{repr(s)} does not affect {nb_qubits} qubits")

        self._nb_qubits = nb_qubits

    @property
    def nb_qubits(self) -> int:
        """Number of qubits affected by the elements."""
        return self._nb_qubits

    def apply_circuit_to_stabs(self, circuit: QuantumCircuit, _override_safeguard: bool = False):
        """Apply the given circuit to all the base-level :class:`PauliOperator` by conjugation.

        Parameters
        ----------
        circuit: :class:`~qiskit.circuit.QuantumCircuit`
            Quantum circuit to apply.
        _override_safeguard : bool, optional
            Whether to bypass the safeguard ensuring that an entangling two-qubit cannot silently be applied.

        Notes
        -----
        If `_override_safeguard` is set to True, the :class:`Stabilizer2D` will not be
        in a coherent state and will have to be manually updated.

        Examples
        --------
        >>> import qiskit
        >>> P = Stabilizer2D([Z(1)], 2)
        >>> Q = Stabilizer2D([X(0)], 2)
        >>> S = StabGen({'X': [Q], 'Z': [P]})
        >>> circuit = qiskit.QuantumCircuit(2)
        >>> circuit.h(0)
        <...
        >>> circuit.cx(0, 1)
        <...
        >>> S.apply_circuit_to_stabs(circuit, True)
        >>> S['Z'][0].order = [0, 1]
        >>> S
        StabGen({'X': StabGen([Stabilizer2D([Z(0)], 2)]), 'Z': StabGen([Stabilizer2D([Z(0), Z(1)], 2)])})

        """
        for stab in self:
            if isinstance(stab, Stabilizer2D):
                stab.apply_circuit(circuit, _override_safeguard)
            else:
                stab.apply_circuit(circuit)

    def to_matrix(self) -> NDArray[np.int64]:
        """Returns a view of this object as a matrix of this symplectic view of its elements."""
        mat = np.zeros((len(self), self._nb_qubits * 2))

        for (i, stab) in enumerate(self):
            mat[i, :] = stab.to_symplectic_array()

        return mat

    def gram_product(self, other: "StabGen") -> NDArray[np.int64]:
        """Builds the gram-like matrix of the symplectic bilinear for to summarize the commutativity relation between all the stored elements of both :class:`StabGen`.

        Parameters
        ----------
        other: StabGen
            Stabgen from which to compute commutativity.

        """
        lambd = np.zeros((self._nb_qubits * 2, self._nb_qubits * 2))
        lambd[:self._nb_qubits, self._nb_qubits:] = np.eye(self.nb_qubits)
        lambd[self._nb_qubits:, :self._nb_qubits] = np.eye(self.nb_qubits)

        return (self.to_matrix() @ lambd @ other.to_matrix().T) % 2

    def reset(self):
        """Resets the stored elements so that they can take part of a new simulation."""
        for s in self:
            s.reset()

    def extend(self, n: int):
        """Extends the domain of all the stored elements.

        Parameters
        ----------
        n: int
            Number of additional qubits to consider. Must be positive.

        """
        for s in self:
            s.extend(n)

    def itranslate(self, n: Optional[int] = None, mapping: Optional[dict[int, int]] = None):
        """Translates in-place the stored elements.

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

        """
        for i, s in enumerate(self):
            self[i] = s.translate(n=n, mapping=mapping)


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)
