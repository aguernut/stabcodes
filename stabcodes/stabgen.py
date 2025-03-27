"""

"""

from typing import Optional, Union
from collections.abc import Sequence, Mapping
from stabcodes.recursivemapping import RecursiveMapping
from stabcodes.pauli import PauliOperator
from qiskit import QuantumCircuit
import numpy as np
from numpy.typing import NDArray


class StabGen(RecursiveMapping):

    def __init__(self, value: Optional[Union[Sequence[PauliOperator], Mapping[str, PauliOperator]]] = None):
        super().__init__(value)

        nb_qubits = None
        for s in self:
            if not isinstance(s, PauliOperator):
                raise TypeError(f"Expected PauliOperator, got {s} ({type(s).__name__})")
            if nb_qubits is None:
                nb_qubits = s.nb_qubits
            elif nb_qubits != s.nb_qubits:
                raise ValueError(f"Stabilizer {repr(s)} does not affect {nb_qubits} qubits")

        self._nb_qubits = nb_qubits

    @property
    def nb_qubits(self) -> int:
        return self._nb_qubits

    def apply_circuit_to_stabs(self, circuit: QuantumCircuit):
        for stab in self:
            stab.apply_circuit(circuit)

    def to_matrix(self) -> NDArray[np.int64]:
        mat = np.zeros((len(self), self._nb_qubits * 2))

        for (i, stab) in enumerate(self):
            mat[i, :] = stab.to_symplectic_array()

        return mat

    def gram_product(self, other: "StabGen") -> NDArray[np.int64]:
        lambd = np.zeros((self._nb_qubits * 2, self._nb_qubits * 2))
        lambd[:self._nb_qubits, self._nb_qubits:] = np.eye(self.nb_qubits)
        lambd[self._nb_qubits:, :self._nb_qubits] = np.eye(self.nb_qubits)

        return (self.to_matrix() @ lambd @ other.to_matrix().T) % 2

    def reset(self):
        for s in self:
            s.reset()

    def extend(self, n: int):
        for s in self:
            s.extend(n)

    def itranslate(self, n: Optional[int] = None, mapping: Optional[dict[int, int]] = None):
        for i, s in enumerate(self):
            self[i] = s.translate(n=n, mapping=mapping)


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

    stabs = StabGen(stabs)
    log = StabGen(log)

    print(stabs.to_matrix())
    print(log.to_matrix())
    print(log.gram_product(log))
