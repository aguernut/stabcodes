import numpy as np
from itertools import chain
from copy import copy

PAULIS = {"I", "X", "Y", "Z"}

CNOT_TABLEAU = {"II": "II",
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

CZ_TABLEAU = {"II": "II",
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

SWAP_TABLEAU = {"II": "II",
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

S_TABLEAU = {"I": "I",
             "X": "Y",
             "Y": "X",
             "Z": "Z"}

SDG_TABLEAU = {"I": "I",
               "X": "Y",
               "Y": "X",
               "Z": "Z"}

H_TABLEAU = {"I": "I",
             "X": "Z",
             "Y": "Y",
             "Z": "X"}

COST = {"II": 0, "IX": 1, "IY": 2, "IZ": 1,
        "XI": 0, "XX": -1, "XY": 0, "XZ": 1,
        "YI": 0, "YX": -1, "YY": -2, "YZ": -1,
        "ZI": 0, "ZX": 1, "ZY": 0, "ZZ": -1}


class Pauli:

    def __init__(self, kind="I", qubit=0):
        if kind not in PAULIS:
            raise ValueError(f"Only {PAULIS} are supported.")
        self.kind = kind
        if qubit < 0:
            raise ValueError("Given qubit must be a positive integer.")
        self.qubit = qubit

    def __repr__(self):
        return f"{self.kind}({self.qubit})"

    def __copy__(self):
        return Pauli(self.kind, self.qubit)

    def copy(self):
        return self.__copy__()

    def commute(self, other):
        if (self.qubit != other.qubit) or \
           (self.kind == "I") or \
           (other.kind == "I"):
            return True
        return self.kind == other.kind

    def __eq__(self, other):
        if isinstance(other, Pauli):
            return self.kind == other.kind and self.qubit == other.qubit
        return False

    def __mul__(self, other):
        if not isinstance(other, Pauli):
            raise TypeError(f"Only Paulis can be multiplied to Paulis, not {type(other).__name__}")

        if self.qubit != other.qubit:
            raise ValueError(f"Paulis {self} and {other} must be applied on same qubit.")

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

    def __imul__(self, other):
        foo = self * other
        self.kind = foo.kind

    def __hash__(self):
        return hash(self.kind + "(" + str(self.qubit) + ")")


P = Pauli


def I(qb):
    return P("I", qb)


def X(qb):
    return P("X", qb)


def Z(qb):
    return P("Z", qb)


def Y(qb):
    return P("Y", qb)


class PauliOperator:

    def __init__(self, paulis=None, nb_qubits=1):
        self.nb_qubits = nb_qubits
        if paulis is None:
            paulis = []

        self.paulis = [Pauli("I", i) for i in range(nb_qubits)]

        for pauli in paulis:
            if not isinstance(pauli, Pauli):
                raise TypeError("PauliOperator must be given\
                Paulis, not {type(pauli)}.")
            if pauli.qubit < 0 or pauli.qubit >= nb_qubits:
                raise ValueError(f"Given Pauli ({pauli}) is applied\
                on an out-of-range qubit ({nb_qubits}).")

            self.paulis[pauli.qubit] = self.paulis[pauli.qubit] * pauli
        # self.__support = None
        # self.support

    def __copy__(self):
        if "last_measure_time" in self.__dict__:
            foo = PauliOperator([p.copy() for p in self.paulis], self.nb_qubits)
            foo.last_measure_time = copy(self.last_measure_time)
            return foo
        return PauliOperator([p.copy() for p in self.paulis], self.nb_qubits)

    def __repr__(self):
        return f"PauliOperator({[pauli for pauli in self.paulis if pauli.kind != 'I']}, {self.nb_qubits})"

    def __str__(self):
        return "+" + "".join(pauli.kind for pauli in self.paulis[::-1])

    def __eq__(self, other):
        return str(self) == str(other)

    def __bool__(self):
        return bool(self.support)

    def __hash__(self):
        return hash(str(self))

    def apply_Clifford(self, tableau, *qubits):
        if len(next(iter(tableau))) != len(qubits):
            raise ValueError(f"The given tableau does not affect {len(qubits)} at once.")

        now = "".join(self.paulis[qb].kind for qb in qubits)
        future = tableau[now]

        for (kind, qb) in zip(future, qubits):
            self.paulis[qb] = P(kind, qb)
        # self.__support = None

    def apply_circuit(self, circuit):
        for (gate, qubits, _) in circuit:
            if gate.name == "cx":
                self.apply_CNOT(qubits[0].index,
                                qubits[1].index)
            elif gate.name == "cz":
                self.apply_CZ(qubits[0].index,
                              qubits[1].index)
            elif gate.name == "s":
                self.apply_S(qubits[0].index)
            elif gate.name == "sdg":
                self.apply_SDG(qubits[0].index)
            elif gate.name == "h":
                self.apply_H(qubits[0].index)
            elif gate.name == "swap":
                self.apply_SWAP(qubits[0].index,
                                qubits[1].index)
            elif gate.name == "z":
                pass
            else:
                raise NotImplementedError(f"Unknown gate: {gate.name}")

    def apply_CNOT(self, control, target):
        self.apply_Clifford(CNOT_TABLEAU, control, target)
        self.__support = None

    def apply_CZ(self, control, target):
        self.apply_Clifford(CZ_TABLEAU, control, target)
        self.__support = None

    def apply_H(self, qubit):
        self.apply_Clifford(H_TABLEAU, qubit)
        self.__support = None

    def apply_S(self, qubit):
        self.apply_Clifford(S_TABLEAU, qubit)
        self.__support = None

    def apply_SDG(self, qubit):
        self.apply_Clifford(SDG_TABLEAU, qubit)

    def apply_SWAP(self, qubit1, qubit2):
        self.apply_Clifford(SWAP_TABLEAU, qubit1, qubit2)
        self.__support = None

    def commute(self, other):
        return sum(not self.paulis[i].commute(other.paulis[i]) for i in range(self.nb_qubits)) % 2 == 0

    def simplify(self, available):
        flag = True
        simplified = self.copy()
        while flag:
            flag = False
            for s in available:
                if simplified.simplification_potential(s) < 0:
                    simplified = simplified * s
                    flag = True
        return simplified

    def isimplify(self, available):
        flag = True
        simplified = self.copy()
        while flag:
            flag = False
            for s in available:
                if simplified.simplification_potential(s) < 0:
                    simplified = simplified * s
                    flag = True
        self.paulis = simplified.paulis
        self.__support = None
        return simplified

    @classmethod
    def from_str(cls, strg):
        if isinstance(strg, PauliOperator):
            return strg
        return PauliOperator([Pauli(s, i) for (i, s) in enumerate(strg[1:][::-1])], len(strg)-1)

    def translate(self, n=None, mapping=None):
        if mapping is None:
            paulis = [P(p.kind, p.qubit + n) for p in self.paulis]
            return PauliOperator(paulis, self.nb_qubits + n)

        paulis = [P(p.kind, mapping[p.qubit] if p.qubit in mapping else p.qubit) for p in self.paulis if p.kind != "I"]
        return PauliOperator(paulis, self.nb_qubits)

    def extend(self, n):
        self.paulis += [P("I", i) for i in range(self.nb_qubits, self.nb_qubits + n)]
        self.nb_qubits += n
        self.__support = None

    def __mul__(self, other):
        new_paulis = []
        for (p, q) in zip(self.paulis, other.paulis):
            new_paulis.append(p * q)

        if "last_measure_time" not in self.__dict__:
            return PauliOperator(new_paulis, self.nb_qubits)
        else:
            foo = PauliOperator(new_paulis, self.nb_qubits)
            foo.last_measure_time = copy(self.last_measure_time)
            return foo

    def to_simplectic_array(self):
        return np.array(list(chain((1 if p.kind in {"Z", "Y"} else 0 for p in self.paulis),
                                   (1 if p.kind in {"Y", "X"} else 0 for p in self.paulis))))

    def to_array(self):
        array = np.zeros((1, self.nb_qubits))
        for (i, pauli) in enumerate(self.paulis):
            if pauli.kind in ("X", "Z"):
                array[0, i] = 1
            if pauli.kind in ("Y",):
                array[0, i] = np.nan

        return array

    def to_symplectic_vector(self):
        return np.array(list(chain((int(pauli.kind in ("X", "Y")) for pauli in self.paulis), (int(pauli.kind in ("Z", "Y")) for pauli in self.paulis))))

    def simplification_potential(self, other):
        return sum(COST[p.kind+q.kind] for p, q in zip(self.paulis, other.paulis))

    @property
    def support(self):
        # if self.__support is None:
        #     self.__support = sorted(p.qubit for p in self.paulis if p.kind != "I")
        return sorted(p.qubit for p in self.paulis if p.kind != "I")

    @property
    def support_as_set(self):
        return set(self.support)

    @property
    def isX(self):
        """
        Returns True when the non-trivial paulis are all Xs.
        """
        return all(p.kind in ("I", "X") for p in self.paulis)

    @property
    def isZ(self):
        """
        Returns True when the non-trivial paulis are all Zs.
        """
        return all(p.kind in ("I", "Z") for p in self.paulis)

    def strip(self, kind):
        """
        Remove either X or Z part of the operator
        """
        for p in self.paulis:
            if p.kind == kind:
                p.kind = "I"
            elif p.kind == "Y":
                p.kind = "X" if kind == "Z" else "Z"
        self.__support = None

    def measure(self, measure_count):
        try:
            foo = next(measure_count)
            if self.last_measure_time[-1] is None or self.last_measure_time[-1] < foo:
                self.last_measure_time.append(foo)
                return self.last_measure_time[-2]
            if self.last_measure_time[-1] >= foo:
                raise RuntimeError("Measurement protocol not started properly")
        except AttributeError:
            raise RuntimeError("Measurement protocol not started properly")


class Stabilizer(PauliOperator):

    def __init__(self, paulis=None, nb_qubits=1, order=None):
        super().__init__(paulis, nb_qubits)
        self.order = [p.qubit for p in paulis if p.kind != "I"] if order is None else order

    def apply_CNOT(self, control, target):
        raise NotImplementedError("CNOT can't be applied directly to Stabilizers, use apply_CNOT_with_caution")

    def apply_CNOT_with_caution(self, control, target):
        """
        Changing support is not defined in general for Stabilizer as ordering
        might not be preserved. This method apply the CNOT on the support, but
        let the user handle the change on the order.
        """
        old_support = self.support
        PauliOperator.apply_CNOT(self, control, target)
        return old_support != self.support

    def apply_CZ_with_caution(self, control, target):
        """
        Changing support is not defined in general for Stabilizer as ordering
        might not be preserved. This method apply the CZ on the support, but
        let the user handle the change on the order.
        """
        old_support = self.support
        PauliOperator.apply_CZ(self, control, target)
        return old_support != self.support

    def apply_SWAP(self, control, target):
        PauliOperator.apply_SWAP(self, control, target)
        indc = None
        indt = None
        if control in self.order:
            indc = self.order.index(control)
        if target in self.order:
            indt = self.order.index(target)

        if indc is not None:
            self.order[indc] = target
        if indt is not None:
            self.order[indt] = control

    def __repr__(self):
        return f"Stabilizer({[self.paulis[qb] for qb in self.order if self.paulis[qb].kind != 'I']}, {self.nb_qubits})"

    def __eq__(self, other):
        if not isinstance(other, Stabilizer):
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

    def __hash__(self):
        if self.order:
            mini = self.order.index(min(self.order))
        else:
            mini = 0
        return hash(str(self) + "/".join(str(qb) for qb in self.order[mini:] + self.order[:mini]))

    def multiply_with_caution(self, other):
        for (p, q) in zip(self.paulis, other.paulis):
            p *= q
        self.__support = None

    def copy(self):
        return Stabilizer(paulis=[p.copy() for p in self.paulis],
                          nb_qubits=self.nb_qubits,
                          order=list(self.order))

    def translate(self, n=None, mapping=None, m=0):
        if mapping is None:
            paulis = [P(self.paulis[i].kind, self.paulis[i].qubit + n) for i in self.order]
            return Stabilizer(paulis, self.nb_qubits + n - m, order=[i+n for i in self.order])

        paulis = [P(self.paulis[i].kind, mapping[i] if i in mapping else i) for i in self.order]
        return Stabilizer(paulis, self.nb_qubits - m, order=[mapping[i] if i in mapping else i for i in self.order])


    def simplify(self, available):
        flag = True
        while flag:
            flag = False
            for s in available:
                if self.simplification_potential(s) < 0:
                    self.multiply_with_caution(s)
                    flag = True


def support_to_PauliOperator(support, kind, nb_qubits):
    """
    Creates a PauliOperator with the same Pauli on given support

    Parameters:
    support: the support of the PauliOperator
    kind: "X" | "Y" | "Z" -- the kind of Pauli wanted
    nb_qubits: maximal index of a qubit in the context

    Returns:
    PauliOperator of the given kind on the given support
    """
    func = X if kind == "X" else Z if kind == "Z" else None
    return PauliOperator([func(qb) for qb in support], nb_qubits)


def support_to_Stabilizer(support, kind, nb_qubits):
    """
    Creates a PauliOperator with the same Pauli on given support

    Parameters:
    support: the support of the PauliOperator
    kind: "X" | "Y" | "Z" -- the kind of Pauli wanted
    nb_qubits: maximal index of a qubit in the context

    Returns:
    PauliOperator of the given kind on the given support
    """
    func = X if kind == "X" else Z if kind == "Z" else None
    return Stabilizer([func(qb) for qb in support], nb_qubits)
