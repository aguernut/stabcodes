"""
Driver for running a stim experiment for a stabilizer code.
"""

from stabcodes.pauli import PauliOperator, Pauli
from stabcodes.stabilizer_code import StabilizerCode
from typing import Union, Iterable, Optional
from itertools import count, product
import sinter
import stim
import uuid


class Variable:
    """
    Placeholder for numerical values to be decided later.
    This relies on f-string and the format function.
    """
    def __init__(self, name):
        """
        Instantiate a new variable.

        Notes
        -----
        Variables are identified by their name (similarly to a programming variable).
        Creating two Variables with the same name will create two independent objects,
        but they will be indistinguishable in their standard use.

        Parameters
        ----------
        name: str
            A name that will be used as an f-string identifier.

        """
        self._name = name

    def __str__(self):
        return f"{{{self._name}}}"

    def __repr__(self):
        return f"Variable({self._name})"


class StimExperiment:
    """
    A high-level class handling a stim experiment.

    Notes
    -----
    Some of the advanced method of this class where created for some
    surface code experiments. You will probably have to create your
    own methods if you want to simulate other codes, especially if
    you need specific annotations for your targeted decoder.

    Notes
    -----
    This class does not allow for circuit-level measurement of stabilizers.
    You will have to devise a method doing that and call it instead of
    :meth:`measure_refined_phenom`.

    Attributes
    ----------
    circuit: string
        F-string representing the current state of the underlying stim circuit.

    """

    def __init__(self):
        self._circuit = ""
        self._variables = {}
        self._measure_clock = None
        self._codes = None
        self._data_qubits = None
        self._physical_measurement = None
        self._Bell_physical_measurement = None

    @property
    def circuit(self):
        """Current state of the underlying stim circuit."""
        return self._circuit

    def startup(self, *codes, init_bases="Z"):
        """
        Initialization of the experiment.

        Notes
        -----
        Any tracked qubit must be part of a code.

        Parameters
        ----------
        codes: StabilizerCode
            Any number of stabilizer codes that takes part in the experiment.

        init_bases: str
            Basis in which the qubits of each code will initialized.
            One basis per code must be specified, inside a common string (e.g. "ZZX").

        """
        if len(init_bases) != len(codes):
            raise ValueError("An initialization basis for each code must be specified.")

        self._measure_clock = MeasureClock()
        self._physical_measurement = {}
        self._Bell_physical_measurement = {}
        self._codes = list(codes)
        N = sum(c.num_qubits for c in codes)
        self._data_qubits = range(N)
        qb_shift = 0
        for c, init_basis in zip(codes, init_bases):
            c._measure_count = self._measure_clock
            c.shift_qubits(qb_shift, N)
            qb_shift += len(c.qubits)
            c._stabilizers.reset()
            c._logical_operators.reset()
            self._circuit += f"R{init_basis} " + " ".join(str(i) for i in c.qubits) + "\n"

    def add_variables(self, newvar, value=None) -> object:
        """
        Defines a new variable for the experiment.

        Parameters
        ----------
        newvar: Variable
            Variable to add to the context.
        value: object, optional
            Typically a default numerical value taken by the variable.

        """
        return self._variables.setdefault(newvar, value)

    def measure_refined_phenom(self, *codes, meas_noise=0.0, project=None, detector_decoration=None):
        """
        Adds a round of stabilizer measurement for the specified codes.

        Parameters
        ----------
        codes: StabilizerCode
            Stabilizer codes whose stabilizers will be measured.
        meas_noise: Union[Variable, float]
            Measurement error rate, defaults to 0.0.
        project: str, optional
            If specified, the stabilizer whose kind ("X" or "Z") matches the parameter will be the only
            one added to a detector. Typically, a first round of stabilizer measurement with qubit initialized
            in the "Z" basis will use project="Z", and subsequent rounds will use None.
        detector_decoration: str, optional
            Some optional decoration added to the included detectors to help the decoder.

        """
        if not codes:
            codes = self._codes

        for code in codes:
            for s in code._stabilizers:
                s.measure(self._measure_clock)
                self._circuit += f"MPP({meas_noise}) " + "*".join(f"{s[i].kind + str(s[i].qubit)}" for i in s.support) + "\n"
                if project is None:
                    self._circuit += "DETECTOR" + (f"({detector_decoration}) " if detector_decoration else " ") + f"rec[-1] rec[{s.last_measure_time[-2] - s.last_measure_time[-1] - 1}]\n"
                elif project == "Z" and s.isZ or project == "X" and s.isX:
                    self._circuit += "DETECTOR" + (f"({detector_decoration}) " if detector_decoration else " ") + "rec[-1]\n"

    def buddy_measurement(self, code0: StabilizerCode, code1: StabilizerCode,
                          mapping: dict[int, int], mode: dict[int, dict[str, str]],
                          meas_noise: Union[Variable, float] = "{meas_noise}",
                          decoding_step0: int = 1, decoding_step1: int = 1):
        """
        Intricate function ensuring that the detectors are properly set after a round of entangling transversal gates.
        Each stabilizer from one code will be associated to a "buddy" stabilizer from the other code with which it has
        been entangled. If a patch is involved, it must be set as a "code1" parameter.

        Detectors are annotated to accomodate the two-step MWPM decoder.

        Notes
        -----
        This function is only useful for CSS codes.

        Parameters
        ----------
        code0: StabilizerCode
            Stabilizer code of reference
        code1: StabilizerCode
            Subordinate stabilizer code. It is advised to put any patch with boundaries as this parameter
        mapping: dict[int, int]
            An injective mapping of the qubits from code0 to code1.
        mode: dict[int, dict[str, str]]
            Small dictionary describing which kind of stabilizers are grown from which code.
            For example, if the entangling gates are CNOT gates, the mode is the following:
            {0: {"X": "X", "Z": ""}, 1: {"X": "", "Z": "Z"}}
        meas_noise: Union[Variable, float]
            Measurement error rate, defaults to 0.0.
        decoding_step0: int
            Decoration for the detectors associated with code0.
        decoding_step1: int
            Decoration for the detectors associated with code1.
        """
        # Buddies computation

        # Buddies computation for code0
        buddies0 = {}
        for basis in mode[0]:
            buddies0[basis] = {}
            for stab in code0.iter_stabilizers(basis):
                buddies0[basis][stab] = []
                for buddy_basis in mode[0][basis]:
                    for buddy in code1.iter_stabilizers(buddy_basis):
                        if set(mapping[i] for i in stab.support_as_set) >= buddy.support_as_set:
                            buddies0[basis][stab].append(buddy)
                            break

        # Buddies computation for code1
        buddies1 = {}
        for basis in mode[1]:
            buddies1[basis] = {}
            for stab in code1.iter_stabilizers(basis):
                buddies1[basis][stab] = []
                for buddy_basis in mode[1][basis]:
                    for buddy in code0.iter_stabilizers(buddy_basis):
                        if set(mapping[i] for i in buddy.support_as_set) >= stab.support_as_set:
                            buddies1[basis][stab].append(buddy)
                            break

        # Measurements
        deb = None

        # Measurement of code0
        for basis in mode[0]:
            for stab in code0.iter_stabilizers(basis):
                stab.measure(self._measure_clock)
                if deb is None:
                    deb = stab.last_measure_time[-1]
                self._circuit += f"MPP({meas_noise}) " + "*".join(p.kind + str(p.qubit) for p in stab.paulis if p.kind != "I") + "\n"
                buddies = buddies0[basis].get(stab)

                if not buddies:
                    self._circuit += f"DETECTOR({decoding_step0}) rec[-1] rec[{stab.last_measure_time[-2] - stab.last_measure_time[-1] - 1}]\n"
                else:
                    for buddy in buddies:
                        self._circuit += f"DETECTOR({decoding_step0}) rec[-1] rec[{stab.last_measure_time[-2] - stab.last_measure_time[-1] - 1}] rec[{buddy.last_measure_time[-1] - stab.last_measure_time[-1] - 1}]\n"

        # Measurement of code1
        for basis in mode[1]:
            for stab in code1.iter_stabilizers(basis):
                stab.measure(self._measure_clock)
                self._circuit += f"MPP({meas_noise}) " + "*".join(p.kind + str(p.qubit) for p in stab.paulis if p.kind != "I") + "\n"
                buddies = buddies1[basis].get(stab)

                if not buddies:
                    self._circuit += f"DETECTOR({decoding_step1}) rec[-1] rec[{stab.last_measure_time[-2] - stab.last_measure_time[-1] - 1}]\n"
                else:
                    for buddy in buddies:
                        self._circuit += f"DETECTOR({decoding_step1}) rec[-1] rec[{stab.last_measure_time[-2] - stab.last_measure_time[-1] - 1}] rec[{(v if (v := buddy.last_measure_time[-1]) < deb else buddy.last_measure_time[-2]) - stab.last_measure_time[-1] - 1}]\n"

    def observable_measurement(self, index: int, operator: PauliOperator,
                               obs_meas_noise: Union[Variable, float] = 0.0):
        """
        Measures given :class:`PauliOperator` and adds it to an observable.

        Parameters
        ----------
        index: int
            Index of the observable to which the measurement should be added.
        operator: PauliOperator
            Operator that should be measured.
        obs_meas_noise: Union[Variable, float]
            Measurement error rate of the operator, defaults to 0.0.

        """
        operator.measure(self._measure_clock)
        self._circuit += f"MPP({obs_meas_noise}) " + "*".join(operator[i].kind + str(operator[i].qubit) for i in operator.support) + "\n"
        self._circuit += f"OBSERVABLE_INCLUDE({index}) rec[-1]\n"

    def destructive_measurement(self, basis: str, to_measure: Optional[Iterable[int]] = None):
        """
        Performs a physical measurement of all the (specified) qubits.

        Parameters
        ----------
        basis: str
            Basis in which the physical qubits are measured. Can be "X", "Y" or "Z".
        to_measure: Iterable[int], optional
            Collection of qubits to be measured. If None is provided, all the qubits in the current context will be measured.

        """
        if to_measure is None:
            to_measure = self._data_qubits
        self._circuit += f"M{basis} " + " ".join(str(i) for i in to_measure) + "\n"
        for i in to_measure:
            self._physical_measurement[i] = (basis, next(self._measure_clock))

    def reconstruct_stabilizers(self, *codes: StabilizerCode, detector_decoration: Optional[int] = None):
        """
        Reconstructs stabilizer measurements from the last physical measurement of the qubits.

        Parameters
        ----------
        codes: StabilizerCode
            Codes whose stabilizers will be reconstructed. Only the stabilizer whose expression matches
            the last physical measurement will be reconstructed, others will be silently ignored.
        detector_decoration: int, optional
            Optional decoration for the added detectors. Can be useful for the decoder.

        Notes
        -----
        The last available measurement of the correct kind will be used, even if some other operations
        happened in-between. Make sure to properly perform a destructive measurement before.

        """
        current = self._measure_clock.current
        if not codes:
            codes = self._codes

        for c in codes:
            for s in c.stabilizers:
                for qb in s.support:
                    (basis, meas_time) = self._physical_measurement.get(qb, (None, None))
                    if basis is None or basis != s[qb].kind:
                        break
                else:
                    self._circuit += "DETECTOR" + (f"({detector_decoration}) " if detector_decoration else " ") + " ".join(f"rec[{self._physical_measurement[qb][1] - current - 1}]" for qb in s.support) + f" rec[{s.last_measure_time[-1] - current - 1}]\n"

    def reconstruct_observable(self, index: int, operator: PauliOperator):
        """
        Reconstructs observable measurements from the last physical measurement of the qubits.

        Parameters
        ----------
        index: int
            Index of the observable to which the reconstructed value should be added.
        operator: PauliOperator
            Operator whose measurement will be added to the observable.

        Raises
        ------
        RuntimeError
            Raised when a requirement measurement has not been done, or was performed in an incorrect basis.

        Notes
        -----
        The last available measurement of the correct kind will be used, even if some other operations
        happened in-between. Make sure to properly perform a destructive measurement before.

        """
        current = self._measure_clock.current
        for qb in operator.support:
            (basis, meas_time) = self._physical_measurement.get(qb, (None, None))
            if basis is None:
                raise RuntimeError(f"Qubit {qb} as not been physically measured yet.")
            if basis != operator[qb].kind:
                raise RuntimeError(f"Qubit {qb} was measured in basis {basis}, not {operator[qb].kind}.")

        self._circuit += f"OBSERVABLE_INCLUDE({index}) " + " ".join(f"rec[{self._physical_measurement[qb][1] - current - 1}]" for qb in operator.support) + "\n"

    def destructive_measurement_Bell(self, to_measure1: list[int], to_measure2: list[int], basis: str, basis2: Optional[str] = None):
        """
        Performs a physical Bell measurement of some physical qubits.

        Parameters
        ----------
        to_measure1: list[int]
            First half of the qubits to measure.
        to_measure1: list[int]
            Second half of the qubits to measure.
        basis: str
            Basis in which the qubit of the first half are measured.
        basis2: str, optional
            Basis in which the qubit of the second half are measured. If None, the same basis as the first half is used.
        """
        assert len(to_measure1) == len(to_measure2)

        if basis2 is None:
            basis2 = basis

        self._circuit += "MPP " + " ".join(basis + str(i) + "*" + basis2 + str(j) for i, j in zip(to_measure1, to_measure2)) + "\n"
        for i, j in zip(to_measure1, to_measure2):
            self._Bell_physical_measurement[Pauli(basis, i)] = (Pauli(basis2, j), next(self._measure_clock))

    def reconstruct_stabilizers_Bell(self, code1: StabilizerCode, code2: StabilizerCode, detector_decoration: Optional[int] = None):
        """
        Reconstructs stabilizer Bell measurements from the last physical Bell measurement of the qubits.

        Parameters
        ----------
        code1: StabilizerCode
            First code from which to take the stabilizers.
        code2: StabilizerCode
            Second code from which to take the stabilizers.
        detector_decoration: int, optional
            Optional decoration for the added detectors. Can be useful for the decoder.

        """
        current = self._measure_clock.current

        for s1, s2 in zip(code1.stabilizers, code2.stabilizers):
            supp1 = {s1[qb] for qb in s1.support}
            supp2 = {s2[qb] for qb in s2.support}
            for pauli in supp1:
                if self._Bell_physical_measurement[pauli][0] not in supp2:
                    raise RuntimeError(f"No Bell measurement with {pauli} as primary target have secondary target in {supp2}.")
            self._circuit += "DETECTOR" + (f"({detector_decoration}) " if detector_decoration else " ") + " ".join(f"rec[{self._Bell_physical_measurement[pauli][1] - current - 1}]" for pauli in supp1) + f" rec[{s1.last_measure_time[-1] - current - 1}]" + f" rec[{s2.last_measure_time[-1] - current - 1}]\n"

    def reconstruct_observable_Bell(self, observable1: PauliOperator, observable2: PauliOperator, index: int):
        """
        Reconstructs observable Bell measurements from the last physical Bell measurement of the qubits.

        Parameters
        ----------
        observable1: PauliOperator
            First operator whose measurement will be added to the observable.
        observable2: PauliOperator
            Second operator whose measurement will be added to the observable.
        index: int
            Index of the observable to which the reconstructed measurement will be added.

        Raises
        ------
        RuntimeError
            Raised when a requirement measurement has not been done, or was performed in an incorrect basis.

        Notes
        -----
        The last available measurement of the correct kind will be used, even if some other operations
        happened in-between. Make sure to properly perform a destructive measurement before.

        """
        current = self._measure_clock.current
        supp1 = {observable1[qb] for qb in observable1.support}
        supp2 = {observable2[qb] for qb in observable2.support}
        for pauli in supp1:
            if self._Bell_physical_measurement[pauli][0] not in supp2:
                raise RuntimeError(f"No Bell measurement happened with {pauli} as primary target")
        self._circuit += f"OBSERVABLE_INCLUDE({index}) " + " ".join(f"rec[{self._Bell_physical_measurement[pauli][1] - current - 1}]" for pauli in supp1) + "\n"

    def depolarize1(self, rate: Union[Variable, float], support: Optional[list[int]] = None):
        """
        Transversal depolarizing noise over the individual physical qubits.

        Parameters
        ----------
        rate: Union[Variable, float]
            Depolarizing error rate.
        support: list[int], optional
            Qubits that will undergo the depolarizing channel.
            If None is provided, all the qubits in context will be depolarized.

        """
        if support is None:
            support = self._data_qubits

        self._circuit += f"DEPOLARIZE1({rate}) " + " ".join(str(i) for i in support) + "\n"

    def depolarize2(self, rate: Union[Variable, float], supports: list[tuple[int, int]]):
        """
        Correlated depolarizing noise over pairs of physical qubits.

        Parameters
        ----------
        rate: Union[Variable, float]
            Depolarizing error rate.
        supports: list[tuple[int, int]]
            Pairs across which the correlated depolarizing noise occurs.

        """
        self._circuit += f"DEPOLARIZE2({rate}) " + " ".join(f"{qb[0]} {qb[1]}" for qb in supports) + "\n"

    def get_task(self, decoder: Optional[sinter.Decoder] = None, pass_circuit: bool = False, **values) -> (list[sinter.Task], dict[str, sinter.Decoder]):
        """
        Create a list of tasks from the different values for the variables of this circuit.

        Parameters
        ----------
        decoder: sinter.Decoder, optional
            Decoder to be used to decode this task. It is an optional argument as built-in decoders can also be passed
            when calling sinter.collect().
        pass_circuit: bool
            Whether the circuit must be passed to the instantiated decoder.
        values: dict[str, list[object]]
            Name and list of possible values for the declared variables used by this experiment.
            A cartesian product of the different values is considered for creating all the possible tasks.

        Returns
        -------
        list[sinter.Task]
            A list of sinter tasks that can be simulated.
        dict[str, sinter.Decoder]
            A mapping of unique decoder names to instantied decoders to give to a sinter.collect() call.

        """
        ordered_keys = values.keys()
        assert set(var._name for var in self._variables.keys()) <= set(ordered_keys), f"All declared variables must be assigned, {set(self._variables.keys()) - set(ordered_keys)} not set"

        tasks = []
        decoder_names = {}
        for vals in product(*values.values()):
            metadata = dict(zip(ordered_keys, vals))
            circuit = stim.Circuit(self._circuit.format(**metadata))
            if decoder is not None and pass_circuit:
                instantiated_decoder = decoder(circuit)
            else:
                instantiated_decoder = decoder
            decoder_name = None if decoder is None else decoder.__name__ + "_" + str(uuid.uuid4())
            decoder_names[decoder_name] = instantiated_decoder
            task = sinter.Task(decoder=decoder_name, circuit=circuit,
                               json_metadata=metadata)
            task.instantiated_decoder = instantiated_decoder
            tasks.append(task)

        return tasks, decoder_names

    def apply_gate(self, gate: str, support: Union[list[int], list[tuple[int]]]):
        """
        Applies a specific gate to a list of targeted qubits.

        Parameters
        ----------
        gate: str
            Gate to apply. Must a valid gate name understood by stim.
        support: Union[list[int], list[tuple[int]]]
            Indices of the qubits supporting the gate.

        """
        if gate in {"CX", "CZ", "CY", "SWAP", "CNOT"}:
            self._circuit += f"{gate} " + " ".join(" ".join(str(qb) for qb in pairs) for pairs in support) + "\n"
        else:
            self._circuit += f"{gate} " + " ".join(str(qb) for qb in support) + "\n"


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)
