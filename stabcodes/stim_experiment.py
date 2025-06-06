"""

"""

from stabcodes.pauli import PauliOperator
from itertools import count, product
import sinter
import stim
import uuid


class MeasureClock:

    def __init__(self):
        self._clock = count()
        self._current = 0

    def __next__(self):
        self._current = next(self._clock)
        return self._current

    @property
    def current(self):
        return self._current


class Variable:

    def __init__(self, name):
        self._name = name

    def __str__(self):
        return f"{{{self._name}}}"

    def __repr__(self):
        return f"Variable({self._name})"


class StimExperiment:

    def __init__(self):
        self._circuit = ""
        self._variables = {}
        self._measure_clock = None
        self._codes = None
        self._data_qubits = None
        self._physical_measurement = None

    def startup(self, *codes, init_bases="Z"):
        if len(init_bases) != len(codes):
            raise ValueError("An initialization basis for each code must be specified.")

        self._measure_clock = MeasureClock()
        self._physical_measurement = {}
        self._codes = list(codes)
        N = sum(c.num_qubits for c in codes)
        self._data_qubits = range(N)
        qb_shift = 0
        for c, init_basis in zip(codes, init_bases):
            c.shift_qubits(qb_shift, N)
            qb_shift += len(c.qubits)
            c._stabilizers.reset()
            c._logical_operators.reset()
            self._circuit += f"R{init_basis} " + " ".join(str(i) for i in self._data_qubits) + "\n"

    def add_variables(self, newvar, value=None):
        return self._variables.setdefault(newvar, value)

    def spacecode_projection(self):
        raise NotImplementedError

    def measure_refined_phenom(self, *codes, meas_noise=0.0, project=None, detector_decoration=None):
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

    def stim_measure_changing_support_text(self):
        raise NotImplementedError

    def buddy_measurement(self, code0, code1, mapping, mode, meas_noise="{meas_noise}", decoding_step0=1, decoding_step1=1):
        """
        Patch must be in code1
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

    def observable_measurement(self, index: int, operator: PauliOperator, obs_meas_noise=0.0):
        operator.measure(self._measure_clock)
        self._circuit += f"MPP({obs_meas_noise}) " + "*".join(operator[i].kind + str(operator[i].qubit) for i in operator.support) + "\n"
        self._circuit += f"OBSERVABLE_INCLUDE({index}) rec[-1]\n"

    def destructive_measurement(self, basis, to_measure=None):
        if to_measure is None:
            to_measure = self._data_qubits
        self._circuit += f"M{basis} " + " ".join(str(i) for i in to_measure) + "\n"
        for i in to_measure:
            self._physical_measurement[i] = (basis, next(self._measure_clock))

    def reconstruct_stabilizers(self, *codes, detector_decoration=None):
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

    def reconstruct_observable(self, index, operator):
        current = self._measure_clock.current
        for qb in operator.support:
            (basis, meas_time) = self._physical_measurement.get(qb, (None, None))
            if basis is None:
                raise RuntimeError(f"Qubit {qb} as not been physically measured yet.")
            if basis != operator[qb].kind:
                raise RuntimeError(f"Qubit {qb} was measured in basis {basis}, not {operator[qb].kind}.")

        self._circuit += f"OBSERVABLE_INCLUDE({index}) " + " ".join(f"rec[{self._physical_measurement[qb][1] - current - 1}]" for qb in operator.support) + "\n"

    def stim_realistic_logical_measure_text(self):
        raise NotImplementedError

    def stim_realistic_Bell_logical_measure_text(self):
        raise NotImplementedError

    def depolarize1(self, rate, support=None):
        if support is None:
            support = self._data_qubits

        self._circuit += f"DEPOLARIZE1({rate}) " + " ".join(str(i) for i in support) + "\n"

    def depolarize2(self, rate, supports):
        self._circuit += f"DEPOLARIZE2({rate}) " + " ".join(f"{qb[0]} {qb[1]}" for qb in supports) + "\n"

    def stim_x_error_text(self):
        raise NotImplementedError

    def get_task(self, decoder=None, pass_circuit=False, decoder_options=dict(), **values):
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

    def apply_gate(self, gate, support):
        if gate in {"CX", "CZ", "CY", "SWAP", "CNOT"}:
            self._circuit += f"{gate} " + " ".join(" ".join(str(qb) for qb in pairs) for pairs in support) + "\n"
        else:
            self._circuit += f"{gate} " + " ".join(str(qb) for qb in support) + "\n"


if __name__ == "__main__":
    from stabcodes.stabilizer_code import SurfaceCode
    import uuid
    from stabcodes.visualization import dump_to_csv, plot_error_rate
    from datetime import date

    def SurfaceMemory(distance):
        code = SurfaceCode.toric_code(distance, distance)
        exp = StimExperiment()
        noise = Variable("noise")
        exp.add_variables(noise)
        exp.startup(code, init_bases="Z")
        exp.measure_refined_phenom(code, meas_noise=noise, project="Z")

        for i, log in enumerate(code.logical_operators["Z"]):
            exp.observable_measurement(i, log, 0.0)

        for _ in range(distance):
            exp.measure_refined_phenom(code, meas_noise=noise)
            exp.depolarize1(noise)

        exp.destructive_measurement("Z")
        exp.reconstruct_stabilizers()
        for i, log in enumerate(code.logical_operators["Z"]):
            exp.reconstruct_observable(i, log)

        return exp

    tasks = []

    for distance in range(3, 12, 2):
        exp = SurfaceMemory(distance)
        tasks.extend(exp.get_task(d=[distance],
                                  noise=[0.035 * ((0.045 / 0.035)**(i / 20)) for i in range(21)]))
    custom_decoders = dict(task.decoder for task in tasks)
    del custom_decoders[None]
    print(custom_decoders)
    for task in tasks:
        task.decoder = task.decoder[0]
    code_stats = sinter.collect(
        num_workers=11,
        tasks=tasks,
        decoders=["pymatching"],
        max_shots=1_000,
        print_progress=True,
        # separated = True
    )

    namefile = "result_memory_" + date.today().isoformat() + str(uuid.uuid1())
    dump_to_csv(code_stats, namefile)

    plot_error_rate(namefile)
