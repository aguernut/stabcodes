"""

"""

from stabcodes.pauli import PauliOperator
from itertools import count


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
        self.codes = list(codes)
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
            if detector_decoration is None:
                detector_decoration = "0"
        for code in codes:
            for s in code._stabilizers:
                s.measure(self._measure_clock)
                self._circuit += "MPP({meas_noise}) " + "*".join(f"{s[i].kind + str(s[i].qubit)}" for i in s.support) + "\n"
                if project is None:
                    self._circuit += "DETECTOR" + (f"({detector_decoration}) " if detector_decoration else " ") + f"rec[-1] rec[{s.last_measure_time[-2] - s.last_measure_time[-1] - 1}]\n"
                elif project == "Z" and s.isZ or project == "X" and s.isX:
                    self._circuit += "DETECTOR" + (f"({detector_decoration}) " if detector_decoration else " ") + "rec[-1]\n"

    def stim_measure_changing_support_text(self):
        raise NotImplementedError

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
        self._circuit += f"DEPOLARIZE2({rate}) " + " ".join(str(qb[0], qb[1]) for qb in supports) + "\n"

    def stim_x_error_text(self):
        raise NotImplementedError


if __name__ == "__main__":
    from stabcodes.stabilizer_code import SurfaceCode
    import stim

    c = SurfaceCode.toric_code(3, 3)
    exp = StimExperiment()
    meas_noise = Variable("meas_noise")
    pauli_noise = Variable("pauli_noise")
    exp.add_variables(meas_noise)
    exp.add_variables(pauli_noise)
    exp.startup(c, init_bases="Z")
    exp.measure_refined_phenom(c, meas_noise=meas_noise, project="Z")

    for i, log in enumerate(c.logical_operators["Z"]):
        exp.observable_measurement(i, log, meas_noise)

    for _ in range(3):
        exp.measure_refined_phenom(c, meas_noise=meas_noise)
        exp.depolarize1(pauli_noise)

    exp.destructive_measurement("Z")
    for i, log in enumerate(c.logical_operators["Z"]):
        exp.reconstruct_observable(i, log)

    with open("exp.circ", "w") as f:
        f.write(exp._circuit)

    print(stim.Circuit(exp._circuit.format(meas_noise=0.01, pauli_noise=0.1)))
