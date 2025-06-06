from stabcodes.stabilizer_code import SurfaceCode
from stabcodes.stim_experiment import StimExperiment, Variable
from stabcodes.visualization import dump_to_csv, plot_error_rate, unique_name
import qiskit
import sinter


def SurfaceIDT(distance):
    code = SurfaceCode.toric_code(distance, distance)
    exp = StimExperiment()
    noise = Variable("noise")
    exp.add_variables(noise)
    exp.startup(code, init_bases="Z")
    exp.measure_refined_phenom(code, meas_noise=noise, project="Z")

    for i, log in enumerate(code.logical_operators["Z"]):
        exp.observable_measurement(i, log, 0.0)

    for _ in range(distance // 2):
        exp.measure_refined_phenom(code, meas_noise=noise)
        exp.depolarize1(noise)

    c = qiskit.QuantumCircuit(len(code.qubits))
    nb_qubits = len(code.qubits)
    supports = [(i, (i + nb_qubits // 2 + distance) % (nb_qubits // 2)) for i in range(nb_qubits // 2, nb_qubits)]
    exp.apply_gate("CX", supports)
    exp.depolarize2(noise, supports)

    for qb1, qb2 in supports:
        c.cx(qb1, qb2)

    code.stabilizers.apply_circuit_to_stabs(c, True)
    code.logical_operators.apply_circuit_to_stabs(c, True)

    exp.destructive_measurement("Z")
    exp.reconstruct_stabilizers()

    for i, log in enumerate(code.logical_operators["Z"]):
        exp.reconstruct_observable(i, log)

    exp.reconstruct_observable(1, code.logical_operators["Z"][0])

    return exp


if __name__ == "__main__":

    tasks = []

    for distance in range(3, 8, 2):
        exp = SurfaceIDT(distance)
        t, _ = exp.get_task(d=[distance],
                            noise=[0.035 * ((0.045 / 0.035)**(i / 20)) for i in range(21)])
        tasks.extend(t)
    code_stats = sinter.collect(
        num_workers=7,
        tasks=tasks,
        decoders=["pymatching"],
        max_shots=100_000,
        # separated = True
    )

    namefile = "result_IDT_" + unique_name()
    dump_to_csv(code_stats, namefile)

    plot_error_rate(namefile)
