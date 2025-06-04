from stabcodes.stabilizer_code import SurfaceCode
from stabcodes.stim_experiment import StimExperiment, Variable
from stabcodes.visualization import dump_to_csv, plot_error_rate
import qiskit
import sinter


def SurfaceTransversalH(distance):
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

    exp.apply_gate("H", code.qubits)
    qis = qiskit.QuantumCircuit(len(code.qubits))
    for i in range(len(code.qubits)):
        qis.h(i)

    code.stabilizers.apply_circuit_to_stabs(qis)
    code.logical_operators.apply_circuit_to_stabs(qis)

    for _ in range(distance // 2):
        exp.measure_refined_phenom(code, meas_noise=noise)
        exp.depolarize1(noise)

    exp.destructive_measurement("X")
    exp.reconstruct_stabilizers()
    for i, log in enumerate(code.logical_operators["Z"]):
        exp.reconstruct_observable(i, log)

    return exp


if __name__ == "__main__":

    tasks = []

    for distance in range(3, 8, 2):
        exp = SurfaceTransversalH(distance)
        t, _ = exp.get_task(d=[distance],
                            noise=[0.035 * ((0.045 / 0.035)**(i / 20)) for i in range(21)])
        tasks.extend(t)
    code_stats = sinter.collect(
        num_workers=11,
        tasks=tasks,
        decoders=["pymatching"],
        max_shots=100_000,
        print_progress=True,
        # separated = True
    )

    namefile = "result_transHadamard_" + unique_name
    dump_to_csv(code_stats, namefile)

    plot_error_rate(namefile)
