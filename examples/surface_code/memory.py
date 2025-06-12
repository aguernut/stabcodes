from stabcodes.stabilizer_code import SurfaceCode
from stabcodes.stim_experiment import StimExperiment, Variable
from stabcodes.visualization import dump_to_csv, plot_error_rate, unique_name
from stabcodes.paulis import PauliOperator
import sinter


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


def SurfaceMemoryFidelity(distance):
    code = SurfaceCode.toric_code(distance, distance)
    exp = StimExperiment()
    noise = Variable("noise")
    exp.add_variables(noise)
    fid1, = exp.startup(code, init_bases="Z", fidelity=1)
    exp.measure_refined_phenom(code, meas_noise=0.0, project="Z")
    exp.apply_gate("H", (fid1,))
    exp.apply_gate("CNOT", ((fid1, qb) for qb in code.qubits))

    log_copy = [log.copy() for log in code.logical_operators]

    for log in log_copy:
        for qb in log.support:
            log.apply_CNOT(fid1, qb)

    # log_copy.append(PauliOperator.from_support(exp.qubits, "X"))

    for i, log in enumerate(log_copy):
        exp.observable_measurement(i, log_copy, 0.0)

    for _ in range(distance):
        exp.measure_refined_phenom(code, meas_noise=noise, project="Z")
        exp.depolarize1(noise)

    exp.destructive_measurement_fidelity("Z", fid1)
    exp.destructive_measurement_fidelity("X", fid1)


if __name__ == "__main__":

    tasks = []

    for distance in range(3, 12, 2):
        exp = SurfaceMemory(distance)
        t, _ = exp.get_task(d=[distance],
                            noise=[0.035 * ((0.045 / 0.035)**(i / 20)) for i in range(21)])
        tasks.extend(t)
    code_stats = sinter.collect(
        num_workers=11,
        tasks=tasks,
        decoders=["pymatching"],
        max_shots=1_000,
        print_progress=True,
        # separated = True
    )

    namefile = "result_memory_" + unique_name()
    dump_to_csv(code_stats, namefile)

    plot_error_rate(namefile)
