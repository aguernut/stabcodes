from stabcodes.stabilizer_code import SurfaceCode
from stabcodes.stim_experiment import StimExperiment, Variable
from stabcodes.visualization import dump_to_csv, plot_error_rate, unique_name
from stabcodes.pauli import PauliOperator
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


def SurfaceMemoryBell(distance):
    code, perfect_code = SurfaceCode.toric_code(distance, distance), SurfaceCode.toric_code(distance, distance)
    exp = StimExperiment()
    noise = Variable("noise")
    exp.add_variables(noise)
    exp.startup_Bell([code], [perfect_code], init_bases="Z")

    for i, (obs1, obs2) in enumerate(zip(code.logical_operators, perfect_code.logical_operators)):
        exp.reconstruct_observable_Bell(obs1, obs2, i)

    for _ in range(distance):
        exp.measure_refined_phenom(code, meas_noise=noise)
        exp.depolarize1(noise, code.qubits)

    exp.destructive_measurement_Bell(code.qubits, perfect_code.qubits, "Z")
    exp.destructive_measurement_Bell(code.qubits, perfect_code.qubits, "X")
    exp.reconstruct_stabilizers_Bell(code, perfect_code)

    for i, (obs1, obs2) in enumerate(zip(code.logical_operators, perfect_code.logical_operators)):
        exp.reconstruct_observable_Bell(obs1, obs2, i)

    return exp


if __name__ == "__main__":

    tasks = []

    for distance in range(3, 14, 2):
        exp = SurfaceMemoryBell(distance)
        t, _ = exp.get_task(d=[distance],
                            noise=[0.02 * ((0.05 / 0.02)**(i / 10)) for i in range(11)])
        tasks.extend(t)

    code_stats = sinter.collect(
        num_workers=16,
        tasks=tasks,
        decoders=["pymatching"],
        max_shots=100_000,
        print_progress=True,
        count_observable_error_combos=True,
    )

    namefile = "build/result_memory_" + unique_name()
    dump_to_csv(code_stats, namefile)

    plot_error_rate(namefile, split=True)
    import matplotlib.pyplot as plt
    plt.show()