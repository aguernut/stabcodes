from stabcodes.stabilizer_code import SurfaceCode
from stabcodes.stim_experiment import StimExperiment, Variable
from stabcodes.visualization import dump_to_csv, plot_error_rate, unique_name
from stabcodes.tools import transpositions_from_mapping, reverse_dict
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


def SurfaceTransversalHBell(distance):
    code, perfect_code = SurfaceCode.toric_code(distance, distance), SurfaceCode.toric_code(distance, distance)
    exp = StimExperiment()
    noise = Variable("noise")
    exp.add_variables(noise)
    exp.startup_Bell([code], [perfect_code], init_bases="Z")

    exp.reconstruct_observable_Bell(code.logical_operators["X"][0],
                                    perfect_code.logical_operators["X"][0], 0)
    exp.reconstruct_observable_Bell(code.logical_operators["X"][1],
                                    perfect_code.logical_operators["X"][1], 1)
    exp.reconstruct_observable_Bell(code.logical_operators["Z"][0],
                                    perfect_code.logical_operators["Z"][0], 2)
    exp.reconstruct_observable_Bell(code.logical_operators["Z"][1],
                                    perfect_code.logical_operators["Z"][1], 3)

    for i in zip(code.logical_operators, perfect_code.logical_operators):
        print(repr(i))


    for _ in range(distance // 2):
        exp.measure_refined_phenom(code, meas_noise=noise)
        exp.depolarize1(noise, code.qubits)

    # mapping = {}
    # for i in range(distance):
    #     for j in range(distance):
    #         mapping[i * distance + (j + 1) % distance + distance**2] = i * distance + j
    #         mapping[i * distance + j] = (i * distance + j + distance**2 - distance + (i == 0) * distance**2) % (2 * distance**2)
    # print(mapping)
    # mapping.update(dict((i + len(code.qubits), j + len(code.qubits)) for (i, j) in mapping.items()))

    # transpositions = transpositions_from_mapping(reverse_dict(mapping))
    # print(transpositions)
    
    exp.apply_gate("H", code.qubits)
    exp.apply_gate("H", perfect_code.qubits)
    # exp.apply_gate("SWAP", transpositions)
    qis = qiskit.QuantumCircuit(len(exp._data_qubits))
    for i in exp._data_qubits:
        qis.h(i)

    # for qb1, qb2 in transpositions:
    #     qis.swap(qb1, qb2)

    code.stabilizers.apply_circuit_to_stabs(qis)
    code.logical_operators.apply_circuit_to_stabs(qis)
    perfect_code.stabilizers.apply_circuit_to_stabs(qis)
    perfect_code.logical_operators.apply_circuit_to_stabs(qis)

    for _ in range(distance // 2):
        exp.measure_refined_phenom(code, meas_noise=noise)
        exp.depolarize1(noise, code.qubits)

    exp.destructive_measurement_Bell(code.qubits, perfect_code.qubits, "Z")
    exp.destructive_measurement_Bell(code.qubits, perfect_code.qubits, "X")
    exp.reconstruct_stabilizers_Bell(code, perfect_code)

    for i in zip(code.logical_operators, perfect_code.logical_operators):
        print(repr(i))


    exp.reconstruct_observable_Bell(code.logical_operators["Z"][0],
                                    perfect_code.logical_operators["Z"][0], 2)
    exp.reconstruct_observable_Bell(code.logical_operators["Z"][1],
                                    perfect_code.logical_operators["Z"][1], 3)
    exp.reconstruct_observable_Bell(code.logical_operators["X"][0],
                                    perfect_code.logical_operators["X"][0], 0)
    exp.reconstruct_observable_Bell(code.logical_operators["X"][1],
                                    perfect_code.logical_operators["X"][1], 1)

    return exp


if __name__ == "__main__":

    tasks = []

    for distance in range(3, 8, 2):
        exp = SurfaceTransversalHBell(distance)
        t, _ = exp.get_task(d=[distance],
                            noise=[0.03 * ((0.04 / 0.03)**(i / 20)) for i in range(21)])
        tasks.extend(t)
    with open("foo.circ", "w") as f:
        f.write(str(tasks[0].circuit))
    code_stats = sinter.collect(
        num_workers=16,
        tasks=tasks,
        decoders=["pymatching"],
        max_shots=1_000_000,
        print_progress=True,
        count_observable_error_combos=True
    )

    namefile = "build/result_transHadamard_"  # + unique_name()
    dump_to_csv(code_stats, namefile)

    plot_error_rate(namefile)
    import matplotlib.pyplot as plt
    plt.show()
