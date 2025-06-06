from stabcodes._decoding_two_step_pymatching import TwoStepPymatching
from stabcodes.stabilizer_code import SurfaceCode
from stabcodes.stim_experiment import StimExperiment, Variable
from stabcodes.visualization import dump_to_csv, plot_error_rate, unique_name
import qiskit
import sinter


def SurfaceTransversalS(distance):
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

    s_support = [i + i * distance for i in range(distance)] + [distance**2 + i * distance + (i + 1) % distance for i in range(distance)]
    CZ_support = [(i * distance + j, j * distance + i) for i in range(distance) for j in range(i + 1, distance)] + [(distance**2 + i * distance + (j + 1) % distance, distance**2 + j * distance + (i+1) % distance) for i in range(distance) for j in range(i+1, distance)]

    exp.apply_gate("S", s_support)
    exp.depolarize1(noise, s_support)
    exp.apply_gate("CZ", CZ_support)
    exp.depolarize2(noise, CZ_support)

    qis = qiskit.QuantumCircuit(len(code.qubits))
    for i in s_support:
        qis.s(i)
    for qbs in CZ_support:
        qis.cz(*qbs)

    code.stabilizers.apply_circuit_to_stabs(qis, True)
    code.logical_operators.apply_circuit_to_stabs(qis)

    codex = code.copy()
    del codex.stabilizers["Z"]
    codez = code.copy()
    del codez.stabilizers["X"]

    for _ in range(distance // 2):
        exp.measure_refined_phenom(codex, meas_noise=noise, detector_decoration="1")
        exp.measure_refined_phenom(codez, meas_noise=noise, detector_decoration="2")
        exp.depolarize1(noise)

    exp.destructive_measurement("Z")
    exp.reconstruct_stabilizers(codex, detector_decoration="1")
    exp.reconstruct_stabilizers(codez, detector_decoration="2")
    for i, log in enumerate(code.logical_operators["Z"]):
        exp.reconstruct_observable(i, log)

    return exp


if __name__ == "__main__":

    tasks = []
    custom_decoders = {}

    for distance in range(3, 12, 2):
        exp = SurfaceTransversalS(distance)
        t, decoders = exp.get_task(decoder=TwoStepPymatching, pass_circuit=True, d=[distance],
                                   noise=[0.01 * ((0.02 / 0.01)**(i / 10)) for i in range(11)])
        tasks.extend(t)
        with open("foo.circ", "w") as f:
            f.write(str(tasks[0].circuit))
        custom_decoders.update(decoders)
    
    code_stats = sinter.collect(
        num_workers=7,
        tasks=tasks,
        decoders=[],
        custom_decoders=custom_decoders,
        max_shots=100_000,
#        print_progress=True
    )
    #     t, _ = exp.get_task(d=[distance],
    #                         noise=[0.01 * ((0.02 / 0.01)**(i / 10)) for i in range(11)])
    #     tasks.extend(t)
    # code_stats = sinter.collect(
    #     num_workers=11,
    #     tasks=tasks,
    #     decoders=["pymatching"],
    #     max_shots=10_000_000,
    #     # print_progress=True,
    #     # separated = True
    # )

    namefile = "result_transHadamard_" + unique_name()
    dump_to_csv(code_stats, namefile)

    plot_error_rate(namefile)
