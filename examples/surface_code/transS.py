from stabcodes._decoding_two_step_pymatching import TwoStepPymatching
from stabcodes.stabilizer_code import SurfaceCode
from stabcodes.stim_experiment import StimExperiment, Variable
from stabcodes.visualization import dump_to_csv, plot_error_rate, unique_name
from stabcodes.tools import reverse_dict
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
    CZ_support = [(i * distance + j, j * distance + i) for i in range(distance) for j in range(i + 1, distance)] + [(distance**2 + i * distance + (j + 1) % distance, distance**2 + j * distance + (i + 1) % distance) for i in range(distance) for j in range(i + 1, distance)]

    exp.apply_gate("S", s_support)
    exp.depolarize1(noise, s_support)
    exp.apply_gate("CZ", CZ_support)
    exp.depolarize2(noise, CZ_support)

    dic = dict(CZ_support)
    dic.update(reverse_dict(dic))
    dic.update([(i, i) for i in s_support])
    exp.buddy_measurement(code, code, dic, {0: {"X": "Z"}, 1: {"Z": ""}}, noise, 1, 2)
    exp.depolarize1(noise)

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


def SurfaceTransversalSBell(distance):
    code, perfect_code = SurfaceCode.toric_code(distance, distance), SurfaceCode.toric_code(distance, distance)
    exp = StimExperiment()
    noise = Variable("noise")
    exp.add_variables(noise)
    exp.startup_Bell([code], [perfect_code], init_bases="Z")

    for i, (obs1, obs2) in enumerate(zip(code.logical_operators, perfect_code.logical_operators)):
        exp.reconstruct_observable_Bell(obs1, obs2, i)


    for _ in range(distance // 2):
        exp.measure_refined_phenom(code, meas_noise=noise)
        exp.depolarize1(noise, code.qubits)

    s_support = [i + i * distance for i in range(distance)] + [distance**2 + i * distance + (i + 1) % distance for i in range(distance)]
    s_support2 = [i + len(code.qubits) for i in s_support]
    CZ_support = [(i * distance + j, j * distance + i) for i in range(distance) for j in range(i + 1, distance)] + [(distance**2 + i * distance + (j + 1) % distance, distance**2 + j * distance + (i + 1) % distance) for i in range(distance) for j in range(i + 1, distance)]
    CZ_support2 = [(i + len(code.qubits), j + len(code.qubits)) for i, j in CZ_support]

    exp.apply_gate("S", s_support)
    exp.apply_gate("S", s_support2)
    exp.depolarize1(noise, s_support)
    exp.apply_gate("CZ", CZ_support)
    exp.apply_gate("CZ", CZ_support2)
    exp.depolarize2(noise, CZ_support)

    dic = dict(CZ_support)
    dic.update(reverse_dict(dic))
    dic.update([(i, i) for i in s_support])
    exp.buddy_measurement(code, code, dic, {0: {"X": "Z"}, 1: {"Z": ""}}, noise, 1, 2)
    exp.depolarize1(noise, code.qubits)

    dic2 = dict(CZ_support2)
    dic2.update(reverse_dict(dic2))
    dic2.update([(i, i) for i in s_support2])
    exp.buddy_measurement(perfect_code, perfect_code, dic2, {0: {"X": "Z"}, 1: {"Z": ""}}, 0.0, 1, 2)

    codex = code.copy()
    del codex.stabilizers["Z"]
    codez = code.copy()
    del codez.stabilizers["X"]

    perfect_codex = perfect_code.copy()
    del perfect_codex.stabilizers["Z"]
    perfect_codez = perfect_code.copy()
    del perfect_codez.stabilizers["X"]

    for _ in range(distance // 2):
        exp.measure_refined_phenom(codex, meas_noise=noise, detector_decoration="1")
        exp.measure_refined_phenom(codez, meas_noise=noise, detector_decoration="2")
        exp.depolarize1(noise, code.qubits)

    exp.destructive_measurement_Bell(code.qubits, perfect_code.qubits, "Z")
    exp.destructive_measurement_Bell(code.qubits, perfect_code.qubits, "X")
    exp.reconstruct_stabilizers_Bell(codex, perfect_codex, detector_decoration="1")
    exp.reconstruct_stabilizers_Bell(codez, perfect_codez, detector_decoration="2")

    for i, (obs1, obs2) in enumerate(zip(code.logical_operators, perfect_code.logical_operators)):
        exp.reconstruct_observable_Bell(obs1, obs2, i)

    for i, (obs1, obs2) in enumerate(zip(code.logical_operators["Z"], perfect_code.logical_operators["Z"])):
        exp.reconstruct_observable_Bell(obs1, obs2, i)

    return exp


if __name__ == "__main__":

    tasks = []
    custom_decoders = {}

    for distance in range(3, 8, 2):
        exp = SurfaceTransversalSBell(distance)
        t, decoders = exp.get_task(decoder=TwoStepPymatching, pass_circuit=True, d=[distance],
                                   noise=[0.03 * ((0.04 / 0.03)**(i / 10)) for i in range(11)])
        tasks.extend(t)
        custom_decoders.update(decoders)
    
    code_stats = sinter.collect(
        num_workers=16,
        tasks=tasks,
        decoders=[],
        custom_decoders=custom_decoders,
        max_shots=100_000,
        print_progress=True,
        count_observable_error_combos=True
    )


    namefile = "build/result_transS_" + unique_name()
    dump_to_csv(code_stats, namefile, clean_after="_")

    plot_error_rate(namefile, split=True)
    import matplotlib.pyplot as plt
    plt.show()
