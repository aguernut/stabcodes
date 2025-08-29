from stabcodes.stim_experiment import StimExperiment, Variable
from stabcodes.stabilizer_code import SurfaceCode
from stabcodes.visualization import dump_to_csv, plot_error_rate, unique_name
from stabcodes._decoding_two_step_pymatching import TwoStepPymatching
from itertools import chain
import sinter


def SurfaceTransversalCZ(distance):
    code1, code2 = SurfaceCode.toric_code(distance, distance), SurfaceCode.toric_code(distance, distance)
    codes = [code1, code2]
    exp = StimExperiment()
    noise = Variable("noise")
    exp.add_variables(noise)

    exp.startup(*codes, init_bases="ZZ")
    code1.measure_count = exp._measure_clock
    code2.measure_count = exp._measure_clock
    exp.measure_refined_phenom(*codes, meas_noise=noise, project="Z")

    for i, log in enumerate(chain(code1.logical_operators["Z"], code2.logical_operators["Z"])):
        exp.observable_measurement(i, log, 0.0)

    for _ in range(distance // 2):
        exp.measure_refined_phenom(*codes, meas_noise=noise)
        exp.depolarize1(noise)

    mapp = SurfaceCode.mapping(code1, code2, {0: "X", 1: "Z"}, {code1.qubits[0]: code2.qubits[0]})
    exp.apply_gate("CZ", mapp.items())
    exp.depolarize2(noise, mapp.items())
    exp.buddy_measurement(code1, code2, mapp, {0: {"X": "Z", "Z": ""}, 1: {"X": "Z", "Z": ""}}, noise, 1, 2)

    for _ in range(distance // 2):
        exp.measure_refined_phenom(code1, meas_noise=noise, detector_decoration="1")
        exp.measure_refined_phenom(code2, meas_noise=noise, detector_decoration="2")
        exp.depolarize1(noise)

    exp.destructive_measurement("Z")
    exp.reconstruct_stabilizers(code1, detector_decoration="1")
    exp.reconstruct_stabilizers(code2, detector_decoration="2")

    for i, log in enumerate(chain(code1.logical_operators["Z"], code2.logical_operators["Z"])):
        exp.reconstruct_observable(i, log)

    return exp


def SurfaceTransversalCZBell(distance):
    code1, code2 = SurfaceCode.toric_code(distance, distance), SurfaceCode.toric_code(distance, distance)
    perfect_code1, perfect_code2 = SurfaceCode.toric_code(distance, distance), SurfaceCode.toric_code(distance, distance)

    exp = StimExperiment()
    noise = Variable("noise")
    exp.add_variables(noise)

    exp.startup(code1, code2, perfect_code1, perfect_code2, init_bases="ZZZZ")

    exp.destructive_measurement_Bell(code1.qubits, perfect_code1.qubits, "Z")
    exp.destructive_measurement_Bell(code2.qubits, perfect_code2.qubits, "Z")
    exp.destructive_measurement_Bell(code1.qubits, perfect_code1.qubits, "X")
    exp.destructive_measurement_Bell(code2.qubits, perfect_code2.qubits, "X")

    for i, (obs1, obs2) in enumerate(zip(code1.logical_operators["Z"], perfect_code1.logical_operators["Z"])):
        exp.reconstruct_observable_Bell(obs1, obs2, i)

    for i, (obs1, obs2) in enumerate(zip(code1.logical_operators["X"], perfect_code1.logical_operators["X"])):
        exp.reconstruct_observable_Bell(obs1, obs2, i + 2)

    for i, (obs1, obs2) in enumerate(zip(code2.logical_operators["Z"], perfect_code2.logical_operators["Z"])):
        exp.reconstruct_observable_Bell(obs1, obs2, i + 4)

    for i, (obs1, obs2) in enumerate(zip(code2.logical_operators["X"], perfect_code2.logical_operators["X"])):
        exp.reconstruct_observable_Bell(obs1, obs2, i + 6)

    exp.measure_refined_phenom(code1, meas_noise=noise, project="")
    exp.measure_refined_phenom(code2, meas_noise=noise, project="")
    exp.measure_refined_phenom(perfect_code1, meas_noise=0.0, project="")
    exp.measure_refined_phenom(perfect_code2, meas_noise=0.0, project="")

    for _ in range(distance // 2):
        exp.measure_refined_phenom(code1, code2, meas_noise=noise)
        exp.depolarize1(noise, code1.qubits)
        exp.depolarize1(noise, code2.qubits)

    mapp = SurfaceCode.mapping(code1, code2, {0: "X", 1: "Z"}, {code1.qubits[0]: code2.qubits[0]})
    mapp2 = SurfaceCode.mapping(perfect_code1, perfect_code2, {0: "X", 1: "Z"}, {perfect_code1.qubits[0]: perfect_code2.qubits[0]})
    exp.apply_gate("CZ", mapp.items())
    exp.apply_gate("CZ", mapp2.items())
    exp.depolarize2(noise, mapp.items())
    exp.buddy_measurement(code1, code2, mapp, {0: {"X": "Z", "Z": ""}, 1: {"X": "Z", "Z": ""}}, noise, 1, 2)
    exp.buddy_measurement(perfect_code1, perfect_code2, mapp2, {0: {"X": "Z", "Z": ""}, 1: {"X": "Z", "Z": ""}}, 0.0, 1, 2)

    for _ in range(distance // 2):
        exp.measure_refined_phenom(code1, meas_noise=noise, detector_decoration="1")
        exp.measure_refined_phenom(code2, meas_noise=noise, detector_decoration="2")
        exp.depolarize1(noise, code1.qubits)
        exp.depolarize1(noise, code2.qubits)

    exp.destructive_measurement_Bell(code1.qubits, perfect_code1.qubits, "Z")
    exp.destructive_measurement_Bell(code2.qubits, perfect_code2.qubits, "Z")
    exp.destructive_measurement_Bell(code1.qubits, perfect_code1.qubits, "X")
    exp.destructive_measurement_Bell(code2.qubits, perfect_code2.qubits, "X")
    exp.reconstruct_stabilizers_Bell(code1, perfect_code1, detector_decoration="1")
    exp.reconstruct_stabilizers_Bell(code2, perfect_code2, detector_decoration="2")

    for i, (obs1, obs2) in enumerate(zip(code1.logical_operators["Z"], perfect_code1.logical_operators["Z"])):
        exp.reconstruct_observable_Bell(obs1, obs2, i)
        exp.reconstruct_observable_Bell(obs1, obs2, i + 6)

    for i, (obs1, obs2) in enumerate(zip(code1.logical_operators["X"], perfect_code1.logical_operators["X"])):
        exp.reconstruct_observable_Bell(obs1, obs2, i + 2)

    for i, (obs1, obs2) in enumerate(zip(code2.logical_operators["Z"], perfect_code2.logical_operators["Z"])):
        exp.reconstruct_observable_Bell(obs1, obs2, i + 2)
        exp.reconstruct_observable_Bell(obs1, obs2, i + 4)

    for i, (obs1, obs2) in enumerate(zip(code2.logical_operators["X"], perfect_code2.logical_operators["X"])):
        exp.reconstruct_observable_Bell(obs1, obs2, i + 6)

    return exp


if __name__ == "__main__":
    tasks = []

    custom_decoders = {}

    for distance in range(3, 7, 2):
        exp = SurfaceTransversalCZBell(distance)
        t, decoders = exp.get_task(decoder=TwoStepPymatching, pass_circuit=True, d=[distance],
                                   noise=[0.01 * ((0.05 / 0.01)**(i / 10)) for i in range(11)])
        tasks.extend(t)
        custom_decoders.update(decoders)

    code_stats = sinter.collect(
        num_workers=7,
        tasks=tasks,
        decoders=[],
        custom_decoders=custom_decoders,
        max_shots=100_000,
#        print_progress=True
    )

    namefile = "result_transCZ_" + unique_name()
    dump_to_csv(code_stats, namefile, clean_after="_")

    plot_error_rate(namefile)
