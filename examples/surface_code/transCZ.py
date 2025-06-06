from stabcodes.stim_experiment import StimExperiment, Variable
from stabcodes.stabilizer_code import SurfaceCode
from stabcodes.visualization import dump_to_csv, plot_error_rate, unique_name
from stabcodes._decoding_two_step_pymatching import TwoStepPymatching
from stabcodes.tools import mapping
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

    exp.apply_gate("CZ", [(qb, qb + len(code1.qubits)) for qb in code1.qubits])
    exp.depolarize2(noise, [(qb, qb + len(code1.qubits)) for qb in code1.qubits])
    mapp = mapping(code1, code2, {0: "X", 1: "Z"}, {code1.qubits[0]: code2.qubits[0]})
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


if __name__ == "__main__":
    tasks = []

    custom_decoders = {}

    for distance in range(3, 5, 2):
        exp = SurfaceTransversalCZ(distance)
        t, decoders = exp.get_task(decoder=TwoStepPymatching, pass_circuit=True, d=[distance],
                                   noise=[0.01 * ((0.05 / 0.01)**(i / 10)) for i in range(11)])
        tasks.extend(t)
        with open("foo.circ", "w") as f:
            f.write(str(tasks[0].circuit))
        custom_decoders.update(decoders)

    code_stats = sinter.collect(
        num_workers=1,
        tasks=tasks,
        decoders=[],
        custom_decoders=custom_decoders,
        max_shots=10_000,
#        print_progress=True
    )

    namefile = "result_transCZ_" + unique_name()
    dump_to_csv(code_stats, namefile)

    plot_error_rate(namefile)
