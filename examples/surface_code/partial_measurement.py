from stabcodes.stim_experiment import StimExperiment, Variable
from stabcodes.stabilizer_code import SurfaceCode
from stabcodes.visualization import dump_to_csv, plot_error_rate, unique_name
from stabcodes._decoding_two_step_pymatching import TwoStepPymatching
from stabcodes.tools import reverse_dict
from itertools import chain
import sinter


def SurfacePartialMeasurementRough(distance):
    code = SurfaceCode.toric_code(distance, distance)
    patch = SurfaceCode.cylindrical_patch(distance, distance)
    codes = [code, patch]
    exp = StimExperiment()
    noise = Variable("noise")
    exp.add_variables(noise)

    exp.startup(*codes, init_bases="ZZ")
    exp.measure_refined_phenom(*codes, meas_noise=noise, project="Z")
    horizontal = True

    for i, log in enumerate(chain(code.logical_operators["Z"], patch.logical_operators["Z"])):
        exp.observable_measurement(i, log, 0.0)

    for _ in range(distance):
        exp.measure_refined_phenom(*codes, meas_noise=noise)
        exp.depolarize1(noise)

    mapp = SurfaceCode.mapping(patch, code, {0: "Z", 1: "Z"}, {patch.qubits[1]: (code.qubits[0] + distance**2 + distance) if horizontal else code.qubits[1]})
    completed_mapp = dict(zip(code.qubits, code.qubits))
    completed_mapp.update(reverse_dict(mapp))

    supports = [(j, i) for (i, j) in mapp.items()]
    exp.apply_gate("CX", supports)
    exp.depolarize2(noise, supports)
    exp.buddy_measurement(code, patch, completed_mapp, {0: {"X": "X", "Z": ""}, 1: {"Z": "Z"}}, meas_noise=noise, decoding_step0=2, decoding_step1=1)

    exp.destructive_measurement("Z")
    exp.reconstruct_stabilizers(patch, detector_decoration="1")
    exp.reconstruct_stabilizers(code, detector_decoration="2")

    for i, log in enumerate(chain(code.logical_operators["Z"], patch.logical_operators["Z"])):
        exp.reconstruct_observable(i, log)

    exp.reconstruct_observable(2, code.logical_operators["Z"][1])

    return exp


if __name__ == "__main__":
    tasks = []

    custom_decoders = {}

    for distance in range(3, 6, 2):
        exp = SurfacePartialMeasurementRough(distance)
        t, decoders = exp.get_task(decoder=TwoStepPymatching, pass_circuit=True, d=[distance],
                                   noise=[0.01 * ((0.05 / 0.01)**(i / 10)) for i in range(11)])
        tasks.extend(t)
        custom_decoders.update(decoders)

    code_stats = sinter.collect(
        num_workers=7,
        tasks=tasks,
        decoders=[],
        custom_decoders=custom_decoders,
        max_shots=10_000,
#        print_progress=True
    )

    namefile = "result_partialmeas_" + unique_name()
    dump_to_csv(code_stats, namefile, clean_after="_")

    plot_error_rate(namefile)
