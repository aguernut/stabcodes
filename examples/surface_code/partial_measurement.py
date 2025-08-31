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

def SurfacePartialMeasurementRoughBell(distance):
    code, perfect_code = SurfaceCode.toric_code(distance, distance), SurfaceCode.toric_code(distance, distance)
    patch, perfect_patch = SurfaceCode.cylindrical_patch(distance, distance), SurfaceCode.cylindrical_patch(distance, distance)

    exp = StimExperiment()
    noise = Variable("noise")
    exp.add_variables(noise)

    exp.startup_Bell([code, patch], [perfect_code, perfect_patch], init_bases="ZZ")

    horizontal = True

    for i, (log1, log2) in enumerate(zip(chain(code.logical_operators["Z"], patch.logical_operators["Z"]), chain(perfect_code.logical_operators["Z"], perfect_patch.logical_operators["Z"]))):
        exp.reconstruct_observable_Bell(log1, log2, i)

    for _ in range(distance):
        exp.measure_refined_phenom(code, patch, meas_noise=noise)
        exp.depolarize1(noise, code.qubits)
        exp.depolarize1(noise, patch.qubits)

    mapp = SurfaceCode.mapping(patch, code, {0: "Z", 1: "Z"}, {patch.qubits[1]: (code.qubits[0] + distance**2 + distance) if horizontal else code.qubits[1]})
    completed_mapp = dict(zip(code.qubits, code.qubits))
    completed_mapp.update(reverse_dict(mapp))
    
    mapp2 = SurfaceCode.mapping(perfect_patch, perfect_code, {0: "Z", 1: "Z"}, {perfect_patch.qubits[1]: (perfect_code.qubits[0] + distance**2 + distance) if horizontal else perfect_code.qubits[1]})
    completed_mapp2 = dict(zip(perfect_code.qubits, perfect_code.qubits))
    completed_mapp2.update(reverse_dict(mapp2))

    supports = [(j, i) for (i, j) in mapp.items()]
    exp.apply_gate("CX", supports)
    exp.depolarize2(noise, supports)
    exp.buddy_measurement(code, patch, completed_mapp, {0: {"X": "X", "Z": ""}, 1: {"X": "","Z": "Z"}}, meas_noise=noise, decoding_step0=2, decoding_step1=1)
    
    supports2 = [(j, i) for (i, j) in mapp2.items()]
    exp.apply_gate("CX", supports2)
    exp.buddy_measurement(perfect_code, perfect_patch, completed_mapp2, {0: {"X": "X", "Z": ""}, 1: {"X": "", "Z": "Z"}}, meas_noise=0.0, decoding_step0=2, decoding_step1=1)

    exp.destructive_measurement_Bell(code.qubits, perfect_code.qubits, "Z")
    exp.destructive_measurement_Bell(patch.qubits, perfect_patch.qubits, "Z")
    exp.destructive_measurement_Bell(code.qubits, perfect_code.qubits, "X")
    exp.destructive_measurement_Bell(patch.qubits, perfect_patch.qubits, "X")
    
    exp.reconstruct_stabilizers_Bell(code, perfect_code, detector_decoration="2")
    exp.reconstruct_stabilizers_Bell(patch, perfect_patch, detector_decoration="1")

    for i, (log1, log2) in enumerate(zip(chain(code.logical_operators["Z"], patch.logical_operators["Z"]), chain(perfect_code.logical_operators["Z"], perfect_patch.logical_operators["Z"]))):
        exp.reconstruct_observable_Bell(log1, log2, i)
    
    exp.reconstruct_observable_Bell(code.logical_operators["Z"][1], perfect_code.logical_operators["Z"][1], 2)

    return exp

def SurfacePartialMeasurementBell(distance):
    code, perfect_code = SurfaceCode.toric_code(distance, distance), SurfaceCode.toric_code(distance, distance)
    patch, perfect_patch = SurfaceCode.cylindrical_patch(distance, distance), SurfaceCode.cylindrical_patch(distance, distance)

    exp = StimExperiment()
    noise = Variable("noise")
    exp.add_variables(noise)

    exp.startup_Bell([code, patch], [perfect_code, perfect_patch], init_bases="ZZ")

    horizontal = True

    exp.reconstruct_observable_Bell(code.logical_operators["Z"][0], perfect_code.logical_operators["Z"][0], 0)
    exp.reconstruct_observable_Bell(code.logical_operators["Z"][1], perfect_code.logical_operators["Z"][1], 1)
    exp.reconstruct_observable_Bell(code.logical_operators["X"][0], perfect_code.logical_operators["X"][0], 2)
    
    exp.reconstruct_observable_Bell(patch.logical_operators["Z"][0], perfect_patch.logical_operators["Z"][0], 3)

    for _ in range(distance):
        exp.measure_refined_phenom(code, patch, meas_noise=noise)
        exp.depolarize1(noise, code.qubits)
        exp.depolarize1(noise, patch.qubits)

    mapp = SurfaceCode.mapping(patch, code, {0: "Z", 1: "Z"}, {patch.qubits[1]: (code.qubits[0] + distance**2 + distance) if horizontal else code.qubits[1]})
    completed_mapp = dict(zip(code.qubits, code.qubits))
    completed_mapp.update(reverse_dict(mapp))
    
    mapp2 = SurfaceCode.mapping(perfect_patch, perfect_code, {0: "Z", 1: "Z"}, {perfect_patch.qubits[1]: (perfect_code.qubits[0] + distance**2 + distance) if horizontal else perfect_code.qubits[1]})
    completed_mapp2 = dict(zip(perfect_code.qubits, perfect_code.qubits))
    completed_mapp2.update(reverse_dict(mapp2))

    supports = [(j, i) for (i, j) in mapp.items()]
    exp.apply_gate("CX", supports)
    exp.depolarize2(noise, supports)
    exp.buddy_measurement(code, patch, completed_mapp, {0: {"X": "X", "Z": ""}, 1: {"X": "","Z": "Z"}}, meas_noise=noise, decoding_step0=2, decoding_step1=1)
    
    supports2 = [(j, i) for (i, j) in mapp2.items()]
    exp.apply_gate("CX", supports2)
    exp.buddy_measurement(perfect_code, perfect_patch, completed_mapp2, {0: {"X": "X", "Z": ""}, 1: {"X": "", "Z": "Z"}}, meas_noise=0.0, decoding_step0=2, decoding_step1=1)

    exp.destructive_measurement_Bell(code.qubits, perfect_code.qubits, "Z")
    exp.destructive_measurement_Bell(patch.qubits, perfect_patch.qubits, "Z")
    exp.destructive_measurement_Bell(code.qubits, perfect_code.qubits, "X")
    exp.destructive_measurement_Bell(patch.qubits, perfect_patch.qubits, "X")
    
    exp.reconstruct_stabilizers_Bell(code, perfect_code, detector_decoration="2")
    exp.reconstruct_stabilizers_Bell(patch, perfect_patch, detector_decoration="1")

    
    exp.reconstruct_observable_Bell(code.logical_operators["Z"][0], perfect_code.logical_operators["Z"][0], 0)
    exp.reconstruct_observable_Bell(code.logical_operators["Z"][1], perfect_code.logical_operators["Z"][1], 1)
    exp.reconstruct_observable_Bell(code.logical_operators["X"][0], perfect_code.logical_operators["X"][0], 2)
    #exp.reconstruct_observable_Bell(code.logical_operators["X"][1], perfect_code.logical_operators["X"][1], 3)
    
    exp.reconstruct_observable_Bell(patch.logical_operators["Z"][0], perfect_patch.logical_operators["Z"][0], 3)
    exp.reconstruct_observable_Bell(code.logical_operators["Z"][1], perfect_code.logical_operators["Z"][1], 3)
    # exp.reconstruct_observable_Bell(patch.logical_operators["X"][0], perfect_patch.logical_operators["X"][0], 5)
    # exp.reconstruct_observable_Bell(code.logical_operators["X"][0], perfect_code.logical_operators["X"][0], 5)

    return exp


if __name__ == "__main__":
    tasks = []

    custom_decoders = {}

    for distance in range(3, 6, 2):
        exp = SurfacePartialMeasurementBell(distance)
        t, decoders = exp.get_task(decoder=TwoStepPymatching, pass_circuit=True, d=[distance],
                                   noise=[0.01 * ((0.05 / 0.01)**(i / 10)) for i in range(11)])
        tasks.extend(t)
        with open("foo.txt", "w") as f:
            f.write(str(t[-1].circuit))
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

    namefile = "build/result_partialmeas_" + unique_name()
    dump_to_csv(code_stats, namefile, clean_after="_")

    plot_error_rate(namefile, split=True)
    import matplotlib.pyplot as plt
    plt.show()