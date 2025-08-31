from stabcodes.stabilizer_code import SurfaceCode
from stabcodes.stim_experiment import StimExperiment, Variable
from stabcodes.visualization import dump_to_csv, plot_error_rate, unique_name
import sinter


def SurfaceDT(distance):
    code = SurfaceCode.toric_code(distance, distance)
    guide = code.logical_operators["Z"][0]
    exp = StimExperiment()
    noise = Variable("noise")
    exp.add_variables(noise)
    exp.startup(code, init_bases="Z")
    exp.measure_refined_phenom(code, meas_noise=noise, project="Z")

    for i, log in enumerate(code.logical_operators["Z"]):
        exp.observable_measurement(i, log, 0.0)

    aux = None
    for _ in range(distance):
        _, cnots, aux = code.dehn_twist(guide.support, aux)
        exp.apply_gate("CX", cnots)
        exp.depolarize2(noise, cnots)
        exp.measure_refined_phenom(code, meas_noise=noise)
        exp.depolarize1(noise)

    exp.destructive_measurement("Z")
    exp.reconstruct_stabilizers()

    for i, log in enumerate(code.logical_operators["Z"]):
        exp.reconstruct_observable(i, log)

    return exp


def SurfaceDTBell(distance):
    code, perfect_code = SurfaceCode.toric_code(distance, distance), SurfaceCode.toric_code(distance, distance)
    exp = StimExperiment()
    noise = Variable("noise")
    exp.add_variables(noise)
    exp.startup_Bell([code], [perfect_code], init_bases="Z")

    exp.reconstruct_observable_Bell(code.logical_operators["Z"][0],
                                    perfect_code.logical_operators["Z"][0], 0)
    exp.reconstruct_observable_Bell(code.logical_operators["Z"][1],
                                    perfect_code.logical_operators["Z"][1], 1)
    exp.reconstruct_observable_Bell(code.logical_operators["X"][0],
                                    perfect_code.logical_operators["X"][0], 2)
    exp.reconstruct_observable_Bell(code.logical_operators["X"][1],
                                    perfect_code.logical_operators["X"][1], 3)

    for _ in range(distance // 2):
        exp.measure_refined_phenom(code, meas_noise=noise)
        exp.depolarize1(noise, code.qubits)

    guide = code.logical_operators["Z"][0]
    guide2 = perfect_code.logical_operators["Z"][0]
    aux = None
    aux2 = None
    for _ in range(distance):
        _, cnots, aux = code.dehn_twist(guide.support, aux, check=False)
        _, cnots2, aux2 = perfect_code.dehn_twist(guide2.support, aux2, check=False)
        exp.apply_gate("CX", cnots)
        exp.apply_gate("CX", cnots2)
        exp.depolarize2(noise, cnots)
        exp.measure_refined_phenom(code, meas_noise=noise)
        exp.depolarize1(noise, support=code.qubits)

    for _ in range(distance // 2):
        exp.measure_refined_phenom(code, meas_noise=noise)
        exp.depolarize1(noise, code.qubits)

    exp.destructive_measurement_Bell(code.qubits, perfect_code.qubits, "Z")
    exp.destructive_measurement_Bell(code.qubits, perfect_code.qubits, "X")
    exp.reconstruct_stabilizers_Bell(code, perfect_code)

    exp.reconstruct_observable_Bell(code.logical_operators["Z"][0],
                                    perfect_code.logical_operators["Z"][0], 0)
    exp.reconstruct_observable_Bell(code.logical_operators["Z"][1],
                                    perfect_code.logical_operators["Z"][1], 1)
    exp.reconstruct_observable_Bell(code.logical_operators["X"][0],
                                    perfect_code.logical_operators["X"][0], 2)
    exp.reconstruct_observable_Bell(code.logical_operators["X"][1],
                                    perfect_code.logical_operators["X"][1], 3)

    return exp


if __name__ == "__main__":

    tasks = []

    for distance in range(3, 8, 2):
        exp = SurfaceDTBell(distance)
        t, _ = exp.get_task(d=[distance],
                            noise=[0.03 * ((0.04 / 0.03)**(i / 20)) for i in range(21)])
        tasks.extend(t)
    code_stats = sinter.collect(
        num_workers=16,
        tasks=tasks,
        decoders=["pymatching"],
        max_shots=1_000_000,
        print_progress=True,
        count_observable_error_combos=True, 
    )

    namefile = "build/result_DT_" + unique_name()
    dump_to_csv(code_stats, namefile)

    plot_error_rate(namefile, split=True)
    import matplotlib.pyplot as plt
    plt.show()
