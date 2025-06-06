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

    # for _ in range(distance):
    #     exp.measure_refined_phenom(code, meas_noise=noise)
    #     exp.depolarize1(noise)

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

    exp.reconstruct_observable(1, code.logical_operators["Z"][0])

    return exp


if __name__ == "__main__":

    tasks = []

    for distance in range(3, 8, 2):
        exp = SurfaceDT(distance)
        t, _ = exp.get_task(d=[distance],
                            noise=[0.035 * ((0.045 / 0.035)**(i / 20)) for i in range(21)])
        tasks.extend(t)
    code_stats = sinter.collect(
        num_workers=7,
        tasks=tasks,
        decoders=["pymatching"],
        max_shots=100_000,
        # separated = True
    )

    namefile = "result_DT_" + unique_name()
    dump_to_csv(code_stats, namefile)

    plot_error_rate(namefile)
