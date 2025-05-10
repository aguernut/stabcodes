from stabcodes.stabilizer_code import SurfaceCode
from stabcodes.stim_experiment import StimExperiment, Variable
from stabcodes.visualization import dump_to_csv, plot_error_rate
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


if __name__ == "__main__":

    tasks = []

    for distance in range(3, 12, 2):
        exp = SurfaceMemory(distance)
        tasks.extend(exp.get_task(d=[distance],
                                  noise=[0.035 * ((0.045 / 0.035)**(i / 20)) for i in range(21)]))
    code_stats = sinter.collect(
        num_workers=11,
        tasks=tasks,
        decoders=["pymatching"],
        max_shots=1_000_000,
        print_progress=True,
        # separated = True
    )

    namefile = "result_memory"
    dump_to_csv(code_stats, namefile)

    plot_error_rate(namefile)
