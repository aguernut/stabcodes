from stabcodes.stim_experiment import StimExperiment, Variable
from stabcodes.stabilizer_code import SurfaceCode
from stabcodes.visualization import dump_to_csv, plot_error_rate
from stabcodes._decoding_two_step_pymatching import TwoStepPymatching
from stabcodes.tools import rotate_to_place_first
from itertools import chain
import sinter


def SurfaceTransversalCNOT(distance):
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

    exp.apply_gate("CX", [(qb, qb + len(code1.qubits)) for qb in code1.qubits])
    exp.depolarize2(noise, [(qb, qb + len(code1.qubits)) for qb in code1.qubits])
    mapp = mapping(code1, code2, {0: "X", 1: "X"}, {code1.qubits[0]: code2.qubits[0]})
    exp._circuit += buddy_measurement(code1, code2, mapp, {0: {"X": "X", "Z": ""}, 1: {"X": "", "Z": "Z"}}, noise, 1, 2)

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


def mapping(code0, code1, modes, qb, rev=False):
    seen = set()
    seen2 = set()
    traited = set()
    while len(seen) < len(code0.stabilizers[modes[0]]):
        for stab in code0.iter_stabilizers(modes[0]):
            if stab in seen:
                continue
            if len(stab.order) == 1:
                seen.add(stab)
                continue

            if inter := set(stab.order) & (qb.keys() - traited):
                if len(inter) == 4:
                    seen.add(stab)
                    continue

                trans_inter = set(qb[i] for i in inter)
                for stab2 in code1.iter_stabilizers(modes[1]):
                    if stab2 in seen2:
                        continue

                    if set(stab2.order) & trans_inter:
                        a = next(iter(inter))
                        try:
                            rotate_to_place_first(stab.order, a)
                            rotate_to_place_first(stab2.order, qb[a])
                        except ValueError:
                            print(stab.order, stab2.order)
                            raise ValueError

                        for (q0, q1) in zip(stab.order, stab2.order):
                            qb[q0] = q1

                        seen.add(stab)
                        seen2.add(stab2)
                        break

                else:
                    raise NotImplementedError
                traited.add(a)
                break
        else:
            if len(seen) < len(code0.stabilizers[modes[0]]):
                raise NotImplementedError

    return qb


def buddy_measurement(code0, code1, mapping, mode, meas_noise="{meas_noise}", decoding_step0=1, decoding_step1=1):
    """
    Patch must be in code1
    """
    circ = ""

    # Buddies computation

    # Buddies computation for code0
    buddies0 = {}
    for basis in mode[0]:
        buddies0[basis] = {}
        for stab in code0.iter_stabilizers(basis):
            buddies0[basis][stab] = []
            for buddy_basis in mode[0][basis]:
                for buddy in code1.iter_stabilizers(buddy_basis):
                    if set(mapping[i] for i in stab.support_as_set) >= buddy.support_as_set:
                        buddies0[basis][stab].append(buddy)
                        break

    # Buddies computation for code1
    buddies1 = {}
    for basis in mode[1]:
        buddies1[basis] = {}
        for stab in code1.iter_stabilizers(basis):
            buddies1[basis][stab] = []
            for buddy_basis in mode[1][basis]:
                for buddy in code0.iter_stabilizers(buddy_basis):
                    if set(mapping[i] for i in buddy.support_as_set) >= stab.support_as_set:
                        buddies1[basis][stab].append(buddy)
                        break

    # Measurements
    deb = None

    # Measurement of code0
    for basis in mode[0]:
        for stab in code0.iter_stabilizers(basis):
            stab.measure(code0.measure_count)
            if deb is None:
                deb = stab.last_measure_time[-1]
            circ += f"MPP({meas_noise}) " + "*".join(p.kind + str(p.qubit) for p in stab.paulis if p.kind != "I") + "\n"
            buddies = buddies0[basis].get(stab)

            if not buddies:
                circ += f"DETECTOR({decoding_step0}) rec[-1] rec[{stab.last_measure_time[-2] - stab.last_measure_time[-1] - 1}]\n"
            else:
                for buddy in buddies:
                    circ += f"DETECTOR({decoding_step0}) rec[-1] rec[{stab.last_measure_time[-2] - stab.last_measure_time[-1] - 1}] rec[{buddy.last_measure_time[-1] - stab.last_measure_time[-1] - 1}]\n"

    # Measurement of code1
    for basis in mode[1]:
        for stab in code1.iter_stabilizers(basis):
            stab.measure(code1.measure_count)
            circ += f"MPP({meas_noise}) " + "*".join(p.kind + str(p.qubit) for p in stab.paulis if p.kind != "I") + "\n"
            buddies = buddies1[basis].get(stab)

            if not buddies:
                circ += f"DETECTOR({decoding_step1}) rec[-1] rec[{stab.last_measure_time[-2] - stab.last_measure_time[-1] - 1}]\n"
            else:
                for buddy in buddies:
                    circ += f"DETECTOR({decoding_step1}) rec[-1] rec[{stab.last_measure_time[-2] - stab.last_measure_time[-1] - 1}] rec[{(v if (v := buddy.last_measure_time[-1]) < deb else buddy.last_measure_time[-2]) - stab.last_measure_time[-1] - 1}]\n"

    return circ


if __name__ == "__main__":
    tasks = []

    for distance in range(3, 6, 2):
        exp = SurfaceTransversalCNOT(distance)
        tasks.extend(exp.get_task(decoder=TwoStepPymatching, pass_circuit=True, d=[distance],
                                  noise=[0.01 * ((0.05 / 0.01)**(i / 10)) for i in range(11)]))
    code_stats = sinter.collect(
        num_workers=11,
        tasks=tasks,
        decoders=[],
        custom_decoders={},
        max_shots=10_000,
        print_progress=True,
        # separated = True
    )

    namefile = "result_transCNOT"
    dump_to_csv(code_stats, namefile)

    plot_error_rate(namefile)
