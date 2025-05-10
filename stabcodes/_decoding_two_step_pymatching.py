import math
from itertools import count
import pathlib
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import hypergraphx as hgx
import pymatching
import numpy as np
import stim
from typing import Callable, List, TYPE_CHECKING
from typing import Tuple


from sinter._decoding._decoding_decoder_class import Decoder

PRODUCE = False
DEBUG = False


class TwoStepPymatching(Decoder):

    def __init__(self, circuit: stim.Circuit):
        super().__init__()
        self.circuit = circuit

    def decode_via_files(self,
                         *,
                         num_shots: int,
                         num_dets: int,
                         num_obs: int,
                         dem_path: pathlib.Path,
                         dets_b8_in_path: pathlib.Path,
                         obs_predictions_b8_out_path: pathlib.Path,
                         tmp_dir: pathlib.Path,
                         ) -> None:
        """Use pymatching to predict observables from detection events."""

        filtered_circ, filtered_detectors = filter_circ(self.circuit, 2)
        filtered_circ2, filtered_detectors2 = filter_circ(self.circuit, 1)
        ids = count(num_obs)
        maps = {}
        maps2 = {}
        maps_id = {}
        dem = self.circuit.detector_error_model(
            allow_gauge_detectors=False,
            approximate_disjoint_errors=True,
            block_decomposition_from_introducing_remnant_edges=True,
            decompose_errors=True,
            flatten_loops=True,
            ignore_decomposition_failures=True,
        )

        # if PRODUCE:
        #     g = hgx.Hypergraph(weighted=True)
        #     g.add_node(99999999)

        for inst in dem:
            if PRODUCE:
                for foo in split_around_all(inst.targets_copy()):
                    bar = {i.val for i in foo[0]}
                    if len(bar) == 1:
                        a, = bar
                        if inst.type != "error" and a in g.get_nodes():
                            continue
                        if (a, 99999999) not in g._edge_list:
                            g.add_edge((a, 99999999), foo[1])
                        continue

                    if len(bar) == 2:
                        a, b = bar
                        if tuple(sorted((a, b))) not in g._edge_list:
                            g.add_edge((a, b), foo[1])
                        continue

                    if len(bar) == 3:
                        a, b, c = bar
                        g.add_edge((a, b, c), foo[1])

            for foo in split_around_separator(inst.targets_copy()):
                bar = {i.val for i in foo}
                if len(foo) == 1:
                    continue
                if len(foo) == 2:
                    if len(bar & filtered_detectors) == 1:
                        filtered = next(iter(bar & filtered_detectors))
                        kept = tuple(sorted(bar - filtered_detectors))
                        maps2[filtered] = kept
                        maps2[kept] = filtered
                        next_id = next(ids)
                        maps_id[kept] = next_id
                        maps_id[next_id] = kept
                    continue
                if len(foo) == 3:
                    try:
                        filtered = next(iter(bar & filtered_detectors))
                    except StopIteration:
                        raise ValueError(f"Some errors flip three unproperly annotated detectors: {bar}")
                    kept = tuple(sorted(bar - filtered_detectors))
                    maps[filtered] = kept
                    maps[kept] = filtered
                    next_id = next(ids)
                    maps_id[kept] = next_id
                    maps_id[next_id] = kept
                else:
                    raise ValueError(f"Some errors flip more than three detectors after decomposition: {set(i.val for i in foo)}")

        G = detector_error_model_to_nx_graph2(dem, maps_id, filtered_detectors, filtered_detectors2)
        boundary_node = max(G)
        # if PRODUCE:
        #     # print(g._edge_list)
        #     # gg = nx.Graph()
        #     # for e in g._edge_list:
        #     #     if len(e) == 2:
        #     #         gg.add_edge(*e)
        #     #     else:
        #     #         gg.add_edge(e[0], e[1], multiedge=e[2])
        #     #         gg.add_edge(e[2], e[1], multiedge=e[0])
        #     #         gg.add_edge(e[0], e[2], multiedge=e[1])
        #     # #print(gg.edges(data=True))
        #     # nx.draw(gg, with_labels=True)
        #     # plt.show()
        #     import os
        #     if f"incidence_{code.split('_')[-1]}.txt" not in os.listdir():
        #         with open(f"incidence_{code.split('_')[-1]}.txt", "wb") as f:
        #             g.binary_incidence_matrix().toarray().dump(f)
        #
        #         with open(f"obs_{code.split('_')[-1]}", "wb") as f:
        #             pickle.dump(g._edge_list, f)

        G1 = G.copy()
        if DEBUG:
            print(filtered_detectors)
        for node in filtered_detectors:
            if DEBUG:
                print(f"\n Scanning {node}")
                print(len(G1[node]))
            for neigh in G1[node]:
                if neigh == boundary_node:
                    if DEBUG:
                        print(f"{neigh} is the boundary node")
                    continue
                if neigh in filtered_detectors:
                    if DEBUG:
                        print(f"Skipped: {neigh} marked by 2")
                    continue

                if node in maps and neigh in maps[node]:
                    if DEBUG:
                        print(f"Skipped: {neigh} because {maps[node]} was kept")
                    continue

                if DEBUG:
                    print(f"Adding: {neigh} to the boundary with auxiliary qubit {maps_id[(neigh,)]}")
                foo = G1[node][neigh]
                if maps_id[(neigh,)] not in foo["fault_ids"]:
                    foo["fault_ids"].append(maps_id[(neigh,)])
                G1.add_edge(neigh, boundary_node, **foo)

            G1.remove_node(node)

        G2 = G.copy()
        G2.remove_nodes_from(filtered_detectors2)
        # H = nx.Graph()
        # for edge in g._edge_list:
        #     H.add_edge(*edge[:2])
        # nx.draw(G2, with_labels=True)
        # plt.show()

        matching_graph1 = pymatching.Matching(G1)
        matching_graph2 = pymatching.Matching(G2)
        num_det_bytes = math.ceil(num_dets / 8)

        # if PRODUCE:
        #     from time import time
        #     v = time()
        #     with open(dets_b8_in_path, 'rb') as dets_in_f:
        #         det_bytes = np.fromfile(dets_in_f, dtype=np.uint8)
        #         with open(f"sample_{code}_{v}", "wb") as f:
        #             det_bytes.tofile(f)
        #
        #     with open(obs_all_path, "rb") as obs_in_f:
        #         obs_bytes = np.fromfile(obs_in_f, dtype=np.uint8)
        #         with open(f"answer_{code}_{v}", "wb") as f:
        #             obs_bytes.tofile(f)

        # note: extra 2 are the boundary node and the invincible-observable-boundary-edge node
        det_bits_buffer = np.zeros(num_dets + 2, dtype=np.bool8)
        with open(dets_b8_in_path, 'rb') as dets_in_f:
            with open(obs_predictions_b8_out_path, 'wb') as obs_out_f:
                for _ in range(num_shots):
                    det_bytes = np.fromfile(dets_in_f, dtype=np.uint8, count=num_det_bytes)
                    det_bits_buffer[:num_dets] = np.unpackbits(det_bytes, count=num_dets, bitorder='little')
                    oldd = det_bits_buffer[:num_dets].copy()
                    filtered_syndrome = filter_syndrome(det_bits_buffer[:num_dets], filtered_detectors)
                    try:
                        first_prediction_bits = matching_graph1.decode(filtered_syndrome)
                    except ValueError as e:
                        print(list((i for i, j in enumerate(oldd) if j)))
                        print(list((i for i, j in enumerate(filtered_syndrome) if j)))
                        raise e
                    det_bits_buffer[:num_dets] ^= filtered_syndrome

                    for i, j in enumerate(first_prediction_bits[num_obs:]):
                        if j:
                            kept = maps_id[i + num_obs]
                            filt = maps[kept] if len(kept) > 1 else maps2[kept]
                            det_bits_buffer[filt] = 1 - det_bits_buffer[filt]

                    filtered_syndrome = filter_syndrome(det_bits_buffer[:num_dets], filtered_detectors2)
                    try:
                        second_prediction_bits = matching_graph2.decode(filtered_syndrome)
                    except ValueError as e:
                        print(oldd)
                        old = filter_syndrome(oldd, filtered_detectors)

                        print(list((i for i, j in enumerate(old) if j)))

                        print(first_prediction_bits)
                        print(list((i for i, j in enumerate(filtered_syndrome) if j)))
                        print(maps)
                        print(maps2)
                        print(maps_id)
                        nx.draw(G1, with_labels=True)
                        plt.show()
                        nx.draw(G2, with_labels=True)
                        plt.show()
                        raise e

                    if len(second_prediction_bits[:num_obs]) < num_obs:
                        second_prediction_bits = np.append(np.zeros(num_obs - len(second_prediction_bits[:num_obs]), dtype=int), second_prediction_bits[:num_obs])
                    flipped_obs = (first_prediction_bits[:num_obs] + second_prediction_bits[:num_obs]) % 2
                    np.packbits(flipped_obs, bitorder='little').tofile(obs_out_f)


def iter_flatten_model(model: stim.DetectorErrorModel,
                       handle_error: Callable[[float, List[int], List[int]], None],
                       handle_detector_coords: Callable[[int, np.ndarray], None]):
    det_offset = 0
    coords_offset = np.zeros(100, dtype=np.float64)

    def _helper(m: stim.DetectorErrorModel, reps: int):
        nonlocal det_offset
        nonlocal coords_offset
        for _ in range(reps):
            for instruction in m:
                if isinstance(instruction, stim.DemRepeatBlock):
                    _helper(instruction.body_copy(), instruction.repeat_count)
                elif isinstance(instruction, stim.DemInstruction):
                    if instruction.type == "error":
                        dets: List[int] = []
                        frames: List[int] = []
                        t: stim.DemTarget
                        p = instruction.args_copy()[0]
                        for t in instruction.targets_copy():
                            if t.is_relative_detector_id():
                                dets.append(t.val + det_offset)
                            elif t.is_logical_observable_id():
                                frames.append(t.val)
                            elif t.is_separator():
                                # Treat each component of a decomposed error as an independent error.
                                # (Ideally we could configure some sort of correlated analysis; oh well.)
                                handle_error(p, dets, frames)
                                frames = []
                                dets = []
                        # Handle last component.
                        handle_error(p, dets, frames)
                    elif instruction.type == "shift_detectors":
                        det_offset += instruction.targets_copy()[0]
                        a = np.array(instruction.args_copy())
                        coords_offset[:len(a)] += a
                    elif instruction.type == "detector":
                        a = np.array(instruction.args_copy())
                        for t in instruction.targets_copy():
                            handle_detector_coords(t.val + det_offset, a + coords_offset[:len(a)])
                    elif instruction.type == "logical_observable":
                        pass
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
    _helper(model, 1)


def detector_error_model_to_nx_graph2(model: stim.DetectorErrorModel, maps, filtered_detectors, filtered_detectors2) -> 'nx.Graph':
    """Convert a stim error model into a NetworkX graph."""

    # Local import to reduce sinter's startup time.
    try:
        import networkx as nx
    except ImportError as ex:
        raise ImportError(
            "pymatching was installed without networkx?"
            "Run `pip install networkx`.\n"
        ) from ex

    g = nx.Graph()
    boundary_node = model.num_detectors
    g.add_node(boundary_node, is_boundary=True, coords=[-1, -1, -1])

    def handle_error(p: float, dets: List[int], frame_changes: List[int]):
        if p == 0:
            return

        dets = tuple(sorted(dets))
        frame_changes = frame_changes + ([maps[dets]] if dets in maps and maps[dets] not in frame_changes else [])
        # print(dets, frame_changes)
        if len(dets) == 0:
            # No symptoms for this error.
            # Code probably has distance 1.
            # Accept it and keep going, though of course decoding will probably perform terribly.
            return
        if len(dets) == 1:
            dets = [dets[0], boundary_node]
        if len(dets) == 3:
            det_as_set = set(dets)
            a = det_as_set & filtered_detectors
            b = det_as_set & filtered_detectors2
            c = next(iter(det_as_set - (a | b)))

            if len(a) != 1 or len(b) != 1:
                raise NotImplementedError(
                    f"Unflaged error with 3 symptoms can't become an edge or boundary edge: {dets!r}.")

            a = next(iter(a))
            b = next(iter(b))

            assert not frame_changes
            handle_error(p, [b, c], [maps[tuple(sorted((b, c)))]])
            handle_error(p, [a, c], [])
            return

        if len(dets) > 3:
            raise NotImplementedError(
                f"Error with more than 3 symptoms can't become an edge or boundary edge: {dets!r}.")
        if g.has_edge(*dets):
            edge_data = g.get_edge_data(*dets)
            old_p = edge_data["error_probability"]
            old_frame_changes = edge_data["fault_ids"]
            # If frame changes differ, the code has distance 2; just keep whichever was first.
            if set(old_frame_changes) == set(frame_changes):
                p = p * (1 - old_p) + old_p * (1 - p)
                g.remove_edge(*dets)
            else:
                raise NotImplementedError
        if p > 0.5:
            p = 1 - p
        if p > 0:
            g.add_edge(*dets, weight=math.log((1 - p) / p), fault_ids=frame_changes, error_probability=p)

    def handle_detector_coords(detector: int, coords: np.ndarray):
        g.add_node(detector, coords=coords)

    iter_flatten_model(model, handle_error=handle_error, handle_detector_coords=handle_detector_coords)

    return g


def detector_error_model_to_nx_graph(model: stim.DetectorErrorModel) -> 'nx.Graph':
    """Convert a stim error model into a NetworkX graph."""

    # Local import to reduce sinter's startup time.
    try:
        import networkx as nx
    except ImportError as ex:
        raise ImportError(
            "pymatching was installed without networkx?"
            "Run `pip install networkx`.\n"
        ) from ex

    g = nx.Graph()
    boundary_node = model.num_detectors
    g.add_node(boundary_node, is_boundary=True, coords=[-1, -1, -1])

    def handle_error(p: float, dets: List[int], frame_changes: List[int]):
        if p == 0:
            return
        if len(dets) == 0:
            # No symptoms for this error.
            # Code probably has distance 1.
            # Accept it and keep going, though of course decoding will probably perform terribly.
            return
        if len(dets) == 1:
            dets = [dets[0], boundary_node]
        if len(dets) > 2:
            raise NotImplementedError(
                f"Error with more than 2 symptoms can't become an edge or boundary edge: {dets!r}.")
        if g.has_edge(*dets):
            edge_data = g.get_edge_data(*dets)
            old_p = edge_data["error_probability"]
            old_frame_changes = edge_data["fault_ids"]
            # If frame changes differ, the code has distance 2; just keep whichever was first.
            if set(old_frame_changes) == set(frame_changes):
                p = p * (1 - old_p) + old_p * (1 - p)
                g.remove_edge(*dets)
        if p > 0.5:
            p = 1 - p
        if p > 0:
            g.add_edge(*dets, weight=math.log((1 - p) / p), fault_ids=frame_changes, error_probability=p)

    def handle_detector_coords(detector: int, coords: np.ndarray):
        g.add_node(detector, coords=coords)

    iter_flatten_model(model, handle_error=handle_error, handle_detector_coords=handle_detector_coords)

    return g


def filter_circ(circ, coords):
    """
    Filters out detectors of the circuit with given coordinate.
    Fails on REPEAT blocks.

    Parameters:
    - circ (stim.Circuit): circuit to filter
    - coords (int): Coordinate to remove (0: "blue", 1: "green", 2: "red")
    """
    new_circ = stim.Circuit()
    filtered_detectors = []
    ids = count()
    for inst in circ:
        if inst.name == "DETECTOR":
            i = next(ids)
            if len(inst.gate_args_copy()) == 0 or inst.gate_args_copy()[-1] != coords:
                new_circ.append(inst)
            else:
                new_circ.append_from_stim_program_text(f"DETECTOR({coords})")
                filtered_detectors.append(i)
        else:
            new_circ.append(inst)

    return new_circ, set(filtered_detectors)


def filter_circ2(circ, coords):
    """
    Filters out detectors of the circuit with given coordinate.
    Fails on REPEAT blocks.

    Parameters:
    - circ (stim.Circuit): circuit to filter
    - coords (int): Coordinate to remove (0: "blue", 1: "green", 2: "red")
    """
    new_circ = stim.Circuit()
    filtered_detectors = []
    ids = count()

    for inst in circ:
        if inst.name == "DETECTOR":
            i = next(ids)
            if len(inst.gate_args_copy()) == 1 and inst.gate_args_copy()[-1] == coords:
                new_circ.append(inst)
            else:
                new_circ.append_from_stim_program_text(f"DETECTOR({coords})")
                filtered_detectors.append(i)
        else:
            new_circ.append(inst)

    return new_circ, set(filtered_detectors)


def split_around_separator(targets):
    res = [[]]
    for target in targets:
        if target.is_separator():
            res.append([])
        elif target.is_relative_detector_id():
            res[-1].append(target)
    return res


def split_around_all(targets):
    res = [([], [])]
    for target in targets:
        if target.is_separator():
            res.append(([], []))
        elif target.is_relative_detector_id():
            res[-1][0].append(target)
        elif target.is_logical_observable_id():
            res[-1][1].append(target.val)
    return res


def filter_syndrome(sample, filtered_detectors):
    """
    Compute the restricted syndrome removing given color.

    Parameters:
    - sample (np.array): Syndrome to restrict
    - filtered_detectors (set): Detectors to forget
    """
    return np.array([j if i not in filtered_detectors else 0 for i, j in enumerate(sample)], dtype=bool)
