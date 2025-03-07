from itertools import product, chain, takewhile, dropwhile, count, repeat, combinations
import networkx as nx
import numpy as np
import stim


from Pauli import P, PauliOperator, Stabilizer, X, Y, Z, support_to_PauliOperator
from tools import prod, rotate_to_place_first, translate_sparse, reindexed, rotate_left, inverse_dict, symp, couples
from copy import deepcopy

class StabilizerCode:

    def __init__(self, stabilizers, logical_operators, qubits, stab_relations={}):
        self.stabilizers = stabilizers
        self.logical_operators = logical_operators
        self.qubits = qubits
        self.stab_relations = stab_relations

    def shift_qubits(self, n, extend_to=None):
        self.qubits = [i + n for i in self.qubits]
        if extend_to is None:
            extend_to = max(self.qubits) + 1
        m = extend_to - max(self.qubits) - 1
        for i, stab in enumerate(self.iter_stabilizers()):
            stab.extend(m)
            self[i] = stab.translate(n)

        for basis in self.logical_operators:
            for i, log in enumerate(self.logical_operators[basis]):
                log = log.translate(n)
                log.extend(m)
                self.logical_operators[basis][i] = log

    # @classmethod
    # def from_file_checkmatrix(cls, filename):
    #     """
    #     Surface code from a check matrix stored in a file.
    #     File must be a JSON file which only include a list of the [list of X stabilizers under
    #     exhaustive characteristic function over the qubits [1,0,0,1,1,1], list of Z stabilizers under
    #     exhaustive characteristic function over the qubits [1,0,0,1,1,1]]
    #     """
    # 
    #     with open("filename" ,"r") as f:
    #         xstabs, zstabs = f.load()

    def __deepcopy__(self):
        return type(self)(deepcopy(self.stabilizers), deepcopy(self.logical_operators), deepcopy(self.qubits), deepcopy(self.stab_relations))

    def __copy__(self):
        return type(self)(deepcopy(self.stabilizers), deepcopy(self.logical_operators), deepcopy(self.qubits), deepcopy(self.stab_relations))

    def copy(self):
        return self.__copy__()

    @property
    def num_stabilizers(self):
        return len(self.stabilizers["X"]) + len(self.stabilizers["Z"])

    def __getitem__(self, index):
        """
        Getter on the stabilizers.
        """
        if index < 0:
            raise IndexError("Index must be positive")

        if index >= len(self.stabilizers["X"]):
            if index >= len(self.stabilizers["Z"]) + len(self.stabilizers["X"]):
                raise IndexError(f"""Index {index} is too high for this code {len(self.stabilizers["Z"]) + len(self.stabilizers["X"])}""")
            return self.stabilizers["Z"][index-len(self.stabilizers["X"])]
        return self.stabilizers["X"][index]

    def __setitem__(self, index, value):
        """
        Setter on the stabilizers.
        """
        if index < 0:
            raise IndexError("Index must be positive")

        if index >= len(self.stabilizers["X"]):
            if index >= len(self.stabilizers["Z"]) + len(self.stabilizers["X"]):
                raise IndexError(f"""Index {index} is too high for this code {len(self.stabilizers["Z"]) + len(self.stabilizers["X"])}""")
            self.stabilizers["Z"][index-len(self.stabilizers["X"])] = value
            return
        self.stabilizers["X"][index] = value


class SurfaceCode(StabilizerCode):

    def __init__(self, stabilizers, logical_operators, qubits, stab_relations={}):
        super.__init__(stabilizers, logical_operators, qubits, stab_relations)

    @classmethod
    def hex_code(self, d1, d2):
        """
        Hexagonal lattice toric code
        """
        assert d1 > 1
        assert d2 > 1

        # Qubits that are horizontally placed in horizontal or vertical position
        horiz = [[] for _ in range(d2)]
        verts = [[] for _ in range(d2)]

        for i in range(d2):
            for j in range(d1):
                horiz[i].append(i * 2 * d1 + 2 * j)
                horiz[i].append(i * 2 * d1 + 2 * j + 1)
                verts[i].append(i * d1 + j + 2 * d1 * d2)

        # Stabilizers associated to faces
        stabx = [Stabilizer([X(horiz[i][(2 * j + i % 2 + 1) % (2 * d1)]),
                             X(horiz[i][(2 * j + i % 2) % (2 * d1)]),
                             X(verts[i][j]),
                             X(horiz[(i + 1) % d2][(2 * j + i % 2 + (i % 2 == 0 and i + 1 == d2)) % (2 * d1)]),
                             X(horiz[(i + 1) % d2][(2 * j + i % 2 + 1 + (i % 2 == 0 and i + 1 == d2)) % (2 * d1)]),
                             X(verts[i][(j + 1) % d1])],
                            3 * d1 * d2) for (i, j) in product(range(d2), range(d1))]

        # Stabilizers associated to vertices
        stabz = [Stabilizer([P("Z", horiz[i][(2 * j + i % 2) % (2 * d1)]),
                             P("Z", horiz[i][(2 * j + i % 2 + 1) % (2 * d1)]),
                             P("Z", verts[(i - 1) % d2][(j + i % 2) % d1])], 3 * d1 * d2) for (i, j) in product(range(d2), range(d1))]

        stabz += [Stabilizer([P("Z", horiz[i][(2 * j + i % 2) % (2 * d1)]),
                              P("Z", horiz[i][(2 * j + i % 2 - 1) % (2 * d1)]),
                              P("Z", verts[i % d2][(j) % d1])], 3 * d1 * d2) for (i, j) in product(range(d2), range(d1))]

        # Non trivial closed paths are logical operators
        # We keep track of one representative for each non-equivalent logical operators
        logical_operators = {"X": [PauliOperator([P("X", horiz[0][i]) for i in range(2 * d1)],
                                                 3 * d1 * d2),
                                   PauliOperator([P("X", horiz[i][0]) for i in range(d2)] + [P("X", verts[i][0]) for i in range(d2)],
                                                 3 * d1 * d2)],
                             "Z": [PauliOperator([P("Z", horiz[i][-1]) for i in range(d2)],
                                                 3 * d1 * d2),
                                   PauliOperator([P("Z", verts[0][i]) for i in range(d1)],
                                                 3 * d1 * d2)]}

        return SurfaceCode({"X": stabx, "Z": stabz}, logical_operators, list(range(3 * d1 * d2)), stab_relations={"X": [(range(len(stabx)-1), [-1])], "Z":  [(range(len(stabz)-1), [-1])]})

    @classmethod
    def toric_code(cls, d1, d2):
        """
        Square lattice toric code
        """
        assert d1 > 1
        assert d2 > 1

        # Qubits that are vertical or horizontal in the drawing
        verts = [[] for _ in range(d2)]
        horiz = [[] for _ in range(d2)]

        for i in range(d2):
            for j in range(d1):
                verts[i].append(i * d1 + j)
                horiz[i].append(i * d1 + j + d1 * d2)

        # Stabilizers associated with faces
        stabx = [Stabilizer([P("X", verts[i//d1][i % d1]),
                             P("X", horiz[i//d1][i % d1]),
                             P("X", verts[(i//d1 + 1) % d2][i % d1]),
                             P("X", horiz[i//d1][(i+1) % d1])],
                            2 * d1 * d2) for i in range(d1 * d2)]

        # Stabilizers associated with vertices
        stabz = [Stabilizer([P("Z", horiz[i//d1][i % d1]),
                             P("Z", verts[(i//d1 + 1) % d2][(i-1) % d1]),
                             P("Z", horiz[(i//d1 + 1) % d2][i % d1]),
                             P("Z", verts[(i//d1 + 1) % d2][i % d1])],
                            2 * d1 * d2) for i in range(d1 * d2)]

        # Non trivial closed paths are logical operators
        # We keep track of one representative for each non-equivalent logical operators
        logical_operators = {"X": [PauliOperator([P("X", qb) for qb in verts[0]], 2 * d1 * d2),
                                   PauliOperator([P("X", qb) for qb in [horiz[i][0] for i in range(d2)]], 2 * d1 * d2)],
                             "Z": [PauliOperator([P("Z", qb) for qb in [verts[i][0] for i in range(d2)]], 2 * d1 * d2),
                                   PauliOperator([P("Z", qb) for qb in horiz[0]], 2 * d1 * d2)]}

        stabilizers = dict()
        stabilizers["X"] = stabx
        stabilizers["Z"] = stabz

        return cls(stabilizers, logical_operators, list(range(2 * d1 * d2)), stab_relations={"X": [(range(len(stabx) - 1), [-1])], "Z": [(range(len(stabz) - 1), [-1])]})

    @classmethod
    def cylinder(cls, dx, dz, big_dx=None):
        """
        Cylindrical fraction of a toric code
        """
        code = cls.toric_code(dx, dz)
        if big_dx is None:
            big_dx = dx
        to_remove = []
        for stab in code.iter_stabilizers("Z"):
            if stab.support_as_set & set((dx*dz + i*dx for i in range(dz))):
                to_remove.append(stab)
        for stab in to_remove[::-1]:
            code.stabilizers["Z"].remove(stab)

        for stab in code.iter_stabilizers():
            for i in (dx*dz + i*dx for i in range(dz)):
                stab.paulis[i] = P("I", i)
            foo = stab.support_as_set
            stab.order = [i for i in stab.order if i in foo]

        N = 2 * dx * dz
        newN = 2 * big_dx * dz
        mapping = dict((i, (i % (N // 2)) // dx * big_dx + (i % (N // 2)) % dx + (i // (N // 2)) * (newN // 2)) for i in range(len(code.qubits)))

        code.qubits = range(newN)

        for (i, stab) in enumerate(code.iter_stabilizers()):
            stab.extend(newN - N)
            code[i] = stab.translate(mapping=mapping)

        s = set(mapping.values()) - set((big_dx*dz + i*big_dx for i in range(dz)))
        for j in range(newN):
            if j in s:
                continue
            code.stabilizers["Z"].append(Stabilizer([Z(j)], len(code.qubits)))

        code.logical_operators["X"] = code.logical_operators["X"][:1]
        code.logical_operators["Z"] = code.logical_operators["Z"][:1]
        code.logical_operators["X"][0].extend(newN - N)
        code.logical_operators["Z"][0].extend(newN - N)
        code.logical_operators["X"][0] = code.logical_operators["X"][0].translate(mapping=mapping)
        code.logical_operators["Z"][0] = code.logical_operators["Z"][0].translate(mapping=mapping)
        code.stab_relations = {"X": [(range(len(code.stabilizers["X"]) - 1), [-1])], "Z": []}

        return code

    @classmethod
    def copy(cls):
        return cls(deepcopy(self.stabilizers), deepcopy(self.logical_operators), deepcopy(self.qubits), deepcopy(self.stab_relations))

    @property
    def num_stabilizers(self):
        return len(self.stabilizers["X"]) + len(self.stabilizers["Z"])

    def __getitem__(self, index):
        """
        Getter on the stabilizers.
        """
        if index < 0:
            raise IndexError("Index must be positive")

        if index >= len(self.stabilizers["X"]):
            if index >= len(self.stabilizers["Z"]) + len(self.stabilizers["X"]):
                raise IndexError(f"""Index {index} is too high for this code {len(self.stabilizers["Z"]) + len(self.stabilizers["X"])}""")
            return self.stabilizers["Z"][index-len(self.stabilizers["X"])]
        return self.stabilizers["X"][index]

    def __setitem__(self, index, value):
        """
        Setter on the stabilizers.
        """
        if index < 0:
            raise IndexError("Index must be positive")

        if index >= len(self.stabilizers["X"]):
            if index >= len(self.stabilizers["Z"]) + len(self.stabilizers["X"]):
                raise IndexError(f"""Index {index} is too high for this code {len(self.stabilizers["Z"]) + len(self.stabilizers["X"])}""")
            self.stabilizers["Z"][index-len(self.stabilizers["X"])] = value
            return
        self.stabilizers["X"][index] = value

    def check(self):
        """
        Check if the code is a surface code with at least two qubits.
        """
        # Make sure the product of all but one stabilizers is equal to the last one.
        for relation in self.stab_relations["X"]:
            assert prod((self.stabilizers["X"][i] for i in relation[0]),
                        start=PauliOperator(nb_qubits=len(self.qubits))) == prod((self.stabilizers["X"][i] for i in relation[1]),
                                                                                 start=PauliOperator(nb_qubits=len(self.qubits)))
        for relation in self.stab_relations["Z"]:
            assert prod((self.stabilizers["Z"][i] for i in relation[0]),
                        start=PauliOperator(nb_qubits=len(self.qubits))) == prod((self.stabilizers["Z"][i] for i in relation[1]),
                                                                                 start=PauliOperator(nb_qubits=len(self.qubits)))

        assert len(self.logical_operators["X"]) == len(self.logical_operators["Z"]) == len(self.stab_relations["X"]) + len(self.stab_relations["Z"]) + len(self.qubits) - len(self.stabilizers["X"]) - len(self.stabilizers["Z"])

        # Make sure the stabilizers properly commute
        for sx in chain(self.stabilizers["X"]):
            for sz in chain(self.stabilizers["Z"],
                            self.logical_operators["Z"]):
                try:
                    assert sx.commute(sz)
                except:
                    print(repr(sx), repr(sz))
                    assert sx.commute(sz)

        # Make sure the logical_operators anticommute with their homologue
        for (i, lx) in enumerate(self.logical_operators["X"]):
            for (j, sz) in enumerate(chain(self.logical_operators["Z"],
                                           self.stabilizers["Z"])):
                if i == j:
                    assert not lx.commute(sz)
                else:
                    assert lx.commute(sz)

    def stab_to_matrix(self):
        mat = np.zeros((len(self.stabilizers["Z"]) + len(self.stabilizers["X"]),
                        len(self.qubits)*2))

        for (i, stab) in enumerate(chain(self.iter_stabilizers("Z"), self.iter_stabilizers("X"))):
            mat[i, :] = stab.to_simplectic_array()

        return mat

    def is_logical_operator(self, operator):
        for s in self.iter_stabilizers():
            if not operator.commute(s):
                return False

        return True

    def iter_stabilizers(self, kind=None):
        if kind is None or kind == "X":
            for s in self.stabilizers["X"]:
                yield s

        if kind is None or kind == "Z":
            for s in self.stabilizers["Z"]:
                yield s

    def iter_logical_operators(self, kind=None):
        if kind is None or kind == "X":
            for log in self.logical_operators["X"]:
                yield log

        if kind is None or kind == "Z":
            for log in self.logical_operators["Z"]:
                yield log

    def symp(self):
        dests = []
        GX = nx.Graph()
        for sx1, sx2 in combinations(list(self.iter_stabilizers("X")), 2):
            if sx1.support_as_set & sx2.support_as_set:
                GX.add_edge(sx1, sx2, qb=next(iter(sx1.support_as_set & sx2.support_as_set)))

        paths = nx.single_source_shortest_path(GX, self.stabilizers["X"][-1])
        for s in self.iter_stabilizers("X"):
            nodes = paths[s]
            if len(nodes) > 1:
                dests.append(support_to_PauliOperator([GX[e[0]][e[1]]["qb"] for e in couples(nodes)], "Z", self.stabilizers["X"][-1].nb_qubits))

        GZ = nx.Graph()
        for sz1, sz2 in combinations(list(self.iter_stabilizers("Z")), 2):
            if sz1.support_as_set & sz2.support_as_set:
                GZ.add_edge(sz1, sz2, qb=next(iter(sz1.support_as_set & sz2.support_as_set)))

        paths = nx.single_source_shortest_path(GZ, self.stabilizers["Z"][-1])

        for s in self.iter_stabilizers("Z"):
            nodes = paths[s]
            if len(nodes) > 1:
                dests.append(support_to_PauliOperator([GZ[e[0]][e[1]]["qb"] for e in couples(nodes)], "X", self.stabilizers["Z"][-1].nb_qubits))

        foo = symp(list(chain(list(self.iter_stabilizers("X"))[:-1], list(self.iter_stabilizers("Z"))[:-1], self.iter_logical_operators("X"), dests, self.iter_logical_operators("Z"))))

        n = self[0].nb_qubits

        SX = foo[:n, :n]
        SZ = foo[:n, n:]
        DX = foo[n:, :n]
        DZ = foo[n:, n:]

        A = SX @ DZ.T + SZ @ DX.T

        B = np.linalg.inv(A)

        SX = B @ SX
        SZ = B @ SZ

        foo[:n, :n] = SX
        foo[:n, n:] = SZ

        for i, v in enumerate(DX @ DZ.T):
            for j, w in enumerate(v):
                if w == 1:
                    foo[i+n] += foo[j]

        self.logical_operators["X"] = [support_to_PauliOperator([j for j, v in enumerate(i[:n]) if v], "X", len(self.qubits)) for i in foo[n-2:n, :]]
        self.logical_operators["Z"] = [support_to_PauliOperator([j for j, v in enumerate(i[n:]) if v], "Z", len(self.qubits)) for i in foo[2*n-2:2*n, :]]
            
        
        return foo


    def dehn_twist(self, guide, auxiliary=None, to_avoid={}, force_cnot=[]):
        """
        guide: A Z (non self-intersecting) logical operator of the code
        auxiliary: One qubit within the Xoperator parallel to the guide,
        indicating the side on which the twist is performed
        """
        # assert self.is_logical_operator(support_to_PauliOperator(guide, "Z", len(self.qubits)))
        if not guide:
            return [], [], None
        guide_as_set, ordered_guide, ordered_pairs, auxiliary = self._compute_ordered_pairs(guide, auxiliary, to_avoid)
        # CNOT operations required for the dehn twist
        CNOTs, to_move = self._compute_CNOTs(guide_as_set, ordered_pairs)

        if force_cnot:
            print("Overriding CNOTs")
            CNOTs = force_cnot

        for stabx in self.stabilizers["X"]:
            for cnot in CNOTs:
                stabx.apply_CNOT_with_caution(*cnot)

        # Orders in Stabilizers touched by CNOTs need to be manually updated
        # Only those whose support changed need update.
        faces_other_kind = set()

        for stabz in self.stabilizers["Z"]:
            to_add = False
            for cnot in CNOTs:
                if stabz.apply_CNOT_with_caution(*cnot):
                    to_add = True
            if to_add:
                faces_other_kind.add(stabz)

        # The only X stabilizers affected by the CNOTs are those of ordered_pairs
        for (stab, deb, end, news) in zip(ordered_pairs, ordered_guide, ordered_guide[1:] + [ordered_guide[0]], to_move[1:] + to_move[:1]):
            stab.order[stab.order.index(deb)+1:stab.order.index(end)] = news

        # For the affected Z stabilizers, all is as if we pulled along the guide
        mapping_dict = dict(zip(ordered_guide, ordered_guide[-1:] + ordered_guide[:-1]))
        for stab in faces_other_kind:
            for (i, elem) in enumerate(stab.order):
                if elem in mapping_dict:
                    stab.order[i] = mapping_dict[elem]

        # Logical operators are also affected
        for log in chain(self.logical_operators["X"],
                         self.logical_operators["Z"]):
            for cnot in CNOTs:
                log.apply_CNOT(*cnot)

        return ordered_guide, CNOTs, auxiliary

    def _compute_ordered_pairs(self, guide, auxiliary, to_avoid={}):
        """
        Select the X stabilizers traversed by a the guide, starting
        with one containing auxiliary.
        """
        guide_as_set = set(guide)  # Speed up the next step

        # Select the X stabilizers traversed by the guide
        pairs = [stab for stab in self.stabilizers["X"]
                 if len(stab.support_as_set & guide_as_set) == 2]

        # Coherent orderings of the selected stabilizers and guide support
        ordered_pairs = []
        ordered_guide = []
        processed = set()  # Speed up the process

        if auxiliary is None:
            stab = pairs[0]
            while stab.order[0] not in guide_as_set or stab.order[1] in guide_as_set:
                rotate_left(stab.order)
            for qb in stab.order[1:]:
                if qb in to_avoid:
                    rotate_left(stab.order)
                    while stab.order[0] not in guide_as_set or stab.order[1] in guide_as_set:
                        rotate_left(stab.order)
                    for qb in stab.order[1:]:
                        if qb in guide_as_set:
                            auxiliary = stab.order[1]
                            break

                if qb in guide_as_set:
                    auxiliary = stab.order[1]
                    break

        if auxiliary is None:
            raise ValueError("Automated twist failed")

        # Used to ensure a traversal of a closed path
        first_target = guide[0]
        next_target = guide[1]
        for stab in pairs:
            # Select the first traversed stabilizer (containing the auxiliary)
            if auxiliary in stab.support:
                # Rotate the view of the traversed stabilizers
                # To place the next qubit of the guide in first place
                rotate_to_place_first(stab.order, auxiliary)
                while stab.order[0] not in guide:
                    stab.order[:0], stab.order[1:] = stab.order[-1:], stab.order[:-1]
                first_target = stab.order[0]

                # The next target is the other qubit also present in the guide
                next_target = next(dropwhile(lambda x: x not in guide,
                                             stab.order[1:]))
                ordered_guide.append(first_target)
                processed.add(stab)
                ordered_pairs.append(stab)
                break
        else:
            # No first traversed stabilizer exists, the auxiliary is wrong
            raise ValueError(f"Given guide {auxiliary} is not suitable")

        # Compute the traversal along the guide
        while next_target != first_target:
            for stab in pairs:
                if stab in processed:
                    continue

                if next_target in stab.support:
                    # Rotate the view of the traversed stabilizers
                    # To place the next qubit of the guide in first place
                    rotate_to_place_first(stab.order, next_target)
                    ordered_guide.append(next_target)

                    # The next target is the other qubit also present in the guide
                    next_target = next(dropwhile(lambda x: x not in guide, stab.order[1:]))
                    processed.add(stab)
                    ordered_pairs.append(stab)
                    break
            else:
                raise ValueError(f"Given guide {guide} is not suitable")

        return guide_as_set, ordered_guide, ordered_pairs, auxiliary

    def _compute_CNOTs(self, guide_as_set, ordered_pairs):
        """
        Compute the CNOTs required for the dehn twist
        """
        # Group of qubit that need to be moved one stabilizer further
        # along the ordered guide
        CNOTs = []
        to_move = []
        for stab in ordered_pairs:
            to_move.append([])
            other = next(iter(guide_as_set & set(stab.order[1:])))
            # One CNOT controlled on the first qubit of the stabilizer that is
            # also in guide to all the qubit of the stabilizer until the second
            # qubit also in guide is counter-clockwisely met.
            for qb in takewhile(lambda x: x not in guide_as_set, stab.order[1:]):
                CNOTs.append((stab.order[0], qb))

                # Targeted qubits are to be moved
                to_move[-1].append(qb)
        return CNOTs, to_move

    def excluding_support_stabilizers_coloring(self, excluding_support):
        """ 
        Compute a non-optimal coloring of the stabilizer
        to schedule a measure protocol
        """

        if excluding_support:
            raise NotImplementedError()
            if not isinstance(excluding_support, dict):
                graph = nx.Graph()
                
                for s in self.iter_stabilizers():
                    graph.add_node(s)
                    for node in graph:
                        if s.support_as_set & node.support_as_set:
                            graph.add_edge(s, node)
                excluding_support = inverse_dict(nx.greedy_color(graph))
        else:
            excluding_support = {0: list(self.iter_stabilizers())}

        return excluding_support

    def stim_startup(self, num_round, meas_basis, meas_noise=0.0, idle_noise=0.0, excluding_support=False, measure_count=None):
        if measure_count is None:
            self.measure_count = count()
        else:
            self.measure_count = measure_count
        for s in chain(self.iter_stabilizers(), self.iter_logical_operators()):
            s.last_measure_time = [None]
        excluding_support = self.excluding_support_stabilizers_coloring(excluding_support)
        circuit = stim.Circuit()
        circuit.append_operation("R", self.qubits)
        for (qb, bas) in enumerate(meas_basis):
            if bas == "I":
                continue
            log = self.logical_operators[bas][qb]
            log.measure(self.measure_count)
            circuit.append_operation("MPP", list(chain(*zip((stim.target_x(p.qubit) if p.kind == "X" else stim.target_y(p.qubit) if p.kind == "y" else stim.target_z(p.qubit) for p in log.paulis if p.kind != "I"), repeat(stim.target_combiner(), len(log.support)))))[:-1], [meas_noise])
            # if bas == "X":
            #     circuit.append_operation("CZ", list(chain(*((stim.target_rec(-1), p.qubit) for p in log.paulis if p.kind != "I"))))
            #     log.measure(self.measure_count)
            #     print(repr(log))
            #     circuit.append_operation("MPP", list(chain(*zip((stim.target_x(p.qubit) if p.kind == "X" else stim.target_y(p.qubit) if p.kind == "y" else stim.target_z(p.qubit) for p in log.paulis if p.kind != "I"), repeat(stim.target_combiner(), len(log.support)))))[:-1], [meas_noise])
        circuit.append_operation("TICK")
        for stabs in excluding_support.values():
            cum_support = set()
            for stab in stabs:
                cum_support |= stab.support_as_set
                stab.measure(self.measure_count)
                circuit.append_operation("MPP", list(chain(*zip((stim.target_x(p.qubit) if p.kind == "X" else stim.target_y(p.qubit) if p.kind == "y" else stim.target_z(p.qubit) for p in stab.paulis if p.kind != "I"), repeat(stim.target_combiner(), len(stab.support)))))[:-1], [meas_noise])

            if idle_noise:
                circuit.append_operation("DEPOLARIZE1",
                                         list(set(self.qubits) - cum_support),
                                         [idle_noise])

        circuit += self.stim_measure(num_round=num_round-1,
                                     meas_noise=meas_noise,
                                     idle_noise=idle_noise,
                                     excluding_support=excluding_support)

        return circuit

    def stim_measure(self, num_round, meas_noise, idle_noise, excluding_support=False):
        """
        A measurement step.

        Parameters:
        num_round: Number of rounds of measurement
        meas_noise: Error probability of a measurement error
        idle_noise: Depolarize error probability on idle qubit (used if excluding_support is not False)
        excluding_support: if True, greedily computes a protocol to measure the stabilizers, measuring only disjoint support stabilizers at the same time
        """
        circuit = stim.Circuit()
        if num_round == 0:
            return circuit
        excluding_support = self.excluding_support_stabilizers_coloring(excluding_support)
        for stabs in excluding_support.values():
            cum_support = set()
            for stab in stabs:
                cum_support |= stab.support_as_set
                last = stab.measure(self.measure_count)
                circuit.append_operation("MPP", list(chain(*zip((stim.target_x(p.qubit) if p.kind == "X" else stim.target_y(p.qubit) if p.kind == "y" else stim.target_z(p.qubit) for p in stab.paulis if p.kind != "I"), repeat(stim.target_combiner(), len(stab.support)))))[:-1], [meas_noise])
                circuit.append_operation("DETECTOR", [stim.target_rec(last - stab.last_measure_time[-1] - 1), stim.target_rec(-1)])

            if idle_noise:
                circuit.append_operation("DEPOLARIZE1",
                                         list(set(self.qubits) - cum_support),
                                         [idle_noise])
        if num_round > 1:
            self.measure_count = count(next(self.measure_count) + (num_round - 2) * (len(self.stabilizers["X"]) + len(self.stabilizers["Z"])))
            for stabs in excluding_support.values():
                for stab in stabs:
                    last = stab.measure(self.measure_count)

        return circuit * num_round

    def stim_startup_text(self, num_round, meas_basis, meas_noise=0.0, noise=0.0, excluding_support=False, measure_count=None, shift=0, init_basis="Z"):
        if measure_count is None:
            self.measure_count = count()
        else:
            self.measure_count = measure_count

        for s in chain(self.iter_stabilizers(), self.iter_logical_operators()):
            s.last_measure_time = [None]
        excluding_support = self.excluding_support_stabilizers_coloring(excluding_support)
        circuit = ""
        circuit += f"R{init_basis} " + " ".join(str(qb) for qb in self.qubits) + "\n"

        if isinstance(meas_basis, str):
            for (qb, bas) in enumerate(meas_basis):
                if bas == "I":
                    continue
                log = self.logical_operators[bas][qb]
                log.measure(self.measure_count)
                circuit += f"MPP(0.0) " + "*".join(p.kind + str(p.qubit) for p in log.paulis if p.kind != "I") + "\n"
                circuit += f"OBSERVABLE_INCLUDE({qb+shift}) rec[-1]\n"

        elif isinstance(meas_basis, list):
            for i, basis in enumerate(meas_basis):
                log = PauliOperator(None, nb_qubits=self[0].nb_qubits)
                for (qb, bas) in enumerate(basis):
                    print(bas)
                    if bas == "I":
                        continue
                    if bas == "Y":
                        log = log * self.logical_operators["Z"][qb]
                        bas = "X"
                    log = log * self.logical_operators[bas][qb]
                    print(log)

                # log.measure(self.measure_count)
                circuit += f"MPP(0.0) " + "*".join(p.kind + str(p.qubit) for p in log.paulis if p.kind != "I") + "\n"
                circuit += f"OBSERVABLE_INCLUDE({i+shift}) rec[-1]\n"

        else:
            raise TypeError(f"meas_basis argument should be of type str or list of str, not {type(meas_basis)} ({meas_basis})")

        for stabs in excluding_support.values():
            cum_support = set()
            for stab in stabs:
                cum_support |= stab.support_as_set
                stab.measure(self.measure_count)
                # circuit.append_operation("MPP", list(chain(*zip((stim.target_x(p.qubit) if p.kind == "X" else stim.target_y(p.qubit) if p.kind == "y" else stim.target_z(p.qubit) for p in stab.paulis if p.kind != "I"), repeat(stim.target_combiner(), len(stab.support)))))[:-1], [meas_noise])
                circuit += f"MPP({meas_noise}) " + "*".join(p.kind + str(p.qubit) for p in stab.paulis if p.kind != "I") + "\n"

                if (init_basis == "X" and stab.isX) or (init_basis == "Z" and stab.isZ):
                    circuit += "DETECTOR rec[-1]\n"

            # if idle_noise:
            #     circuit.append_operation("DEPOLARIZE1",
            #                              list(set(self.qubits) - cum_support),
            #                              [idle_noise])

        for i in range(num_round-1):
            circuit += self.stim_depolarize1_text(noise)
            circuit += self.stim_measure_text(num_round=1,
                                              meas_noise=meas_noise,
                                              noise=noise)

        return circuit

    def stim_measure(self, num_round, meas_noise, idle_noise, excluding_support=False):
        """
        A measurement step.

        Parameters:
        num_round: Number of rounds of measurement
        meas_noise: Error probability of a measurement error
        idle_noise: Depolarize error probability on idle qubit (used if excluding_support is not False)
        excluding_support: if True, greedily computes a protocol to measure the stabilizers, measuring only disjoint support stabilizers at the same time
        """
        circuit = stim.Circuit()
        if num_round == 0:
            return circuit
        excluding_support = self.excluding_support_stabilizers_coloring(excluding_support)
        for stabs in excluding_support.values():
            cum_support = set()
            for stab in stabs:
                cum_support |= stab.support_as_set
                last = stab.measure(self.measure_count)
                circuit.append_operation("MPP", list(chain(*zip((stim.target_x(p.qubit) if p.kind == "X" else stim.target_y(p.qubit) if p.kind == "y" else stim.target_z(p.qubit) for p in stab.paulis if p.kind != "I"), repeat(stim.target_combiner(), len(stab.support)))))[:-1], [meas_noise])
                circuit.append_operation("DETECTOR", [stim.target_rec(last - stab.last_measure_time[-1] - 1), stim.target_rec(-1)])

            if idle_noise:
                circuit.append_operation("DEPOLARIZE1",
                                         list(set(self.qubits) - cum_support),
                                         [idle_noise])
        if num_round > 1:
            self.measure_count = count(next(self.measure_count) + (num_round - 2) * (len(self.stabilizers["X"]) + len(self.stabilizers["Z"])))
            for stabs in excluding_support.values():
                for stab in stabs:
                    last = stab.measure(self.measure_count)

        return circuit * num_round

    def stim_measure_text(self, num_round, meas_noise, noise, excluding_support=False, detector_arg=None):
        """
        A measurement step.

        Parameters:
        num_round: Number of rounds of measurement
        meas_noise: Error probability of a measurement error
        noise: Depolarize error probability on qubits (used if excluding_support is not False)
        excluding_support: if True, greedily computes a protocol to measure the stabilizers, measuring only disjoint support stabilizers at the same time
        """
        circuit = ""
        if detector_arg is None:
            detector_arg = ""
        else:
            detector_arg = "(" + str(detector_arg) + ")"
        if num_round == 0:
            return circuit
        if num_round > 1:
            raise NotImplementedError
        for stab in self.iter_stabilizers():
            last = stab.measure(self.measure_count)
            # circuit.append_operation("MPP", list(chain(*zip((stim.target_x(p.qubit) if p.kind == "X" else stim.target_y(p.qubit) if p.kind == "y" else stim.target_z(p.qubit) for p in stab.paulis if p.kind != "I"), repeat(stim.target_combiner(), len(stab.support)))))[:-1], [meas_noise])
            circuit += f"MPP({meas_noise}) " + "*".join(p.kind + str(p.qubit) for p in stab.paulis if p.kind != "I") + "\n"
            # circuit.append_operation("DETECTOR", [stim.target_rec(last - stab.last_measure_time[-1] - 1), stim.target_rec(-1)])
            # circuit += f"DETECTOR({next(self.det)}) rec[{last - stab.last_measure_time[-1] - 1}] rec[-1]\n"
            circuit += f"DETECTOR{detector_arg} rec[{last - stab.last_measure_time[-1] - 1}] rec[-1]\n"

        return circuit

    def stim_measure_changing_support_text(self, num_round, transition, meas_noise, idle_noise, excluding_support=False):
        """
        A measurement step.

        Parameters:
        num_round: Number of rounds of measurement
        transition: Dictionary of stabilizers used to change support (simplification)
        meas_noise: Error probability of a measurement error
        idle_noise: Depolarize error probability on idle qubit (used if excluding_support is not False)
        excluding_support: if True, greedily computes a protocol to measure the stabilizers, measuring only disjoint support stabilizers at the same time
        """
        # self.check()
        circuit = ""
        lasts = {}
        if num_round == 0:
            return circuit
        num = count(0)
        for kind in ["X", "Z"]:
            for stab in self.stabilizers[kind]:
                last = stab.measure(self.measure_count)
                #circuit.append_operation("MPP", list(chain(*zip((stim.target_x(p.qubit) if p.kind == "X" else stim.target_y(p.qubit) if p.kind == "Y" else stim.target_z(p.qubit) for p in stab.paulis if p.kind != "I"), repeat(stim.target_combiner(), len(stab.support)))))[:-1], [meas_noise])
                circuit += f"MPP({meas_noise}) " + "*".join(p.kind + str(p.qubit) for p in stab.paulis if p.kind != "I") + "\n"

                lasts[stab] = last

        lastlast = stab.last_measure_time[-1]
        num = count(0)
        for kind in ["X", "Z"]:
            for stab in self.stabilizers[kind]:
                j = next(num)
                last = lasts[stab]
                if stab in transition:
                    # trans = []#stim.target_rec(s.last_measure_time[-1] - lastlast - 1) for s in transition[stab]]
                    # for s in transition[stab]:
                    #     trans.append(stim.target_rec(s.last_measure_time[-1] - lastlast - 1))
                    #     last_s = lasts[s]
                    #     trans.append(stim.target_rec(last_s - lastlast - 1))
                    trans = [s.last_measure_time[-1] - lastlast - 1 for s in transition[stab]]
                    # circuit.append_operation("DETECTOR", [stim.target_rec(stab.last_measure_time[-1] - lastlast - 1), stim.target_rec(last - lastlast - 1)] + trans, i)
                    circuit += f"DETECTOR({next(self.det)}) " + " ".join(f"rec[{foo}]" for foo in [stab.last_measure_time[-1] - lastlast - 1, last - lastlast - 1] + trans) + "\n"

                else:
                    #circuit.append_operation("DETECTOR", [stim.target_rec(stab.last_measure_time[-1] - lastlast - 1), stim.target_rec(last - lastlast - 1)], i)
                    circuit += f"DETECTOR({next(self.det)}) " + " ".join(f"rec[{foo}]" for foo in [stab.last_measure_time[-1] - lastlast - 1, last - lastlast - 1]) + "\n"

        for _ in range(num_round - 1):
            circuit += self.stim_measure_text(1, meas_noise, idle_noise, excluding_support)

        return circuit

    def stim_logical_measure(self, meas_basis, meas_noise=0.0, shift=0):
        circuit = stim.Circuit()

        for (qb, bas) in enumerate(meas_basis):
            log = self.logical_operators[bas][qb]
            last = log.measure(self.measure_count)
            circuit.append_operation("MPP", list(chain(*zip((stim.target_x(p.qubit) if p.kind == "X" else stim.target_z(p.qubit) for p in log.paulis if p.kind != "I"), repeat(stim.target_combiner(), len(log.support)))))[:-1], [meas_noise])

#             circuit.append_operation("MX", list(p.qubit for p in log.paulis if p.kind == "X"), [meas_noise])
#             circuit.append_operation("MZ", list(p.qubit for p in log.paulis if p.kind == "Z"), [meas_noise])

            circuit.append_operation("OBSERVABLE_INCLUDE",
                                     [stim.target_rec(-1), stim.target_rec(last - log.last_measure_time[-1] - 1)],
                                     [qb+1+shift])

        return circuit

    def stim_logical_measure_text(self, meas_basis, meas_noise=0.0, shift=0):
        circuit = ""

        for (qb, bas) in enumerate(meas_basis):
            log = self.logical_operators[bas][qb]
            last = log.measure(self.measure_count)
            # circuit.append_operation("MPP", list(chain(*zip((stim.target_x(p.qubit) if p.kind == "X" else stim.target_z(p.qubit) for p in log.paulis if p.kind != "I"), repeat(stim.target_combiner(), len(log.support)))))[:-1], [meas_noise])

#             circuit.append_operation("MX", list(p.qubit for p in log.paulis if p.kind == "X"), [meas_noise])
#             circuit.append_operation("MZ", list(p.qubit for p in log.paulis if p.kind == "Z"), [meas_noise])

            # circuit.append_operation("OBSERVABLE_INCLUDE",
            #                          [stim.target_rec(-1), stim.target_rec(last - log.last_measure_time[-1] - 1)],
            #                          [qb+1])
            circuit += f"MPP({meas_noise}) " + "*".join(p.kind + str(p.qubit) for p in log.paulis if p.kind != "I") + "\n"
            circuit += f"OBSERVABLE_INCLUDE({qb+shift}) rec[-1] rec[{last - log.last_measure_time[-1] - 1}]" + "\n"

        return circuit

    def stim_realistic_logical_measure_text(self, logical_operator, observable, meas_dict):
        circuit = ""
        to_measure = set()
        for p in logical_operator.paulis:
            if p.kind == "I":
                continue
            if p.qubit in meas_dict:
                if p.kind != meas_dict[p.qubit][0]:
                    raise ValueError(f"Impossible measure {p.kind}({p.qubit}), {meas_dict[p.qubit][0]}({p.qubit}) has already been measured.")
            else:
                to_measure.add(p)

        for qb in meas_dict:
            meas_dict[qb][1] -= len(to_measure)

        for i, p in enumerate(to_measure):
            next(self.measure_count)
            circuit += f"M{p.kind} {p.qubit} \n"
            meas_dict[p.qubit] = [p.kind, -len(to_measure)+i]

        circuit += f"OBSERVABLE_INCLUDE({observable}) " + " ".join(f"rec[{meas_dict[qb][1]}]" for qb in logical_operator.support) + "\n"

        return circuit

    def stim_realistic_Bell_logical_measure_text(self, logical_operator1, logical_operator2, observable, meas_dict, pairs=None):
        circuit = ""

        if pairs is None:
            assert len(logical_operator1.support) == len(logical_operator2.support)
            pairs = [tuple(sorted((s1, s2))) for s1, s2 in zip(logical_operator1.support, logical_operator2.support)]
        to_measure = set()
        for pair in pairs:
            if pair not in meas_dict:
                to_measure.add(PauliOperator([logical_operator1.paulis[pair[0]], logical_operator2.paulis[pair[1]]], logical_operator1.nb_qubits))
                continue

            foo = PauliOperator([logical_operator1.paulis[pair[0]], logical_operator2.paulis[pair[1]]], logical_operator1.nb_qubits)
            for m in meas_dict[pair]:
                if not foo.commute(m[0]):
                    raise ValueError(f"Impossible measure {logical_operator1.paulis[pair[0]].kind}({logical_operator1.paulis[pair[0]].qubit}){logical_operator2.paulis[pair[1]].kind}({logical_operator2.paulis[pair[1]].qubit}), {'*'.join(p.kind+'('+str(p.qubit)+')' for p in meas_dict[pair][0])} has already been measured.")

                if m[0] == foo:
                    break
            else:
                to_measure.add(foo)

        for qb in meas_dict:
            for m in meas_dict[qb]:
                m[1] -= len(to_measure)

        for i, op in enumerate(to_measure):
            next(self.measure_count)
            circuit += "MPP " + "*".join(p.kind + str(p.qubit) for p in op.paulis if p.kind != "I") + "\n"
            if tuple(sorted(op.support)) in meas_dict:
                meas_dict[tuple(sorted(op.support))].append([op, -len(to_measure)+i])
            else:
                meas_dict[tuple(sorted(op.support))] = [[op, -len(to_measure)+i]]

        recs = []
        for s1, s2 in zip(logical_operator1.support, logical_operator2.support):
            pair = tuple(sorted((s1, s2)))
            foo = PauliOperator([logical_operator1.paulis[s1], logical_operator2.paulis[s2]], logical_operator1.nb_qubits)
            for m in meas_dict[pair]:
                if m[0] == foo:
                    recs.append(m[1])
        if observable is not None:
            circuit += f"OBSERVABLE_INCLUDE({observable}) " + " ".join(f"rec[{rec}]" for rec in recs) + "\n"

        return circuit

    def stim_depolarize1(self, noise, qubits="all"):
        circuit = stim.Circuit()
        if qubits == "all":
            circuit.append_operation("DEPOLARIZE1", self.qubits, [noise])
        else:
            circuit.append_operation("DEPOLARIZE1", qubits, [noise])

        return circuit

    def stim_depolarize1_text(self, noise, qubits="all"):
        circuit = ""
        if qubits == "all":
            # circuit.append_operation("DEPOLARIZE1", self.qubits, [noise])
            circuit += f"DEPOLARIZE1({noise}) " + " ".join(str(qb) for qb in self.qubits) + "\n"

        else:
            # circuit.append_operation("DEPOLARIZE1", qubits, [noise])
            circuit += "DEPOLARIZE1({noise}) " + " ".join(str(qb) for qb in qubits) + "\n"

        return circuit

    def stim_x_error_text(self, noise, qubits="all"):
        circuit = ""
        if qubits == "all":
            # circuit.append_operation("DEPOLARIZE1", self.qubits, [noise])
            circuit += f"DEPOLARIZE1({noise}) " + " ".join(str(qb) for qb in self.qubits) + "\n"
        else:
            # circuit.append_operation("DEPOLARIZE1", qubits, [noise])
            circuit += "X_ERROR({noise}) " + " ".join(str(qb) for qb in qubits) + "\n"

        return circuit

def pretty_print_operators(ops):
    """
    Pretty-print an iterable of PauliOperators

    Parameters;
    ops: Iterable of PauliOperators
    """
    print()
    for op in ops:
        print(PauliOperator.__repr__(op), op.support_as_set)
    print()


def pretty_print_stabilizers(stabs):
    """
    Pretty-print an iterable of Stabilizers

    Parameters;
    stabs: Iterable of Stabilizers
    """
    print()
    for op in stabs:
        print(Stabilizer.__repr__(op))
    print()


if __name__ == "__main__":
    c = SurfaceCode.cylinder(3, 5, 5)
    pretty_print_operators(c.iter_stabilizers())
    pretty_print_operators(c.iter_logical_operators())
    #foo = c.symp()

    c.check()
