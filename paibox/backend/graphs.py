import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar, Union

from paicorelib import HwConfig

from paibox.base import NeuDyn
from paibox.collector import Collector
from paibox.components import FullConnectedSyn, InputProj, NeuModule
from paibox.exceptions import GraphBuildError, GraphConnectionError, NotSupportedError
from paibox.network import DynSysGroup
from paibox.utils import check_elem_unique

from .constrs import GraphNodeConstrs
from .context import _BACKEND_CONTEXT
from .graphs_types import *
from .placement import CoreBlock, neuron_repl_prop
from .routing import RoutingGroup
from .segment_utils import get_neu_segments


@dataclass
class PAIGraph:
    """Directed graph of PAIBox. We treat networks as one whole graph. \
        In the graph, synapses are edges while neurons are nodes.
    """

    _raw_networks: tuple[DynSysGroup, ...] = field(default_factory=tuple)
    """All networks are seen as one graph."""
    _raw_nodes: Collector[NodeName, NodeType] = field(default_factory=Collector)
    """Raw nodes in the networks."""
    _raw_edges: Collector[EdgeName, EdgeType] = field(default_factory=Collector)
    """Raw edges in the graph."""
    _raw_fmodules: Collector[NodeName, NeuModule] = field(default_factory=Collector)
    """Raw functional modules in the graph."""

    nodes: dict[NodeName, NodeAttr] = field(default_factory=dict)
    """General nodes in the graph."""
    edges: dict[EdgeName, EdgeAttr] = field(default_factory=dict)
    """General edges in the graph."""

    inodes: Collector[NodeName, SourceNodeType] = field(default_factory=Collector)
    """Input nodes in the graph."""
    onodes: Collector[NodeName, DestNodeType] = field(default_factory=Collector)
    """Output nodes in the graph."""

    ordered_nodes: list[NodeName] = field(default_factory=list)
    """Nodes in topological sort order."""

    succ_dg: dict[NodeName, dict[NodeName, EdgeAttr]] = field(default_factory=dict)
    """Successor edges & nodes of every node in the graph."""

    pred_dg: dict[NodeName, dict[NodeName, EdgeAttr]] = field(default_factory=dict)
    """Predecessor edges & nodes of every node in the graph."""

    degree_of_nodes: dict[NodeName, NodeDegree] = field(default_factory=dict)
    """A dictionary of in/out-degree tuple of nodes."""

    """Status options"""
    has_built: bool = field(default=False)

    # node_constrs: GraphNodeConstrs = field(default_factory=GraphNodeConstrs)

    def clear(self, total: bool = True) -> None:
        """Clear the PAIGraph."""
        self.has_built = False

        self.nodes.clear()
        self.edges.clear()
        self.inodes.clear()
        self.onodes.clear()
        self.ordered_nodes.clear()
        self.succ_dg.clear()
        self.pred_dg.clear()
        self.degree_of_nodes.clear()

        if total:
            self._raw_networks = ()
            self._raw_nodes.clear()
            self._raw_edges.clear()
            self._raw_fmodules.clear()

    def build(self, *networks: DynSysGroup, **build_options) -> None:
        self.clear()

        if not check_elem_unique(networks):
            raise GraphBuildError("duplicated networks are not allowed.")

        self._raw_networks = networks
        self._pre_build(**build_options)

        nodes: Collector[NodeName, NodeType] = Collector()
        edges: Collector[EdgeName, EdgeType] = Collector()
        fm: Collector[NodeName, NeuModule] = Collector()

        for subnet in self._raw_networks:
            fm += subnet.components.subset(NeuModule).unique()
            nodes += (
                subnet.components.include(InputProj, NeuDyn).exclude(NeuModule).unique()
            )
            edges += subnet.components.subset(FullConnectedSyn).unique()

        self._raw_nodes += nodes.val_on_condition(
            lambda node: not node.__gh_build_ignore__
        )
        self._raw_edges += edges.val_on_condition(
            lambda edge: not edge.__gh_build_ignore__
        )
        self._raw_fmodules = fm

        self._update_graph(**build_options)

    def _pre_build(self, **build_options) -> None:
        """Preprocessing before obtaining the topology."""
        # Build functional modules in the subnets
        for subnet in self._raw_networks:
            DynSysGroup.build_fmodule(subnet, **build_options)

    def _update_graph(self, **build_options) -> None:
        self.clear(total=False)

        # TODO Check isolated nodes in _raw_nodes
        for node in self._raw_nodes:
            self.succ_dg[node] = dict()
            self.pred_dg[node] = dict()

        for syn in self._raw_edges.values():
            u, v = syn.source.name, syn.dest.name
            if u not in self._raw_nodes:
                raise GraphConnectionError(
                    f"the source neuron {u} of {syn.name} is not included in the graph."
                )

            if v not in self._raw_nodes:
                raise GraphConnectionError(
                    f"the dest neuron {v} of {syn.name} is not included in the graph."
                )

            _edge_attr = EdgeAttr(edge=syn, distance=syn.source.delay_relative)
            self.succ_dg[u][v] = _edge_attr
            self.pred_dg[v][u] = _edge_attr

        self.degree_of_nodes = get_node_degrees(self.succ_dg)

        # `InputProj` nodes are input nodes definitely.
        self.inodes = self._raw_nodes.subset(InputProj)

        # By default, nodes with out-degree = 0 are considered as output nodes.
        self.onodes = self._raw_nodes.key_on_condition(
            lambda node: self.degree_of_nodes[node].out_degree == 0
        )  # type: ignore

        for name, node in self._raw_nodes.items():
            self.nodes[name] = NodeAttr(
                node=node,
                position=self._node_pos(name),
                degree=self.degree_of_nodes[name],
            )

        for name, syn in self._raw_edges.items():
            self.edges[name] = EdgeAttr(edge=syn, distance=syn.source.delay_relative)

        self.ordered_nodes = toposort(self.succ_dg)
        self.has_built = True

    def untwist_branch_nodes(self) -> None:
        # FIXME Input nodes may need to be excluded from the nodes to be traversed?
        for node_nn in filter(
            lambda node: self.degree_of_nodes[node].out_degree > 1,
            reversed(self.ordered_nodes),
        ):
            # succ_dg will be updated in _copy_node, so use the copy of succ_dg.
            for succ_nn in self.succ_dg[node_nn].copy():
                if self.degree_of_nodes[succ_nn].in_degree > 1:
                    node = self._raw_nodes[node_nn]
                    self._copy_node(
                        node, keep_pred_conn=True, grab_succ_nodes=succ_nn, update=False
                    )

        self._update_graph()

    def topo_support_check(self) -> None:
        _degree_check(self.degree_of_nodes, self.succ_dg)

        # Only support output nodes with <= 1152 neurons so far.
        if any(
            onode.num_out > HwConfig.N_FANIN_PER_DENDRITE_MAX
            for onode in self.onodes.values()
        ):
            raise NotSupportedError(
                f"only output nodes with no more than {HwConfig.N_FANIN_PER_DENDRITE_MAX} "
                f"neurons are supported."
            )

    def _node_pos(self, node: NodeName) -> NodePosition:
        if node in self.inodes:
            return NodePosition.INPUT
        elif node in self.onodes:
            return NodePosition.OUTPUT
        else:
            return NodePosition.MEMBER

    def build_check(self) -> None:
        if not self.has_built:
            raise GraphBuildError(f"the graph hasn't been built yet.")

    def group_edges(self) -> tuple[list[frozenset[EdgeType]], list[int]]:
        """Group all edges according to a certain rule.

        Return: a list of set of grouped edges and a list of routing groups id.
        """
        self.build_check()

        rgid = 0  # routing group id
        routing_groups_id: list[int] = []
        gathered: list[frozenset[EdgeType]] = []
        seen_edges: set[EdgeType] = set()  # Check if all edges are traversed

        for node in self.ordered_nodes:
            """Process the predecessor nodes of nodes first, then process the successor nodes."""
            if self.degree_of_nodes[node].in_degree > 1:
                edge_group = set(
                    [pred_nodes.edge for pred_nodes in self.pred_dg[node].values()]
                )
                # Get the edges traversed for the first time
                comming_edges = edge_group.difference(seen_edges)

                seen_edges.update(comming_edges)
                routing_groups_id.append(rgid)
                rgid += 1
                gathered.append(frozenset(comming_edges))

            if self.degree_of_nodes[node].out_degree > 1:
                """Consider the constraints to the nodes."""
                succ_edges = [e.edge for e in self.succ_dg[node].values()]
                succ_nodes = [self.nodes[n].node for n in self.succ_dg[node]]

                # Get the subgroup of indices, like [[0, 1], [2], [3, 4]]
                idx_of_sg = GraphNodeConstrs.tick_wait_attr_constr(succ_nodes)

                if len(idx_of_sg) > 0:
                    for idx in idx_of_sg:
                        succ_edges_sg = frozenset([succ_edges[i] for i in idx])
                        if succ_edges_sg not in gathered:
                            seen_edges.update(succ_edges_sg)
                            routing_groups_id.append(rgid)
                            gathered.append(succ_edges_sg)
                        else:
                            # FIXME Will this happen?
                            raise NotSupportedError
                else:
                    succ_edges_sg = frozenset(succ_edges)
                    if succ_edges_sg not in gathered:
                        seen_edges.update(succ_edges_sg)
                        routing_groups_id.append(rgid)
                        gathered.append(succ_edges_sg)
                    else:
                        # FIXME Will this happen?
                        raise NotSupportedError

                rgid += 1

            elif self.degree_of_nodes[node].out_degree == 1:
                succ_node = list(self.succ_dg[node].keys())[0]
                # Check the in-degree of the only following node.
                if self.degree_of_nodes[succ_node].in_degree == 1:
                    gathered.append(frozenset({self.succ_dg[node][succ_node].edge}))
                    routing_groups_id.append(rgid)
                    rgid += 1
                else:
                    # This edge is waiting to be processed when
                    # traversing the following node `succ_node`.
                    pass
            else:
                # out-degree = 0, do nothing.
                continue

        return gathered, routing_groups_id

    def multicast_optim(
        self,
        core_blocks: list[CoreBlock],
        routing_groups: list[RoutingGroup],
        optim_nodes: tuple[NodeName, ...] = (),
    ) -> bool:
        """Multicast optimization.

        NOTE: Only applies to a node that only has 2 successors, and they belong to the same core block.
        """
        ONLY_SUPPORT_N_SUCC = 2

        def _roundup_to_pow2(n: int) -> int:
            return 1 if n < 1 else 2 ** math.ceil(math.log(n, 2))

        is_optimized = False

        if optim_nodes == ():
            _optim_nodes = reversed(self.ordered_nodes)
        else:
            _optim_nodes = optim_nodes

        # visit ordered nodes for end to front
        for node_name in filter(lambda node: isinstance(node, NeuDyn), _optim_nodes):
            node = self._raw_nodes[node_name]

            succ_nn = list(self.succ_dg[node_name].keys())
            if len(succ_nn) != ONLY_SUPPORT_N_SUCC:
                continue

            succ_cbs = get_succ_cb_by_node(node, core_blocks)
            pred_cbs = get_pred_cb_by_node(node, core_blocks)

            # the node to be optimized can only has one successor core block & predecessor core block.
            if len(succ_cbs) != 1 or len(pred_cbs) != 1:
                continue

            succ_cb = succ_cbs[0]
            pred_cb = pred_cbs[0]

            if set(d.name for d in succ_cb.dest) != set(succ_nn):
                continue

            pred_rg = self._find_rg_by_cb(pred_cb, routing_groups)
            succ_rg = self._find_rg_by_cb(succ_cb, routing_groups)

            # The expected previous core block will add a new replicated node.
            pred_cb_dest = pred_cb.dest.copy()
            pred_cb_dest.append(node.copy())

            n_core_required_after_copy = len(
                get_neu_segments(
                    pred_cb_dest,
                    pred_cb.neuron_capacity,
                    neuron_repl_prop(pred_cb.n_weight_bits, pred_cb.n_timeslot),
                    _BACKEND_CONTEXT.cflags["grouping_optim_target"],
                )
            )
            pred_rg_n_core = pred_rg.n_core_required
            pred_rg_n_core_after_copy = (
                pred_rg_n_core - pred_cb.n_core_required + n_core_required_after_copy
            )

            n_core_after_split = [0] * ONLY_SUPPORT_N_SUCC
            for i in range(ONLY_SUPPORT_N_SUCC):
                dest = [self._raw_nodes[succ_nn[i]]]
                n_core_after_split[i] = len(
                    get_neu_segments(
                        dest,  # type: ignore
                        succ_cb.neuron_capacity,
                        neuron_repl_prop(succ_cb.n_weight_bits, succ_cb.n_timeslot),
                        _BACKEND_CONTEXT.cflags["grouping_optim_target"],
                    )
                )

            # 2^log2(#N of source rg) + 2^log2(#N of dest rg)
            n_core_before = _roundup_to_pow2(pred_rg_n_core) + _roundup_to_pow2(
                succ_rg.n_core_required
            )
            # 2^log2(#N of source rg after copy) + sum(2^log2(#N of dest rg[i]))
            n_core_after = _roundup_to_pow2(pred_rg_n_core_after_copy) + sum(
                _roundup_to_pow2(n) for n in n_core_after_split
            )

            # TODO actually here is: n_core_after < n_core_before
            if True:
                if not is_optimized:
                    is_optimized = True

                self._copy_node(node, keep_pred_conn=True, grab_succ_nodes=succ_nn[-1])

        return is_optimized

    def _copy_node(
        self,
        node: NodeType,
        *,
        keep_pred_conn: bool = False,
        keep_succ_conn: bool = False,
        grab_pred_nodes: Union[NodeName, Sequence[NodeName]] = (),
        grab_succ_nodes: Union[NodeName, Sequence[NodeName]] = (),
        update: bool = True,
    ) -> NodeType:
        def _copy_pred_conn(
            copied: DestNodeType, pred_nodes: dict[NodeName, EdgeAttr], orig_ind: int
        ) -> None:
            for pred_nn, pred_edge_attr in pred_nodes.items():
                copied_edge = pred_edge_attr.edge.copy(target=copied)
                self._raw_edges[copied_edge.name] = copied_edge
                # If don't _update_graph(), update partial information:
                # 1. Add the copied node & its incomming edges to succ_dg.
                # 2. Update the in-degree of copied node = in-degree of the original node.
                # 3. The out-degree of predecessors +1.
                # 1/2 -> A -> ...
                # ---
                # 1/2 -> A -> ...
                # 1/2 -> A'-> ...
                if not update:
                    copied_edge_attr = EdgeAttr(copied_edge, pred_edge_attr.distance)
                    self.succ_dg[pred_nn][copied.name] = copied_edge_attr
                    self.pred_dg[copied.name] = {pred_nn: copied_edge_attr}
                    self.degree_of_nodes[pred_nn].out_degree += 1

            if not update:
                self.degree_of_nodes[copied.name].in_degree = orig_ind

        def _copy_succ_conn(
            copied: DestNodeType, succ_nodes: dict[NodeName, EdgeAttr], orig_oud: int
        ) -> None:
            for succ_nn, succ_edge_attr in succ_nodes.items():
                copied_edge = succ_edge_attr.edge.copy(source=copied)
                self._raw_edges[copied_edge.name] = copied_edge
                # If don't _update_graph(), update partial information:
                # 1. Add the copied node & its outcoming edges to pred_nodes_dict & pred_dg.
                # 2. Update the out-degree of copied node = out-degree of the original node.
                # 3. The in-degree of successors +1.
                # ... -> A -> 1/2
                # ---
                # ... -> A -> 1/2
                # ... -> A'-> 1/2
                if not update:
                    copied_edge_attr = EdgeAttr(copied_edge, succ_edge_attr.distance)
                    self.pred_dg[succ_nn][copied.name] = copied_edge_attr
                    self.succ_dg[copied.name] = {succ_nn: copied_edge_attr}
                    self.degree_of_nodes[succ_nn].in_degree += 1

            if not update:
                self.degree_of_nodes[copied.name].out_degree = orig_oud

        pred_nodes = self.pred_dg[node.name]
        succ_nodes = self.succ_dg[node.name]

        copied: NeuDyn = node.copy()
        self._raw_nodes[copied.name] = copied

        if not update:
            self.degree_of_nodes[copied.name] = NodeDegree()

        if keep_pred_conn:
            orig_ind = self.degree_of_nodes[node.name].in_degree
            _copy_pred_conn(copied, pred_nodes, orig_ind)
        else:
            if isinstance(grab_pred_nodes, NodeName):
                grab_pred_nodes = (grab_pred_nodes,)

            if any(nn not in pred_nodes for nn in grab_pred_nodes):
                raise ValueError(
                    f"not all nodes in 'grab_pred_nodes' are in node {node.name}'s predecessors. "
                    f"Got {', '.join(grab_pred_nodes)}, but predecessors are {', '.join(pred_nodes)}."
                )
            else:
                for pred_nn in grab_pred_nodes:
                    pred_edge = pred_nodes[pred_nn].edge
                    pred_edge.target = copied
                    # If don't _update_graph(), update partial information:
                    # 1. Remove the original connection & add the copied node & the edge
                    # with the modified target to succ_nodes_dict & succ_dg.
                    # 2. Update the in-degree of copied node = len(grab).
                    # 3. The out-degree of predecessors keep the same.
                    # 1/2/3 -> A -> ...
                    # ---
                    # 1     -> A -> ...
                    # 2/3   -> A'-> ...
                    if not update:
                        _orig_edge_attr = self.succ_dg[pred_nn].pop(node.name)
                        new_edge_attr = EdgeAttr(pred_edge, _orig_edge_attr.distance)

                        self.succ_dg[pred_nn][copied.name] = new_edge_attr
                        self.pred_dg[copied.name] = dict()
                        self.pred_dg[copied.name][pred_nn] = new_edge_attr

                if not update:
                    self.degree_of_nodes[node.name].in_degree -= len(grab_pred_nodes)
                    self.degree_of_nodes[copied.name].in_degree = len(grab_pred_nodes)

        if keep_succ_conn:
            orig_oud = self.degree_of_nodes[node.name].out_degree
            _copy_succ_conn(copied, pred_nodes, orig_oud)
        else:
            if isinstance(grab_succ_nodes, NodeName):
                grab_succ_nodes = (grab_succ_nodes,)

            if any(nn not in succ_nodes for nn in grab_succ_nodes):
                raise ValueError(
                    f"not all nodes in 'grab_succ_nodes' are in node {node.name}'s successors."
                    f"Got {', '.join(grab_succ_nodes)}, but successors are {', '.join(succ_nodes)}."
                )
            else:
                for succ_nn in grab_succ_nodes:
                    succ_edge = succ_nodes[succ_nn].edge
                    succ_edge.source = copied
                    # If don't _update_graph(), update partial information:
                    # 1. Remove the original connection & add the copied node & the edge
                    # with the modified target to succ_dg.
                    # 2. Update the out-degree of copied node = len(grab).
                    # 3. The in-degree of successors keep the same.
                    # ... -> A -> 1/2/3
                    # ---
                    # ... -> A -> 1
                    # ... -> A'-> 2/3
                    if not update:
                        self.succ_dg[node.name].pop(succ_nn)
                        _orig_edge_attr = self.pred_dg[succ_nn].pop(node.name)
                        new_edge_attr = EdgeAttr(succ_edge, _orig_edge_attr.distance)
                        self.succ_dg[copied.name] = {succ_nn: new_edge_attr}
                        self.pred_dg[succ_nn][copied.name] = new_edge_attr
                        # self.pred_dg = reverse_edges2(self.succ_dg)

                if not update:
                    self.degree_of_nodes[node.name].out_degree -= len(grab_succ_nodes)
                    self.degree_of_nodes[copied.name].out_degree = len(grab_succ_nodes)

        if update:
            self._update_graph()

        return copied

    @staticmethod
    def _find_rg_by_cb(
        core_block: CoreBlock, routing_groups: list[RoutingGroup]
    ) -> RoutingGroup:
        """Find which routing group the target core block is in."""
        _rgs = [rg for rg in routing_groups if core_block in rg.core_blocks]

        if len(_rgs) != 1:
            raise GraphConnectionError(
                f"the core block can only be assigned to 1 routing group, but got {len(_rgs)}."
            )

        return routing_groups[0]

    @property
    def inherent_timestep(self) -> int:
        self.build_check()
        _, distance = get_longest_path(self.succ_dg, self.ordered_nodes)

        return distance

    @property
    def graph_name_repr(self) -> str:
        _prefix = "graph_of_"
        return _prefix + "_and_".join(network.name for network in self._raw_networks)


_NT = TypeVar("_NT", CoreBlock, NodeName)
_T = TypeVar("_T")


def _degree_check(
    degree_of_nodes: Mapping[_NT, NodeDegree], succ_dg: Mapping[_NT, Iterable[_NT]]
) -> None:
    """Filter out such network structure, which is currently not supported."""
    for node in filter(lambda node: degree_of_nodes[node].out_degree > 1, succ_dg):
        for succ_node in succ_dg[node]:
            if degree_of_nodes[succ_node].in_degree > 1:
                _node_repr = (
                    succ_node.name if isinstance(succ_node, CoreBlock) else succ_node
                )
                raise NotSupportedError(
                    f"If out-degree of a node is greater than 1, the in-degree of its sucessors must be 1. "
                    f"However, in-degree of {_node_repr} is {degree_of_nodes[succ_node].in_degree}."
                )


def convert2routing_groups(
    succ_dg_of_cb: dict[CoreBlock, list[CoreBlock]],
    degrees_of_cb: dict[CoreBlock, NodeDegree],
    input_core_blocks: dict[SourceNodeType, list[CoreBlock]],
) -> list[RoutingGroup]:
    ordered_core_blocks = toposort(succ_dg_of_cb)
    seen_cb = set()
    routing_groups = []
    succ_cb_gid_dict = defaultdict(list)

    # _degree_check(degrees_of_cb, succ_dg_of_cb)

    # After that, all input core blocks have been traversed.
    for input_cbs in input_core_blocks.values():
        # FIXME Temporary solution. This case should be solved first:
        # I1 -> A/B, I2 -> B/C.
        if not seen_cb.isdisjoint(input_cbs):
            if len(input_cbs) > 1:
                raise ValueError
            else:
                seen_cb.update(input_cbs)
                routing_groups.append(RoutingGroup(*input_cbs))

    for cb in ordered_core_blocks:
        # Check whether the core block has been traversed. This judgment condition is for
        # core blocks with out-degree = 1 & output core blocks (out-degree = 0).
        if cb not in seen_cb:
            seen_cb.add(cb)
            routing_groups.append(RoutingGroup(cb))

        # If out-degree > 1, group successor core blocks according to their routing id.
        if degrees_of_cb[cb].out_degree > 1:
            succ_cbs = succ_dg_of_cb[cb]
            seen_cb.update(succ_cbs)

            succ_cb_gid_dict.clear()
            for succ_cb in succ_cbs:
                if succ_cb._routing_id in succ_cb_gid_dict:
                    succ_cb_gid_dict[succ_cb._routing_id].append(succ_cb)
                else:
                    succ_cb_gid_dict[succ_cb._routing_id] = [succ_cb]

            for succ_cb in succ_cb_gid_dict.values():
                routing_groups.append(RoutingGroup(*succ_cb))

    return routing_groups


def toposort(directed_edges: Mapping[_NT, Iterable[_NT]]) -> list[_NT]:
    """
    Topological sort algorithm by Kahn [1]_.

    Complexity is O(nodes + vertices).

    Parameters
    ----------
    edges : dict
        dict of the form {a: {b, c}} where b and c depend on a

    Returns
    -------
    An ordered list of nodes that satisfy the dependencies of ``edges``

    Examples
    --------

    .. testcode::

       from nengo.utils.graphs import toposort

       print(toposort({1: {2, 3}, 2: {3}, 3: set()}))

    .. testoutput::

       [1, 2, 3]

    Notes
    -----
    Closely follows the wikipedia page [2]_.

    References
    ----------
    .. [1] Kahn, Arthur B. (1962), "Topological sorting of large networks",
       Communications of the ACM
    .. [2] https://en.wikipedia.org/wiki/Toposort#Algorithms
    """
    incoming_edges = reverse_edges(directed_edges)
    vertices = set(
        v for v in directed_edges if v not in incoming_edges or not incoming_edges[v]
    )
    ordered = []

    while vertices:
        n = vertices.pop()
        ordered.append(n)
        for m in directed_edges.get(n, ()):
            assert n in incoming_edges[m]
            incoming_edges[m].remove(n)
            if not incoming_edges[m]:
                vertices.add(m)

    if any(incoming_edges.get(v, None) for v in directed_edges):
        raise NotSupportedError("the graph with cycles is not supported.")

    return ordered


def reverse_edges(directed_edges: Mapping[_NT, Iterable[_NT]]) -> dict[_NT, list[_NT]]:
    """
    Reverses direction of dependence dict.

    Parameters
    ----------
    directed_edges : dict
        dict of the form {a: {b, c}, b: set(), c: set()} where b and c depend
        on a.

    Returns
    -------
    dict of the form {a: set(), b: {a}, c: {a}} where b and c depend on a.
    """
    reversed = {k: list() for k in directed_edges}
    for key in directed_edges:
        for val in directed_edges[key]:
            if key in reversed[val]:
                raise ValueError(f"edge {key} -> {val} is repeated.")

            reversed[val].append(key)

    return reversed


def reverse_edges2(
    directed_edges: Mapping[_NT, Mapping[_NT, _T]]
) -> dict[_NT, dict[_NT, _T]]:
    reversed = {k: dict() for k in directed_edges}
    for key in directed_edges:
        for val, edge in directed_edges[key].items():
            if key in reversed[val]:
                raise ValueError(f"edge {key} -> {val} is repeated.")

            reversed[val][key] = edge

    return reversed


# def _bounded_nodes_check(constrs: Sequence[Sequence[NeuDyn]]) -> None:
#     seen = set()

#     for bounded in constrs:
#         for node in bounded:
#             if node in seen:
#                 raise ValueError(f"Node {node} is repeated in the list of constraints.")

#             seen.add(node)


# def _bounded_by(node: NodeName, constrs: Sequence[Sequence[NeuDyn]]) -> list[NodeName]:
#     for constr in constrs:
#         for bounded_node in constr:
#             if node == bounded_node.name:
#                 return list(n.name for n in set(constr))

#     return []


# def _conflicted_by(
#     node: NodeName, constrs: dict[NodeName, Sequence[NeuDyn]]
# ) -> list[NodeName]:
#     """Find all the conflicted nodes of node.

#     Example: {"1": {"2", "3"}, "4": {"1"}}. For node 1, return ["2", "3", "4"].
#     """
#     c = set(constrs.get(node, []))

#     for k, v in constrs.items():
#         for conf_node in v:
#             if node == conf_node.name:
#                 c.add(k)

#     return list(n.name for n in c)


def get_node_degrees(
    succ_edges: Mapping[_NT, Union[Sequence[_NT], Mapping[_NT, Any]]]
) -> dict[_NT, NodeDegree]:
    degree = defaultdict(NodeDegree)
    in_degrees = defaultdict(int)
    out_degrees = defaultdict(int)

    for node, succ_nodes in succ_edges.items():
        out_degrees[node] = len(succ_nodes)

        for succ_node in succ_nodes:
            in_degrees[succ_node] += 1

    for node in succ_edges:
        degree[node] = NodeDegree(in_degrees[node], out_degrees[node])

    return degree


def get_longest_path(
    edges_with_d: dict[_NT, dict[_NT, EdgeAttr]],
    ordered_nodes: list[_NT],
) -> tuple[list[_NT], int]:
    """Get the longest path in the DAG.

    Args:
        - edges_with_d: a dictionary of directed edges with distance.
        - ordered_nodes: nodes in topological sorting order.

    Return:
        A tuple containing the longest path in the graph and its distance.
    """
    distances: dict[_NT, int] = {node: 0 for node in ordered_nodes}
    pred_nodes: dict[_NT, _NT] = dict()

    for node in ordered_nodes:
        for neighbor, edge_attr in edges_with_d[node].items():
            d = edge_attr.distance
            if distances[node] + d > distances[neighbor]:
                distances[neighbor] = distances[node] + d
                pred_nodes[neighbor] = node

    # When there are more than one output nodes
    # with same distance, choose the first one.
    node = max(
        filter(lambda node: len(edges_with_d[node]) == 0, distances),
        key=lambda node: distances.get(node, 0),
    )

    distance = distances[node]
    path = [node]

    # Construct the longest path by following the predecessors
    while path[-1] in pred_nodes:
        path.append(pred_nodes[path[-1]])

    path.reverse()
    return path, distance


MAX_DISTANCE = 999  # I don't like float('inf')


def get_shortest_path(
    edges_with_d: dict[_NT, dict[_NT, EdgeAttr]],
    ordered_nodes: list[_NT],
    input_nodes: list[_NT],
) -> tuple[list[_NT], int]:
    """Get the shortest path in the DAG.

    Args:
        - edges_with_d: a list of directed edges with distance.
        - ordered_nodes: nodes in topological sorting.
        - input_nodes: input nodes.

    Return: the shortest distance in the graph.
    """
    distances: dict[_NT, int] = defaultdict(lambda: MAX_DISTANCE)
    pred_nodes: dict[_NT, _NT] = dict()

    # set initial value for all inputs nodes. If there is no input node,
    # the first node after topological sorting will be used as the starting node.
    if input_nodes:
        for inode in input_nodes:
            distances[inode] = 0
    else:
        distances[ordered_nodes[0]] = 0

    for node in ordered_nodes:
        for neighbor, edge_attr in edges_with_d[node].items():
            d = edge_attr.distance
            if distances[node] + d < distances[neighbor]:
                distances[neighbor] = distances[node] + d
                pred_nodes[neighbor] = node

    # When there are more than one output nodes
    # with same distance, choose the first one.
    node = min(
        filter(lambda node: len(edges_with_d[node]) == 0, distances),
        key=lambda node: distances.get(node, 0),
    )

    distance = distances[node]
    path = [node]

    # Construct the shortest path by following the predecessors
    while path[-1] in pred_nodes:
        path.append(pred_nodes[path[-1]])

    path.reverse()
    return path, distance


def get_succ_cb_by_node(
    node: NodeType, core_blocks: Sequence[CoreBlock]
) -> list[CoreBlock]:
    return [cb for cb in core_blocks if node in cb.source]


def get_pred_cb_by_succ_cb(
    succ_cb: dict[CoreBlock, list[CoreBlock]]
) -> dict[CoreBlock, list[CoreBlock]]:
    return reverse_edges(succ_cb)


def get_pred_cb_by_node(
    node: NodeType, core_blocks: Sequence[CoreBlock]
) -> list[CoreBlock]:
    return [cb for cb in core_blocks if node in cb.dest]


def get_pred_dg_by_succ_dg(
    succ_dg: dict[NodeName, dict[NodeName, EdgeAttr]]
) -> dict[NodeName, dict[NodeName, EdgeAttr]]:
    return reverse_edges2(succ_dg)


def get_pred_nodes_by_succ_dg(
    node: NodeType, succ_dg: dict[NodeName, dict[NodeName, EdgeAttr]]
) -> list[NodeName]:
    pred_nodes = []

    for pred, succ_nodes in succ_dg.items():
        if node in succ_nodes:
            pred_nodes.append(pred)

    return pred_nodes
