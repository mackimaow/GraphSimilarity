from enum import Enum
import cmath
import typing as tp
import cmath

from graph_similarity import Graph, StochasticGraphSimiliarity


def test_stochastic_graph_similiarity_on_chemical_compounds():
    
    class Atom(Enum):
        OXYGEN = 8
        VANADIUM = 23
    
    BOND_LENGTH = float
    
    graph_1 = Graph[Atom, BOND_LENGTH].create(
        (
            *[Atom.VANADIUM] * 4,  # 0 - 3
            *[Atom.OXYGEN] * 14  # 4 - 17
        ),
        {
            (0, 4): 1,  # everything has a bond length of 1
            (0, 5): 1,
            (0, 6): 1,
            (0, 7): 1,
            (1, 7): 1,
            (1, 8): 1,
            (1, 9): 1,
            (1, 10): 1,
            (2, 10): 1,
            (2, 11): 1,
            (2, 12): 1,
            (2, 13): 1,
            (2, 14): 1,
            (3, 14): 1,
            (3, 15): 1,
            (3, 16): 1,
            (3, 17): 1,
        }
    )
    
    # same graph as above but changed the last oxygen to vanadium
    graph_2 = Graph[Atom, BOND_LENGTH].create(
        (
            *[Atom.VANADIUM] * 5,  # 0 - 4
            *[Atom.OXYGEN] * 13  # 5 - 17
        ),
        {
            (0, 5): 1,  # everything has a bond length of 1
            (0, 6): 1,
            (0, 7): 1,
            (0, 8): 1,
            (1, 8): 1,
            (1, 9): 1,
            (1, 10): 1,
            (1, 11): 1,
            (2, 11): 1,
            (2, 12): 1,
            (2, 13): 1,
            (2, 14): 1,
            (2, 15): 1,
            (3, 15): 1,
            (3, 16): 1,
            (3, 17): 1,
            (3, 4): 1,
        }
    )
    
    # same graph as above but missing the last 2 oxygen
    graph_3 = Graph[Atom, BOND_LENGTH].create(
        (
            *[Atom.VANADIUM] * 4,  # 0 - 3
            *[Atom.OXYGEN] * 11  # 4 - 14
        ),
        {
            (0, 4): 1,  # everything has a bond length of 1
            (0, 5): 1,
            (0, 6): 1,
            (0, 7): 1,
            (1, 7): 1,
            (1, 8): 1,
            (1, 9): 1,
            (1, 10): 1,
            (2, 10): 1,
            (2, 11): 1,
            (2, 12): 1,
            (2, 13): 1,
            (2, 14): 1,
            (3, 14): 1,
        }
    )
    
    # same as graph 1 but has a loop
    graph_4 = Graph[Atom, BOND_LENGTH].create(
        (
            *[Atom.VANADIUM] * 4,  # 0 - 3
            *[Atom.OXYGEN] * 14  # 4 - 17
        ),
        {
            (0, 4): 1,  # everything has a bond length of 1
            (0, 5): 1,
            (0, 6): 1,
            (0, 7): 1,
            (1, 7): 1,
            (1, 8): 1,
            (1, 9): 1,
            (1, 10): 1,
            (2, 10): 1,
            (2, 11): 1,
            (2, 12): 1,
            (2, 13): 1,
            (2, 14): 1,
            (3, 14): 1,
            (3, 15): 1,
            (3, 16): 1,
            (3, 17): 1,
            (0, 3): 1, # loop to beginning
        }
    )
    
    
    # ---- making similiarity metric ----
    
    # this is used to add extra values to the nodes that we may want as we
    # do while doing graph reductions within the metric computation.
    # Here we are keeping track of how many cycles and nodes were merged.
    MERGED_CYCLES = int
    MERGED_NODES = int
    
    NODE_METRIC_TYPE = tp.Tuple[complex, MERGED_CYCLES, MERGED_NODES]
    
    def initialize_nodes(
        characteristic: int,
        number_of_characteristics: int,
        atom: Atom
    ) -> NODE_METRIC_TYPE:
        if atom == Atom.OXYGEN:
            return (1.0 + 0j, 1, 1)
        else:
            return (2.0 + 0j, 1, 1)
    
    # we do the same for edges
    EDGE_METRIC_TYPE = tp.Tuple[complex, MERGED_CYCLES]
    def initialize_edges(
        characteristic: int,
        number_of_characteristics: int,
        node_1: tp.Tuple[Atom, NODE_METRIC_TYPE],
        node_2: tp.Tuple[Atom, NODE_METRIC_TYPE],
        edge: BOND_LENGTH
    ) -> EDGE_METRIC_TYPE:
        return (edge + 0j, 1)
    
    # now we create a function computes when two nodes merge
    # for some characteristic function index
    
    CHARACTERISTIC_INDEX = int
    
    def merge_nodes(
        n: CHARACTERISTIC_INDEX,
        num_characteristics: int,
        node_1: NODE_METRIC_TYPE,
        node_2: NODE_METRIC_TYPE,
        edge_1_to_2: EDGE_METRIC_TYPE
    ) -> NODE_METRIC_TYPE:
        (value_1, merged_cycles_1, merged_nodes_1) = node_1
        (value_2, merged_cycles_2, merged_nodes_2) = node_2
        (edge_value, edge_cycles) = edge_1_to_2
        
        value_1_weight = merged_cycles_1 / (
            merged_cycles_1 + merged_cycles_2
        ) * cmath.exp(
            (cmath.pi * 1j) * (1.0 * n / num_characteristics)
        )
        
        value_2_weight = merged_cycles_2 / (
            merged_cycles_1 + merged_cycles_2
        ) * cmath.exp(
            - (cmath.pi * 1j) * (1.0 * n / num_characteristics)
        )
        
        new_value = cmath.sqrt(
            value_1_weight * value_1 ** 2 + value_2_weight * value_2 ** 2
        ) + edge_value
        
        return (
            new_value,
            merged_cycles_1 + merged_cycles_2 + edge_cycles,
            merged_nodes_1 + merged_nodes_2
        )
    
    # do the same for edges
    
    
    #              base node (b)                              base node (b)                               
    #                  /\                                          .
    #                 /  \                merge edges              |
    #      edge_b_1  /    \  edge_b_2        --->                  |  merged edge
    #               /      \                                       |  
    #              /        \                                      |
    #     node 1  .__________.  node 2                             . 
    #             edge 1 to 2                               merged node 1 2
    
    def merge_edges(
        n: CHARACTERISTIC_INDEX,
        num_characteristics: int,
        base_node: NODE_METRIC_TYPE,
        base_to_1_and_edge: tp.Tuple[NODE_METRIC_TYPE, EDGE_METRIC_TYPE],
        base_to_2_and_edge: tp.Tuple[NODE_METRIC_TYPE, EDGE_METRIC_TYPE],
        edge_1_to_2: EDGE_METRIC_TYPE,
        merged_node_1_2: NODE_METRIC_TYPE
    ) -> EDGE_METRIC_TYPE:
        edge_b_1 = base_to_1_and_edge[1]
        edge_b_2 = base_to_2_and_edge[1]
        edge_b_1_value, num_cyles_1 = edge_b_1
        edge_b_2_value, num_cyles_2 = edge_b_2
        
        edge_1_weight = cmath.exp(
            (cmath.pi * 1j) * (1.0 * n / num_characteristics)
        )
        edge_2_weight = cmath.exp(
            -(cmath.pi * 1j) * (1.0 * n / num_characteristics)
        )
        
        new_value = cmath.sqrt(
            edge_b_1_value * edge_1_weight
            + edge_b_2_value * edge_2_weight
        )
        
        return (
            new_value,
            num_cyles_1 + num_cyles_2
        )
    
    # finally after all the merging is done,
    # we are left with 1 node of the graph which we must
    # extract a value called the characteristic value
    
    CHARACTERISTIC_TYPE = complex
    
    def calc_characteristic(
        x: CHARACTERISTIC_INDEX,
        num_characteristics: int,
        final_node: NODE_METRIC_TYPE
    ) -> CHARACTERISTIC_TYPE:
        return final_node[0]
    
    # we need to also specify how to average our characteristic values.
    # This is easy in this example since our characteristic is just an
    # float
    
    def calc_average(
        characteristic_values: tp.List[CHARACTERISTIC_TYPE]
    ) -> CHARACTERISTIC_TYPE:
        return sum(characteristic_values) / len(characteristic_values)
    
    # we need to specify how to compare two different characteristic list
    # represent two graph respectively. Here we do a simple root mean square
    
    FINAL_OUTPUT = float
    
    def compare_chars(
        characteristic_values_graph_1: tp.List[CHARACTERISTIC_TYPE],
        characteristic_values_graph_2: tp.List[CHARACTERISTIC_TYPE]
    ) -> FINAL_OUTPUT:
        return abs(cmath.sqrt(sum(
            (char_1 - char_2) ** 2
            for char_1, char_2 in zip(
                characteristic_values_graph_1,
                characteristic_values_graph_2
            )    
        )))
    
    # some times we need to specify what a characteric value would be if the graph
    # is empty. For this we can just put zero
    
    EMPTY_GRAPH_CHARACTERISTIC_VALUE = 0.0
    
    # now we can make our metric to use on any graph:
    metric = StochasticGraphSimiliarity(
        initialize_nodes,
        initialize_edges,
        merge_nodes,
        merge_edges,
        calc_characteristic,
        calc_average,
        compare_chars,
        EMPTY_GRAPH_CHARACTERISTIC_VALUE
    )
    
    # test 1: run on graph 1 twice (hopefully its zero):
    similiarity = metric.compare(graph_1, graph_1) 
    print(f"Simliarity metric on graph1 on itself: {similiarity}")
    
    # test 2: run on graph 2 twice (hopefully its zero):
    similiarity = metric.compare(graph_2, graph_2) 
    print(f"Simliarity metric on graph2 on itself: {similiarity}")
    
    # test 3: run on graph 3 twice (hopefully its zero):
    similiarity = metric.compare(graph_3, graph_3) 
    print(f"Simliarity metric on graph3 on itself: {similiarity}")

    # test 4: run on graph 4 twice (hopefully its zero):
    similiarity = metric.compare(graph_4, graph_4) 
    print(f"Simliarity metric on graph4 on graph4: {similiarity}")    
    
    # test 5: run on the graph 1 and 2:
    similiarity = metric.compare(graph_1, graph_2) 
    print(f"Simliarity metric on graph1 on graph2: {similiarity}")
    
    # test 6: run on the graph 1 and 3:
    similiarity = metric.compare(graph_1, graph_3) 
    print(f"Simliarity metric on graph1 on graph3: {similiarity}")
    
    # test 7: run on the graph 2 and 3:
    similiarity = metric.compare(graph_2, graph_3) 
    print(f"Simliarity metric on graph2 on graph3: {similiarity}")
    
    # test 8: run on the graph 1 and 4:
    similiarity = metric.compare(graph_1, graph_4) 
    print(f"Simliarity metric on graph1 on graph4: {similiarity}")
