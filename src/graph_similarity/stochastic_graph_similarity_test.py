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
    
    # graph with unique ids set
    graph_0 = Graph.create(
        nodes={
            193: Atom.OXYGEN,
            196: Atom.OXYGEN,
            200: Atom.OXYGEN,
            203: Atom.OXYGEN,
            207: Atom.OXYGEN,
            210: Atom.OXYGEN,
            214: Atom.OXYGEN,
            217: Atom.OXYGEN,
            11: Atom.OXYGEN,
            23: Atom.OXYGEN,
            30: Atom.OXYGEN,
            35: Atom.OXYGEN,
            47: Atom.OXYGEN,
            83: Atom.OXYGEN,
            95: Atom.OXYGEN,
            107: Atom.OXYGEN,
            131: Atom.OXYGEN,
            137: Atom.OXYGEN,
            143: Atom.OXYGEN,
            194: Atom.OXYGEN,
            197: Atom.OXYGEN,
            199: Atom.OXYGEN,
            201: Atom.OXYGEN,
            202: Atom.OXYGEN,
            204: Atom.OXYGEN,
            205: Atom.OXYGEN,
            206: Atom.OXYGEN,
            208: Atom.OXYGEN,
            209: Atom.OXYGEN,
            211: Atom.OXYGEN,
            212: Atom.OXYGEN,
            213: Atom.OXYGEN,
            215: Atom.OXYGEN,
            218: Atom.OXYGEN,
            219: Atom.OXYGEN,
        },
        edges={
            (193, 201): 1.8252426389586287,
            (193, 202): 1.788684790694484,
            (193, 218): 1.8195464398872772,
            (193, 219): 1.5841212046212758,
            (201, 193): 1.8252426389586287,
            (201, 200): 1.7903377481308678,
            (202, 193): 1.788684790694484,
            (202, 203): 1.8165483627525514,
            (218, 193): 1.8195464398872772,
            (218, 214): 1.8021995796561434,
            (219, 193): 1.5841212046212758,
            (196, 107): 1.5926202940624583,
            (196, 137): 1.750083876759335,
            (196, 143): 1.782656025334686,
            (196, 197): 1.8173867293890706,
            (107, 196): 1.5926202940624583,
            (137, 196): 1.750083876759335,
            (143, 196): 1.782656025334686,
            (197, 196): 1.8173867293890706,
            (197, 217): 1.7698676670880205,
            (200, 23): 1.766910039542031,
            (200, 35): 1.617732255968037,
            (200, 201): 1.7903377481308678,
            (200, 211): 1.7339878174852728,
            (23, 200): 1.766910039542031,
            (35, 200): 1.617732255968037,
            (35, 203): 2.2527015403432844,
            (211, 200): 1.7339878174852728,
            (203, 30): 1.9920325363885332,
            (203, 35): 2.2527015403432844,
            (203, 131): 2.0056513152886226,
            (203, 199): 1.6865294977617489,
            (203, 202): 1.8165483627525514,
            (203, 205): 1.7479215123775147,
            (30, 203): 1.9920325363885332,
            (131, 203): 2.0056513152886226,
            (131, 214): 1.8918293935267383,
            (199, 203): 1.6865294977617489,
            (199, 210): 1.9342735990137783,
            (205, 203): 1.7479215123775147,
            (207, 11): 1.7574124574198957,
            (207, 206): 1.8195765804507003,
            (207, 208): 1.768080313782476,
            (207, 209): 1.5912810350565199,
            (11, 207): 1.7574124574198957,
            (206, 207): 1.8195765804507003,
            (206, 210): 1.7600976965693529,
            (208, 207): 1.768080313782476,
            (209, 207): 1.5912810350565199,
            (210, 47): 1.8110999062409023,
            (210, 95): 1.815532710374183,
            (210, 199): 1.9342735990137783,
            (210, 206): 1.7600976965693529,
            (210, 212): 1.7453589001583016,
            (47, 210): 1.8110999062409023,
            (95, 210): 1.815532710374183,
            (212, 210): 1.7453589001583016,
            (214, 83): 2.0159497045816877,
            (214, 131): 1.8918293935267383,
            (214, 204): 1.943494503113588,
            (214, 215): 1.5808480622436185,
            (214, 218): 1.8021995796561434,
            (83, 214): 2.0159497045816877,
            (204, 214): 1.943494503113588,
            (204, 217): 1.7948111258661488,
            (215, 214): 1.5808480622436185,
            (217, 194): 1.7881295308228207,
            (217, 197): 1.7698676670880205,
            (217, 204): 1.7948111258661488,
            (217, 213): 1.5955665080467436,
            (194, 217): 1.7881295308228207,
            (213, 217): 1.5955665080467436,
        }
    )
    
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
    
    # same as graph 1 but is disjoint in the middle
    graph_5 = Graph[Atom, BOND_LENGTH].create(
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
            (1, 10): 1, # disjoint right here
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

    # same as graph 1 but linearized (each node has at most two edges)
    graph_6 = Graph[Atom, BOND_LENGTH].create(
        (
            *[Atom.VANADIUM] * 4,  # 0 - 3
            *[Atom.OXYGEN] * 14  # 4 - 17
        ),
        {
            (0, 4): 1,  # everything has a bond length of 1
            (4, 5): 1,
            (5, 6): 1,
            (6, 7): 1,
            (7, 1): 1,
            (1, 8): 1,
            (8, 9): 1,
            (9, 10): 1,
            (10, 2): 1,
            (2, 11): 1,
            (11, 12): 1,
            (12, 13): 1,
            (13, 14): 1,
            (14, 3): 1,
            (3, 15): 1,
            (15, 16): 1,
            (16, 17): 1,
        }
    )
    
    # same as graph 1 but everything is oxygen
    graph_7 = Graph[Atom, BOND_LENGTH].create(
        tuple(
            [Atom.OXYGEN] * 18  # 0 - 17
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
    
    # same as graph 1 but everything is vanadium
    graph_8 = Graph[Atom, BOND_LENGTH].create(
        tuple(
            [Atom.VANADIUM] * 18  # 0 - 17
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
    
    # same as graph 1 except 1 vanadium atom switches places with 1 oxygen
    graph_9 = Graph[Atom, BOND_LENGTH].create(
        (
            *[Atom.VANADIUM] * 4,  # 0 - 3
            *[Atom.OXYGEN] * 14  # 4 - 17
        ),
        {
            (12, 4): 1,  # everything has a bond length of 1
            (12, 5): 1,
            (12, 6): 1,
            (12, 7): 1,
            (1, 7): 1,
            (1, 8): 1,
            (1, 9): 1,
            (1, 10): 1,
            (2, 10): 1,
            (2, 11): 1,
            (2, 0): 1,  # switched vandanium 0 with oxygen 12
            (2, 13): 1,
            (2, 14): 1,
            (3, 14): 1,
            (3, 15): 1,
            (3, 16): 1,
            (3, 17): 1,
        }
    )
    
    
    # ---- making similiarity metric ----
    
    # this is used to add extra values to the nodes that we may want as we
    # do while doing graph reductions within the metric computation.
    # Here we are keeping track of how many cycles and nodes were merged.
    MergedCycles = int
    MergedNodes = int
    
    NodeMetricType = tp.Tuple[complex, MergedCycles, MergedNodes]
    
    def initialize_nodes(
        characteristic: int,
        number_of_characteristics: int,
        atom: Atom
    ) -> NodeMetricType:
        if atom == Atom.OXYGEN:
            return (1.0 + 0j, 1, 1)
        else:
            return (2.0 + 0j, 1, 1)
    
    # we do the same for edges
    EdgeMetricType = tp.Tuple[complex, MergedCycles]
    def initialize_edges(
        characteristic: int,
        number_of_characteristics: int,
        node_1: tp.Tuple[Atom, NodeMetricType],
        node_2: tp.Tuple[Atom, NodeMetricType],
        edge: BOND_LENGTH
    ) -> EdgeMetricType:
        return (edge + 0j, 1)
    
    # now we create a function computes when two nodes merge
    # for some characteristic function index
    
    CharacteristicIndex = int
    
    def merge_nodes(
        n: CharacteristicIndex,
        num_characteristics: int,
        node_1: NodeMetricType,
        node_2: NodeMetricType,
        edge_1_to_2: EdgeMetricType
    ) -> NodeMetricType:
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
        n: CharacteristicIndex,
        num_characteristics: int,
        base_node: NodeMetricType,
        base_to_1_and_edge: tp.Tuple[NodeMetricType, EdgeMetricType],
        base_to_2_and_edge: tp.Tuple[NodeMetricType, EdgeMetricType],
        edge_1_to_2: EdgeMetricType,
        merged_node_1_2: NodeMetricType
    ) -> EdgeMetricType:
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
        
        merged_edge_value = cmath.sqrt(
            edge_b_1_value * edge_1_weight
            + edge_b_2_value * edge_2_weight
        )
        
        merged_edge = (
            merged_edge_value,
            num_cyles_1 + num_cyles_2
        )
    
        return merged_edge
    
    # The case where the input graph is composed of two or more disjoint graphs,
    # one needs to reduce the final merged node of each disjoint graph. This is 
    # is specified by the following function: 
    
    # ("merge amount" is how many verticies were merged for that node)
    MergeAmount = int
    MergedDisjointType = complex
    
    def merge_disjoint_graph_nodes(
        x: CharacteristicIndex,
        num_characteristics: int,
        disjoint_graph_nodes: tp.List[tp.Tuple[MergeAmount, NodeMetricType]]
    ) -> MergedDisjointType:
        return sum(
            node[0]
            for num_merged_verticies, node in disjoint_graph_nodes
        ) / len(disjoint_graph_nodes)
    
    
    # finally after all the merging is done,
    # we are left with 1 node of the graph which we must
    # extract a value called the characteristic value
    
    CharacteristicType = complex
    
    def calc_characteristic(
        x: CharacteristicIndex,
        num_characteristics: int,
        merged_value: MergedDisjointType
    ) -> CharacteristicType:
        return merged_value
    
    # we need to also specify how to average our characteristic values.
    # This is easy in this example since our characteristic is just an
    # float
    
    def calc_average(
        characteristic_values: tp.List[CharacteristicType]
    ) -> CharacteristicType:
        return sum(characteristic_values) / len(characteristic_values)
    
    # we need to specify how to compare two different characteristic list
    # represent two graph respectively. Here we do a simple root mean square
    
    FinalOutput = float
    
    def compare_chars(
        characteristic_values_graph_1: tp.List[CharacteristicType],
        characteristic_values_graph_2: tp.List[CharacteristicType]
    ) -> FinalOutput:
        return abs(cmath.sqrt(sum(
            (char_1 - char_2)**2
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
        merge_disjoint_graph_nodes,
        calc_characteristic,
        calc_average,
        compare_chars,
        EMPTY_GRAPH_CHARACTERISTIC_VALUE
    )
    
    import math
    
    def regular_compare(
        metric: StochasticGraphSimiliarity,
        graph_1: Graph,
        graph_2: Graph,
        num_samples: int = 30
    ) -> float:
        return metric.compare(graph_1, graph_2, num_samples)
    
    # gets rid of non-zero offset due to accumulated error
    def zero_base_compare(
        metric: StochasticGraphSimiliarity,
        graph_1: Graph,
        graph_2: Graph,
        num_samples: int = 30
    ) -> float:
        
        graph_1_on_1 = metric.compare(graph_1, graph_1, num_samples)
        graph_2_on_2 = metric.compare(graph_2, graph_2, num_samples)
        graph_1_on_2 = metric.compare(graph_1, graph_2, num_samples)
        
        return math.sqrt(
            (graph_1_on_1 - graph_1_on_2) ** 2
            + (graph_2_on_2 - graph_1_on_2) ** 2
        ) / 2
    
    # faster version of zero_base_compare
    def zero_base_compare_efficient(
        metric: StochasticGraphSimiliarity,
        graph_1: Graph,
        graph_2: Graph,
        num_samples: int = 30
    ) -> float:
        num_characteristics = metric.number_of_characteristics_needed_for_comparison(
            graph_1,
            graph_2
        )
        graph_1_characteristics_1 = metric.compute_graph_characteristics(
            graph_1,
            num_characteristics,
            num_samples
        )
        graph_1_characteristics_2 = metric.compute_graph_characteristics(
            graph_1,
            num_characteristics,
            num_samples
        )
        graph_2_characteristics_1 = metric.compute_graph_characteristics(
            graph_2,
            num_characteristics,
            num_samples
        )
        graph_2_characteristics_2 = metric.compute_graph_characteristics(
            graph_2,
            num_characteristics,
            num_samples
        )
        graph_1_on_1 = metric.compare_characteristics(
            graph_1_characteristics_1,
            graph_1_characteristics_2
        )
        graph_2_on_2 = metric.compare_characteristics(
            graph_2_characteristics_1,
            graph_2_characteristics_2
        )
        graph_1_on_2 = metric.compare_characteristics(
            graph_1_characteristics_1,
            graph_2_characteristics_2
        )
        
        return math.sqrt(
            (graph_1_on_1 - graph_1_on_2) ** 2
            + (graph_2_on_2 - graph_1_on_2) ** 2
        ) / 2
        
    
    NUM_SAMPLES = 30
    compare = zero_base_compare_efficient
    
    # run on graph 0 twice (hopefully its zero):
    similiarity = compare(metric, graph_0, graph_0, NUM_SAMPLES) 
    print(f"Simliarity metric between graph0 & itself: {similiarity}")
    
    # run on graph 1 twice (hopefully its zero):
    similiarity = compare(metric, graph_1, graph_1, NUM_SAMPLES) 
    print(f"Simliarity metric between graph1 & itself: {similiarity}")
    
    # run on graph 2 twice (hopefully its zero):
    similiarity = compare(metric, graph_2, graph_2, NUM_SAMPLES) 
    print(f"Simliarity metric between graph2 & itself: {similiarity}")
    
    # run on graph 3 twice (hopefully its zero):
    similiarity = compare(metric, graph_3, graph_3, NUM_SAMPLES) 
    print(f"Simliarity metric between graph3 & itself: {similiarity}")

    # run on graph 4 twice (hopefully its zero):
    similiarity = compare(metric, graph_4, graph_4, NUM_SAMPLES) 
    print(f"Simliarity metric between graph4 & itself: {similiarity}")
    
    # run on graph 5 twice (hopefully its zero):
    similiarity = compare(metric, graph_5, graph_5, NUM_SAMPLES) 
    print(f"Simliarity metric between graph5 & itself: {similiarity}")
    
    # run on graph 6 twice (hopefully its zero):
    similiarity = compare(metric, graph_6, graph_6, NUM_SAMPLES) 
    print(f"Simliarity metric between graph6 & itself: {similiarity}")
    
    # run on graph 7 twice (hopefully its zero):
    similiarity = compare(metric, graph_7, graph_7, NUM_SAMPLES) 
    print(f"Simliarity metric between graph7 & itself: {similiarity}")
    
    # run on graph 8 twice (hopefully its zero):
    similiarity = compare(metric, graph_8, graph_8, NUM_SAMPLES) 
    print(f"Simliarity metric between graph8 & itself: {similiarity}")
    
    # run on graph 9 twice (hopefully its zero):
    similiarity = compare(metric, graph_9, graph_9, NUM_SAMPLES) 
    print(f"Simliarity metric between graph9 & itself: {similiarity}")
    
    # run on the graph 1 and 2:
    similiarity = compare(metric, graph_1, graph_2, NUM_SAMPLES) 
    print(f"Simliarity metric between graph1 & graph2: {similiarity}")
    
    # run on the graph 1 and 3:
    similiarity = compare(metric, graph_1, graph_3, NUM_SAMPLES) 
    print(f"Simliarity metric between graph1 & graph3: {similiarity}")
    
    # run on the graph 2 and 3:
    similiarity = compare(metric, graph_2, graph_3, NUM_SAMPLES) 
    print(f"Simliarity metric between graph2 & graph3: {similiarity}")
    
    # run on the graph 1 and 4:
    similiarity = compare(metric, graph_1, graph_4, NUM_SAMPLES) 
    print(f"Simliarity metric between graph1 & graph4: {similiarity}")
    
    # run on the graph 1 and 5:
    similiarity = compare(metric, graph_1, graph_5, NUM_SAMPLES) 
    print(f"Simliarity metric between graph1 & graph5: {similiarity}")
    
    # run on the graph 1 and 6:
    similiarity = compare(metric, graph_1, graph_6, NUM_SAMPLES) 
    print(f"Simliarity metric between graph1 & graph6: {similiarity}")  
      
    # run on the graph 1 and 7:
    similiarity = compare(metric, graph_1, graph_7, NUM_SAMPLES) 
    print(f"Simliarity metric between graph1 & graph7: {similiarity}")  
    
    # run on the graph 1 and 8:
    similiarity = compare(metric, graph_1, graph_8, NUM_SAMPLES) 
    print(f"Simliarity metric between graph1 & graph8: {similiarity}")
    
    # run on the graph 1 and 9:
    similiarity = compare(metric, graph_1, graph_9, NUM_SAMPLES) 
    print(f"Simliarity metric between graph1 & graph9: {similiarity}")

if __name__ == "__main__":
    test_stochastic_graph_similiarity_on_chemical_compounds()