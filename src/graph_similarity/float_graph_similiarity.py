from stochastic_graph_similarity import StochasticGraphSimiliarity
import typing as tp
import cmath


MergeAmount = int
MergedDisjointType = complex

CharacteristicIndex = int
MergedCycles = int
MergedNodes = int
NodeMetricType = tp.Tuple[complex, MergedCycles, MergedNodes]
EdgeMetricType = tp.Tuple[complex, MergedCycles]
CharacteristicType = complex
FinalOutput = float


NodeType = tp.TypeVar("NodeType")
EdgeType = tp.TypeVar("EdgeType")


class FloatGraphSimilarity(
    tp.Generic[NodeType, EdgeType],
    StochasticGraphSimiliarity[
        NodeType,
        EdgeType,
        NodeMetricType,
        EdgeMetricType,
        MergedDisjointType,
        CharacteristicType,
        FinalOutput
    ]
):

    # ---- making similiarity metric ----
    
    # this is used to add extra values to the nodes that we may want as we
    # do while doing graph reductions within the metric computation.
    # Here we are keeping track of how many cycles and nodes were merged.
    
    # the golden ratio is used here to make that the multiplicative factor for each characteristic
    # will never be the same for any two different characteristics. This is important for
    # characteristc values to be linearly independent.
    GOLDEN_RATIO = (1 + 5 ** 0.5) / 2
    
    @classmethod
    def __linearly_independent_multiplier(
        cls,
        characteristic: CharacteristicIndex 
    ) -> complex:
        return cmath.exp(
            (cmath.pi * 2j) * (cls.GOLDEN_RATIO * characteristic)
        )
    
    @classmethod
    def __initialize_nodes(
        cls,
        characteristic: CharacteristicIndex,
        node: NodeType,
        node_to_float: tp.Callable[[NodeType], float]
    ) -> NodeMetricType:
        return (node_to_float(node) + 0j, 1, 1)
    
    # we do the same for edges
    @classmethod
    def __initialize_edges(
        cls,
        characteristic: CharacteristicIndex,
        node_1: tp.Tuple[NodeType, NodeMetricType],
        node_2: tp.Tuple[NodeType, NodeMetricType],
        edge: EdgeType,
        edge_to_float: tp.Callable[
            [
                tp.Tuple[NodeType, float],
                tp.Tuple[NodeType, float],
                EdgeType
            ],
            float
        ]
    ) -> EdgeMetricType:
        return (
            edge_to_float(
                (node_1[0], node_1[1][0].real),
                (node_2[0], node_2[1][0].real),
                edge
            ) + 0j,
            1
        )
    
    # now we create a function computes when two nodes merge
    # for some characteristic function index
    
    @classmethod
    def __merge_nodes(
        cls,
        n: CharacteristicIndex,
        node_1: NodeMetricType,
        node_2: NodeMetricType,
        edge_1_to_2: EdgeMetricType
    ) -> NodeMetricType:
        (value_1, merged_cycles_1, merged_nodes_1) = node_1
        (value_2, merged_cycles_2, merged_nodes_2) = node_2
        (edge_value, edge_cycles) = edge_1_to_2
        
        value_1_weight = merged_cycles_1 / (
            merged_cycles_1 + merged_cycles_2
        ) * cls.__linearly_independent_multiplier(n)
        
        value_2_weight = merged_cycles_2 / (
            merged_cycles_1 + merged_cycles_2
        ) / cls.__linearly_independent_multiplier(n)
        
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
    
    @classmethod
    def __merge_edges(
        cls,
        n: CharacteristicIndex,
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
        
        edge_1_weight =  cls.__linearly_independent_multiplier(n)
        edge_2_weight =  1.0 / cls.__linearly_independent_multiplier(n)
        
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
    @classmethod
    def __merge_disjoint_graph_nodes(
        cls,
        x: CharacteristicIndex,
        disjoint_graph_nodes: tp.List[tp.Tuple[MergeAmount, NodeMetricType]]
    ) -> MergedDisjointType:
        return sum(
            node[0]
            for _num_merged_verticies, node in disjoint_graph_nodes
        ) / len(disjoint_graph_nodes)
    
    
    # finally after all the merging is done,
    # we are left with 1 node of the graph which we must
    # extract a value called the characteristic value
    @classmethod
    def __calc_characteristic(
        cls,
        x: CharacteristicIndex,
        merged_value: MergedDisjointType
    ) -> CharacteristicType:
        return merged_value
    
    # we need to also specify how to average our characteristic values.
    # This is easy in this example since our characteristic is just an
    # float
    @classmethod
    def __calc_average(
        cls,
        characteristic_values: tp.List[CharacteristicType],
        old_average: tp.Optional[tp.Tuple[CharacteristicType, int]]
    ) -> CharacteristicType:
        if old_average is not None:
            return (
                old_average[0] * old_average[1] + sum(characteristic_values)
            ) / (
                old_average[1] + len(characteristic_values)
            )
        else:
            return sum(characteristic_values) / len(characteristic_values)
    
    # we need to specify how to compare two different characteristic list
    # represent two graph respectively. Here we do a simple root mean square    
    @classmethod
    def __compare_chars(
        cls,
        characteristic_values_graph_1: tp.List[CharacteristicType],
        characteristic_values_graph_2: tp.List[CharacteristicType]
    ) -> FinalOutput:
        divisor = len(characteristic_values_graph_1) if len(characteristic_values_graph_1) > 0 else 1
        return abs(
            sum(
                cmath.polar(char_1 - char_2)[0]
                for char_1, char_2 in zip(
                    characteristic_values_graph_1,
                    characteristic_values_graph_2
                )
            ) / divisor
        )
    
    # some times we need to specify what a characteric value would be if the graph
    # is empty. For this we can just put zero
    
    EMPTY_GRAPH_CHARACTERISTIC_VALUE = 0.0
    
    
    def __init__(
        self,
        node_to_float: tp.Callable[[NodeType], float],
        edge_to_float: tp.Callable[
            [
                tp.Tuple[NodeType, float],
                tp.Tuple[NodeType, float],
                EdgeType
            ],
            float
        ],
    ):
        super().__init__(
            lambda characteristic, node: 
                self.__initialize_nodes(
                    characteristic,
                    node,
                    node_to_float
                ),
            lambda characteristic, node_1, node_2, edge:
                self.__initialize_edges(
                    characteristic,
                    node_1,
                    node_2,
                    edge,
                    edge_to_float
                ),
            self.__merge_nodes,
            self.__merge_edges,
            self.__merge_disjoint_graph_nodes,
            self.__calc_characteristic,
            self.__calc_average,
            self.__compare_chars,
            self.EMPTY_GRAPH_CHARACTERISTIC_VALUE
        )