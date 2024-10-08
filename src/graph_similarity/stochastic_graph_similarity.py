import typing as tp
import random as rand

from graph import Graph
from graph_similarity_cache import GraphSimilarityCache


N = tp.TypeVar("N")
E = tp.TypeVar("E")

N1 = tp.TypeVar("N1")
E1 = tp.TypeVar("E1")

Output = tp.TypeVar("Output")
Characteristic = tp.TypeVar("Characteristic")
MergedDisjoints = tp.TypeVar("MergedDisjoints")
class StochasticGraphSimiliarity(tp.Generic[N, E, N1, E1, MergedDisjoints, Characteristic, Output]):
    
    def __init__(
        self,
        initialize_nodes: tp.Callable[
            [int, N], 
            N1
        ],
        initialize_edges: tp.Callable[
            [
                int,
                tp.Tuple[N, N1],
                tp.Tuple[N, N1], 
                E
            ],
            E1
        ],
        merge_node: tp.Callable[
            [int, N1, N1, E1],
            N1
        ],
        merge_edge: tp.Callable[
            [
                int,
                N1,
                tp.Tuple[N1, E1],
                tp.Tuple[N1, E1],
                E1,
                N1
            ],
            E1
        ],
        merge_disjoint_graph_nodes: tp.Callable[
            [
                int,
                tp.List[tp.Tuple[int, N1]]
            ],
            MergedDisjoints
        ],
        calc_characteristic: tp.Callable[
            [int, MergedDisjoints],
            Characteristic
        ],
        average_characteristic: tp.Callable[
            [
                tp.List[Characteristic],
                tp.Optional[
                    tp.Tuple[
                        # old average
                        Characteristic,
                        # number of values in average
                        int,    
                    ]
                ]
            ],
            Characteristic
        ],
        compare_characteristic: tp.Callable[
            [
                tp.List[Characteristic],
                tp.List[Characteristic]
            ],
            Output
        ],
        empty_graph_characteristic: Characteristic,
    ):
        def wrap_initialize_nodes(
            characteristic: int,
        ) -> tp.Callable[[N], N1]:
            return lambda x: initialize_nodes(characteristic, x)
        
        self.__initialize_nodes = wrap_initialize_nodes
        
        def wrap_initialize_edges(
            characteristic: int,
        ) -> tp.Callable[[tp.Tuple[N, N1], tp.Tuple[N, N1], E], E1]:
            return lambda x, y, z: initialize_edges(characteristic, x, y, z)
        
        self.__initialize_edges = wrap_initialize_edges
        def wrap_merge_node(
            characteristic: int,
        ) -> tp.Callable[[N1, N1, E1], N1]:
            return lambda x, y, z: merge_node(characteristic, x, y, z)
        self.__merge_node = wrap_merge_node
        def wrap_merge_edge(
            characteristic: int,
        ) -> tp.Callable[[N1, tp.Tuple[N1, E1], tp.Tuple[N1, E1], E1, N1], E1]:
            return lambda v, w, x, y, z: merge_edge(characteristic, v, w, x, y, z)
        
        self.__merge_disjoint_graph_nodes = merge_disjoint_graph_nodes
        
        self.__merge_edge = wrap_merge_edge
        def wrap_calc_characteristic(
            characteristic: int,
        ) -> tp.Callable[[MergedDisjoints], Characteristic]:
            return lambda x: calc_characteristic(characteristic, x)
        self.__calc_characteristic = wrap_calc_characteristic
        self.__average_characteristic = average_characteristic
        self.__compare_characteristic = compare_characteristic
        self.__empty_graph_characteristic = empty_graph_characteristic
    
    def compare(
        self,
        graph_1_u: tp.Union[
            Graph[N, E],
            tp.Tuple[
                Graph[N, E],
                GraphSimilarityCache[Characteristic]
            ]
        ],
        graph_2_u: tp.Union[
            Graph[N, E],
            tp.Tuple[
                Graph[N, E],
                GraphSimilarityCache[Characteristic]
            ]
        ],
        
        # This determines convergence of the algorithm, 
        # the higher the number the more convergent the results, but at the cost of computation
        # a good value for this is 30
        num_samples: int = 30,
        
        # will ultimately determine how many characteristics are needed for comparison
        # a higher faithfullness will have more precise results at the cost of more computation
        faithfullness: float = 1.0
    ) -> Output:
        
        if isinstance(graph_1_u, tuple):
            graph_1, cache_1 = graph_1_u
        else:
            graph_1 = graph_1_u
            cache_1 = None
        
        if isinstance(graph_2_u, tuple):
            graph_2, cache_2 = graph_2_u
        else:
            graph_2 = graph_2_u
            cache_2 = None
            
        num_characteristics = self.number_of_characteristics_needed_for_comparison(
            graph_1,
            graph_2,
            faithfullness
        )
        average_characteristics_graph_1 = self.compute_graph_characteristics(
            graph_1,
            num_characteristics,
            num_samples,
            cache_1
        )
        average_characteristics_graph_2 = self.compute_graph_characteristics(
            graph_2,
            num_characteristics,
            num_samples,
            cache_2
        )
        return self.__compare_characteristic(
            average_characteristics_graph_1,
            average_characteristics_graph_2
        )
    
    def number_of_characteristics_needed_for_comparison(
        self,
        graph_1: Graph[N,E],
        graph_2: Graph[N,E],
        faithfullness: float = 1.0
    ) -> int:
        graph_1_size = graph_1.num_nodes()
        graph_2_size = graph_2.num_nodes()
        
        num_characteristics = (max([graph_1_size, graph_2_size])) ** 2 * faithfullness
        return round(num_characteristics)
        
    def compute_graph_characteristics(
        self,
        graph: Graph[N,E],
        num_characteristics: int,
        samples: int = 30,
        cache: tp.Optional[GraphSimilarityCache[Characteristic]] = None
    ) -> tp.List[Characteristic]:
        if cache is None:
            return self.__compute_average_characteristics(
                graph,
                num_characteristics,
                samples,
            )
        else:
            return self.__compute_average_characteristics_with_cache(
                graph,
                num_characteristics,
                samples,
                cache
            )
    
    def create_cache(self) -> GraphSimilarityCache[Characteristic]:
        return GraphSimilarityCache[Characteristic].new()
    
    def __compute_average_characteristics_with_cache(
        self,
        graph: Graph[N,E],
        num_characteristics: int,
        samples: int,
        cache: GraphSimilarityCache[Characteristic]
    ) -> tp.List[Characteristic]:
        num_characteristics_in_cache = len(cache.characteristic_averages)
        
        if (
            cache.number_of_samples != 0 and
            num_characteristics_in_cache < num_characteristics
        ):
            list_of_new_characteristics = [
                self.__compute_characteristics(
                    graph,
                    range(num_characteristics_in_cache, num_characteristics)
                )
                for _ in range(cache.number_of_samples)
            ]
            characteristic_by_sample: tp.List[
                tp.Tuple[Characteristic, ...]
            ] = list(zip(*list_of_new_characteristics))
            new_characteristic_averages = [
                self.__average_characteristic(list(output), None)
                for output in characteristic_by_sample
            ]
            cache.characteristic_averages.extend(new_characteristic_averages)

        if cache.number_of_samples == 0:
            cache.characteristic_averages = self.__compute_average_characteristics(
                graph,
                num_characteristics,
                samples
            )
            cache.number_of_samples = samples
        
        elif cache.number_of_samples < samples:
            list_of_characteristics = [
                self.__compute_characteristics(
                    graph,
                    range(num_characteristics),
                )
                for _ in range(cache.number_of_samples, samples)
            ]
            
            characteristic_by_sample: tp.List[
                tp.Tuple[Characteristic, ...]
            ] = list(zip(*list_of_characteristics))

            cache.characteristic_averages = [
                self.__average_characteristic(list(output), (old_average, cache.number_of_samples))
                for output, old_average in zip(
                    characteristic_by_sample,
                    cache.characteristic_averages
                )
            ]
            cache.number_of_samples = samples
        
        return cache.characteristic_averages
    
    def __compute_average_characteristics(
        self,
        graph: Graph[N,E],
        num_characteristics: int,
        samples: int,
    ) -> tp.List[Characteristic]:
        list_of_characteristics = [
            self.__compute_characteristics(
                graph,
                range(num_characteristics)
            )
            for _ in range(samples)
        ]
        
        characteristic_by_sample: tp.List[
            tp.Tuple[Characteristic, ...]
        ] = list(zip(*list_of_characteristics))

        return [
            self.__average_characteristic(list(output), None)
            for output in characteristic_by_sample
        ]
    
    def __compute_characteristics(
        self,
        graph: Graph[N, E],
        characteristic_range: range
    ) -> tp.List[Characteristic]:
        return [
            self.__compute_characteristic(
                graph,
                c,
            )
            for c in characteristic_range
        ]
    
    def __compute_characteristic(
        self,
        graph: Graph[N, E],
        characteristic: int,
    ) -> Characteristic:
        # augmented graph
        aug_graph = graph.map(
            self.__initialize_nodes(characteristic),
            self.__initialize_edges(characteristic)
        )
        num_nodes = aug_graph.num_nodes()
        if num_nodes == 0:
            return self.__empty_graph_characteristic
        
        node_merge_numbers = {
            i: 1 for i in aug_graph.node_ids
        }
        
        disjoint_nodes: tp.List[tp.Tuple[int, N1]] = []
        
        while aug_graph.num_nodes() > 0:
            lowest_merge_number = min(node_merge_numbers.values())
            nodes_with_lowest_merge_number = [
                i
                for i in node_merge_numbers
                if node_merge_numbers[i] == lowest_merge_number
            ]
            selected_node_1_rel_index = rand.randint(0, len(nodes_with_lowest_merge_number) - 1)
            node_index_1 = nodes_with_lowest_merge_number[selected_node_1_rel_index]
            
            if aug_graph.num_neigbbors(node_index_1) == 0:
                disjoint_nodes.append(
                    (node_merge_numbers[node_index_1], aug_graph.get_node(node_index_1))
                )
                del node_merge_numbers[node_index_1]
                del aug_graph[node_index_1]
                continue
            
            node_1_neighbors = aug_graph.neighbors(node_index_1)
            neighbor_merge_numbers = {
                node_merge_numbers[neighbor_index]
                for neighbor_index in node_1_neighbors
            }
            
            lowest_neighbor_merge_number = min(neighbor_merge_numbers)
            neighbors_with_lowest_merge_number = [
                i
                for i in node_1_neighbors
                if node_merge_numbers[i] == lowest_neighbor_merge_number
            ]
            selected_node_2_rel_index = rand.randint(0, len(neighbors_with_lowest_merge_number) - 1)
            node_index_2 = neighbors_with_lowest_merge_number[selected_node_2_rel_index]
            aug_graph.merge(
                node_index_1,
                node_index_2,
                self.__merge_node(characteristic),
                self.__merge_edge(characteristic)
            )
            del node_merge_numbers[node_index_2]
            node_merge_numbers[node_index_1] = lowest_merge_number + lowest_neighbor_merge_number
        
        merged_disjoints = self.__merge_disjoint_graph_nodes(
            characteristic,
            disjoint_nodes
        )
        
        return self.__calc_characteristic(characteristic)(merged_disjoints)
