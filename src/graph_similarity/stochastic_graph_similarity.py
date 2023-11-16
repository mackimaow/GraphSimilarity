import typing as tp
import random as rand

from .graph import Graph


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
        initialize_nodes: tp.Callable[[int, int, N], N1],
        initialize_edges: tp.Callable[[int, int, tp.Tuple[N, N1], tp.Tuple[N, N1], E], E1],
        merge_node: tp.Callable[[int, int, N1, N1, E1], N1],
        merge_edge: tp.Callable[[int, int, N1, tp.Tuple[N1, E1], tp.Tuple[N1, E1], E1, N1], E1],
        merge_disjoint_graph_nodes: tp.Callable[[int, int, tp.List[tp.Tuple[int, N1]]], MergedDisjoints],
        calc_characteristic: tp.Callable[[int, int, MergedDisjoints], Characteristic],
        average_characteristic: tp.Callable[[tp.List[Characteristic]], Characteristic],
        compare_characteristic: tp.Callable[[tp.List[Characteristic],tp.List[Characteristic]], Output],
        empty_graph_characteristic: Characteristic
    ):
        def wrap_initialize_nodes(
            characteristic: int,
            number_of_characteristics: int
        ) -> tp.Callable[[N], N1]:
            return lambda x: initialize_nodes(characteristic, number_of_characteristics, x)
        
        self.__initialize_nodes = wrap_initialize_nodes
        
        def wrap_initialize_edges(
            characteristic: int,
            number_of_characteristics: int
        ) -> tp.Callable[[tp.Tuple[N, N1], tp.Tuple[N, N1], E], E1]:
            return lambda x, y, z: initialize_edges(characteristic, number_of_characteristics, x, y, z)
        
        self.__initialize_edges = wrap_initialize_edges
        def wrap_merge_node(
            characteristic: int,
            number_of_characteristics: int
        ) -> tp.Callable[[N1, N1, E1], N1]:
            return lambda x, y, z: merge_node(characteristic, number_of_characteristics, x, y, z)
        self.__merge_node = wrap_merge_node
        def wrap_merge_edge(
            characteristic: int,
            number_of_characteristics: int
        ) -> tp.Callable[[N1, tp.Tuple[N1, E1], tp.Tuple[N1, E1], E1, N1], E1]:
            return lambda v, w, x, y, z: merge_edge(characteristic, number_of_characteristics, v, w, x, y, z)
        
        self.__merge_disjoint_graph_nodes = merge_disjoint_graph_nodes
        
        self.__merge_edge = wrap_merge_edge
        def wrap_calc_characteristic(
            characteristic: int,
            number_of_characteristics: int
        ) -> tp.Callable[[MergedDisjoints], Characteristic]:
            return lambda x: calc_characteristic(characteristic, number_of_characteristics, x)
        self.__calc_characteristic = wrap_calc_characteristic
        self.__average_characteristic = average_characteristic
        self.__compare_characteristic = compare_characteristic
        self.__empty_graph_characteristic = empty_graph_characteristic
    
    def compare(
        self,
        graph_1: Graph[N, E],
        graph_2: Graph[N, E],
        num_samples: int = 30
    ) -> Output:
        num_characteristics = self.number_of_characteristics_needed_for_comparison(
            graph_1,
            graph_2
        )
        average_characteristics_graph_1 = self.__compute_average_characteristics(
            graph_1,
            num_characteristics,
            num_samples
        )
        average_characteristics_graph_2 = self.__compute_average_characteristics(
            graph_2,
            num_characteristics,
            num_samples
        )
        return self.__compare_characteristic(
            average_characteristics_graph_1,
            average_characteristics_graph_2
        )
    
    def compare_characteristics(
        self,
        graph_1_characteristics: tp.List[Characteristic],
        graph_2_characteristics: tp.List[Characteristic]
    ) -> Output:
        num_1_chars = len(graph_1_characteristics)
        num_2_chars = len(graph_2_characteristics)
        if num_1_chars != num_2_chars:
            raise ValueError(
                f'Cannot compare two graph characteristics of different lengths {num_1_chars} and {num_2_chars}.'
            )

        return self.__compare_characteristic(
            graph_1_characteristics,
            graph_2_characteristics
        )
    
    def number_of_characteristics_needed_for_comparison(
        self,
        graph_1: Graph[N,E],
        graph_2: Graph[N,E],
    ) -> int:
        graph_1_size = graph_1.num_nodes()
        graph_2_size = graph_2.num_nodes()
        
        num_characteristics = (max([graph_1_size, graph_2_size])) ** 2
        return num_characteristics
        
    def compute_graph_characteristics(
        self,
        graph: Graph[N,E],
        num_characteristics: int,
        samples: int = 30  
    ) -> tp.List[Characteristic]:
        return self.__compute_average_characteristics(
            graph,
            num_characteristics,
            samples
        )
        
    def __compute_average_characteristics(
        self,
        graph: Graph[N,E],
        num_characteristics: int,
        samples: int
    ) -> tp.List[Characteristic]:
        list_of_characteristics = [
            self.__compute_characteristics(
                graph,
                num_characteristics,
                num_characteristics
            )
            for i in range(samples)
        ]
        
        characteristic_by_sample: tp.List[
            tp.Tuple[Characteristic, ...]
        ] = list(zip(*list_of_characteristics))

        return [
            self.__average_characteristic(list(output))
            for output in characteristic_by_sample
        ]
    
    def __compute_characteristics(
        self,
        graph: Graph[N, E],
        num_characteristic: int,
        number_of_characteristics: int
    ) -> tp.List[Characteristic]:
        return [
            self.__compute_characteristic(
                graph,
                c,
                number_of_characteristics
            )
            for c in range(num_characteristic)
        ]
    
    def __compute_characteristic(
        self,
        graph: Graph[N, E],
        characteristic: int,
        number_of_characteristics: int
    ) -> Characteristic:
        # augmented graph
        aug_graph = graph.map(
            self.__initialize_nodes(characteristic, number_of_characteristics),
            self.__initialize_edges(characteristic, number_of_characteristics)
        )
        num_nodes = aug_graph.num_nodes()
        if num_nodes == 0:
            return self.__empty_graph_characteristic
        
        node_merge_numbers = {
            i: 1 for i in range(num_nodes)
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
                self.__merge_node(characteristic, number_of_characteristics),
                self.__merge_edge(characteristic, number_of_characteristics)
            )
            del node_merge_numbers[node_index_2]
            node_merge_numbers[node_index_1] = lowest_merge_number + lowest_neighbor_merge_number
        
        merged_disjoints = self.__merge_disjoint_graph_nodes(
            characteristic,
            number_of_characteristics,
            disjoint_nodes
        )
        
        return self.__calc_characteristic(characteristic, number_of_characteristics)(merged_disjoints)
