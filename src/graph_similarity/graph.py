import typing as tp

N = tp.TypeVar("N")
E = tp.TypeVar("E")

N1 = tp.TypeVar("N1")
E1 = tp.TypeVar("E1")
class Graph(tp.Generic[N, E]):
    @classmethod
    def create(
        cls,
        nodes: tp.Union[
            tp.Iterable[N],
            tp.Dict[int, N]
        ],
        edges: tp.Dict[tp.Tuple[int, int], E]
    ) -> "Graph[N, E]":
        if isinstance(nodes, dict):
            nodes_dict: tp.Dict[int, N] = nodes  # type: ignore
        else:
            nodes_dict: tp.Dict[int, N] = {
                node_id: node
                for node_id, node in enumerate(nodes)
            }
        edge_dict = {
            node_1_index: {
                node_2_index: 
                    edges[(node_1_index, node_2_index)] 
                    if (node_1_index, node_2_index) in edges
                    else edges[(node_2_index, node_1_index)]
                for node_2_index in nodes_dict
                if (
                    (
                        (node_1_index, node_2_index) in edges
                        or (node_2_index, node_1_index) in edges
                    )
                    and node_1_index != node_2_index
                )
            }
            for node_1_index in nodes_dict
        }
        
        return Graph(
            nodes_dict,
            edge_dict
        )
    
    def __init__(
        self,
        nodes: tp.Dict[int, N],
        edges:  tp.Dict[int, tp.Dict[int, E]],
    ):
        self.__nodes = nodes
        self.__edges = edges
    
    def num_nodes(self) -> int:
        return len(self.__nodes)
    
    @property
    def node_ids(self) -> tp.Set[int]:
        return set(self.__nodes)
    
    @property
    def nodes(self) -> tp.Tuple[N, ...]:
        return tuple(self.__nodes.values())
    
    def copy_edges(self) -> tp.Dict[int, tp.Dict[int, E]]:
        return {
            node_index: {
                adjacent_node: edge
                for adjacent_node, edge in adjacent_nodes.items()
            }
            for node_index, adjacent_nodes in self.__edges.items()
        }
    
    def get_node(
        self,
        node_index: int
    ) -> N:
        return self.__nodes[node_index]
    
    def num_neigbbors(
        self,
        node_index: int
    ) -> int:
        return len(
            self.__edges[node_index]
        )
    
    def neighbors(
        self,
        node_index: int
    ) -> tp.Set[int]:
        return {
            neighbor_index
            for neighbor_index in self.__edges[node_index]
        }
    
    def edge(self, node_index_1: int, node_index_2: int) -> E:
        return self.__edges[node_index_1][node_index_2]
    
    def edges(self, node_index: int) -> tp.Dict[int, E]:
        return self.__edges[node_index]
    
    def map(
        self,
        node_map: tp.Callable[[N], N1],
        edge_map: tp.Callable[[tp.Tuple[N, N1], tp.Tuple[N, N1], E], E1]
    ) -> "Graph[N1, E1]":
        node_transistion = tuple(map(
            lambda x: (
                x,
                (
                    self.__nodes[x],
                    node_map(self.__nodes[x])
                )
            ),
            self.__nodes
        ))
        new_edges = {
            node_index_1: {
                node_index_2: edge_map(n_1, n_2, self.__edges[node_index_1][node_index_2])
                for (node_index_2, n_2) in node_transistion
                if node_index_1 in self.__edges and node_index_2 in self.__edges[node_index_1] 
            }
            for (node_index_1, n_1) in node_transistion
        }
        new_nodes = {
            id_: new
            for id_, (_, new) in node_transistion
        }
        return Graph(
            new_nodes,
            new_edges
        )

    def __delitem__(self, node_index: int):
        del self.__nodes[node_index]
        del self.__edges[node_index]
        for node_edges in self.__edges.values():
            if node_index in node_edges: 
                del node_edges[node_index]

    def merge(
        self,
        node_index_1: int,
        node_index_2: int,
        merge_node_fnc: tp.Callable[[N, N, E], N],
        merge_edge_fnc: tp.Callable[[N, tp.Tuple[N, E], tp.Tuple[N, E], E, N], E]
    ):
        node_1 = self.__nodes[node_index_1]
        node_2 = self.__nodes[node_index_2]
        edge_1_to_2 = self.__edges[node_index_1][node_index_2]
        
        new_node_1 = merge_node_fnc(node_1, node_2, edge_1_to_2)
        node_1_neighbors = set(self.__edges[node_index_1].keys())
        node_2_neighbors = set(self.__edges[node_index_2].keys())
        common_neighbors = node_1_neighbors.intersection(node_2_neighbors)
        node_2_unique_neighbors = node_2_neighbors.difference(common_neighbors)
        
        new_common_edges = {
            common_neighbor: merge_edge_fnc(
                self.__nodes[common_neighbor],
                (node_1, self.__edges[node_index_1][common_neighbor]),
                (node_2, self.__edges[node_index_2][common_neighbor]),
                edge_1_to_2,
                new_node_1
            )
            for common_neighbor in common_neighbors
        }
        unique_edges_from_node_2 = {
            unique_neighbor: self.__edges[node_index_2][unique_neighbor]
            for unique_neighbor in node_2_unique_neighbors
            if unique_neighbor != node_index_1
        }
        
        # --- update graph ---
        # remove node 2
        del self[node_index_2]
        # update value of node 1
        self.__nodes[node_index_1] = new_node_1
        # update node2 common ancestor edges
        for common_ancestor, edge in new_common_edges.items():
            self.__edges[common_ancestor][node_index_1] = edge
            self.__edges[node_index_1][common_ancestor] = edge
        # add other node2 edges
        for node_2_neighbor, edge in unique_edges_from_node_2.items():
            self.__edges[node_2_neighbor][node_index_1] = edge
            self.__edges[node_index_1][node_2_neighbor] = edge

    def __repr__(self) -> str:
        class_name = type(self).__name__
        return (
            f'{class_name}(nodes={self.__nodes}, edges={self.__edges})'
        )
    
    def pretty_string_print_graph(
        self
    ) -> str:
        class_name = type(self).__name__
        nodes_components_str = "\n".join(
            f'    {node_index}: {node}'
            for node_index, node in self.__nodes.items()
        )
        
        if len(self.__nodes) == 0:
            nodes_str = '  nodes={}'
        else:
            nodes_str = (
                f'  nodes={{\n'
                f'{nodes_components_str}\n'
                f'  }}'
            )
        
        edge_components_str = "\n".join(
            f'    {node_index_1} - {node_index_2}: {edge}'
            for node_index_1, adjacent_nodes in self.__edges.items()
            for node_index_2, edge in adjacent_nodes.items()
        )
        
        if len(self.__edges) == 0:
            edges_str = '  edges={}'
        else:
            edges_str = (
                f'  edges={{\n'
                f'{edge_components_str}\n'
                f'  }}'
            )
        
        if len(self.__nodes) == 0 and len(self.__edges) == 0:
            final_str = f'{class_name}(nodes={{}}, edges={{}})'
        else:
            final_str = "\n".join((
                f'{class_name}(',
                nodes_str,
                edges_str,
                f')'
            ))
        return final_str
