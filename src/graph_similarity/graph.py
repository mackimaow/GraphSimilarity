import typing as tp

N = tp.TypeVar("N")
E = tp.TypeVar("E")

N1 = tp.TypeVar("N1")
E1 = tp.TypeVar("E1")
class Graph(tp.Generic[N, E]):
    @classmethod
    def create(
        cls,
        nodes: tp.Tuple[N, ...],
        edges: tp.Dict[tp.Tuple[int, int], E]
    ):
        node_ids = set(range(len(nodes)))
        node_dict = {
            node_id: node
            for node_id, node in enumerate(nodes)
        }
        edge_dict = {
            node1: {
                node2: 
                    edges[(node1, node2)] 
                    if (node1, node2) in edges
                    else edges[(node2, node1)]
                for node2 in node_ids
                if ((node1, node2) in edges or (node2, node1) in edges) and node1 != node2
            }
            for node1 in node_ids
        }
        return Graph(
            node_ids,
            node_dict,
            edge_dict
        )
    
    def __init__(
        self,
        node_ids: tp.Set[int],
        nodes: tp.Dict[int, N],
        edges:  tp.Dict[int, tp.Dict[int, E]],
    ):
        self.__node_ids = node_ids
        self.__nodes = nodes
        self.__edges = edges
    
    def num_nodes(self) -> int:
        return len(self.__node_ids)
    
    @property
    def nodes(self) -> tp.Tuple:
        return tuple(self.__nodes.values())
    
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
            lambda x: (x, node_map(x)),
            self.nodes
        ))
        new_edges = {
            node_index_1: {
                node_index_2: edge_map(n_1, n_2, self.__edges[node_index_1][node_index_2])
                for (node_index_2, n_2) in enumerate(node_transistion)
                if node_index_1 in self.__edges and node_index_2 in self.__edges[node_index_1] 
            }
            for (node_index_1, n_1) in enumerate(node_transistion)
        }
        new_node_ids = set(self.__node_ids)
        new_nodes = {
            id_: new
            for id_, (_, new) in enumerate(node_transistion)
        }
        return Graph(
            new_node_ids,
            new_nodes,
            new_edges
        )

    def __delitem__(self, node_index: int):
        del self.__nodes[node_index]
        self.__node_ids.remove(node_index)
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
