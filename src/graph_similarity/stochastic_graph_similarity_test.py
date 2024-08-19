from enum import Enum
import math
from itertools import combinations
import random

from dataclasses import dataclass
from graph_similarity_cache import GraphSimilarityCache

from graph import Graph
from float_graph_similiarity import FloatGraphSimilarity

BOND_LENGTH = float

class Atom(Enum):
    OXYGEN = 8
    VANADIUM = 23


def test_stochastic_graph_similiarity_on_chemical_compounds():    
    
    # graph with unique ids set
    graph_0 = Graph[Atom, BOND_LENGTH].create(
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
    
    # same as graph 1 except everything is fully connected
    graph_10 = Graph[Atom, BOND_LENGTH].create(
        (
            *[Atom.VANADIUM] * 4,  # 0 - 3
            *[Atom.OXYGEN] * 14  # 4 - 17
        ),
        {
            i: 1
            for i in combinations(range(18), 2)
        }
    )
    
    # same as graph 10 except everything is fully connected with random edge weights (0 - 1)
    graph_11 = Graph[Atom, BOND_LENGTH].create(
        (
            *[Atom.VANADIUM] * 4,  # 0 - 3
            *[Atom.OXYGEN] * 14  # 4 - 17
        ),
        {
            i: random.random()
            for i in combinations(range(18), 2)
        }
    )
    
    
    # now we can make our metric to use on any graph with atoms and bond lengths:
    metric = FloatGraphSimilarity[
        Atom,
        BOND_LENGTH
    ](
        # node to float
        lambda node: 1 if node == Atom.OXYGEN else 2,
        
        # edge to float
        lambda node_1, node_2, edge_1_to_2: edge_1_to_2,   
    )
    
    
    @dataclass(eq=False)
    class GraphAndCache:
        graph: Graph[Atom, BOND_LENGTH]
        cache1: GraphSimilarityCache[complex]
        cache2: GraphSimilarityCache[complex]
        
        @classmethod
        def create(cls, graph: Graph[Atom, BOND_LENGTH]) -> "GraphAndCache":
            return GraphAndCache(
                graph,
                metric.create_cache(),
                metric.create_cache(),
            )
    
    def regular_compare(
        metric: FloatGraphSimilarity[Atom, BOND_LENGTH],
        graph_and_cache_1: GraphAndCache,
        graph_and_cache_2: GraphAndCache,
        num_samples: int = 30,
        use_cache: bool = True,
        faithfullness: float = 1.0
    ) -> float:
        if use_cache:
            graph_1 = graph_and_cache_1.graph 
            cache_1 = graph_and_cache_1.cache1
            
            graph_2 = graph_and_cache_2.graph
            cache_2 = graph_and_cache_2.cache2
            
            return metric.compare(
                (graph_1, cache_1),
                (graph_2, cache_2),
                num_samples,
                faithfullness
            )
        else :
            return metric.compare(
                graph_and_cache_1.graph,
                graph_and_cache_2.graph,
                num_samples,
                faithfullness
            )
    
    # gets rid of non-zero offset due to accumulated error
    def zero_base_compare(
        metric: FloatGraphSimilarity[Atom, BOND_LENGTH],
        graph_and_cache_1: GraphAndCache,
        graph_and_cache_2: GraphAndCache,
        num_samples: int = 30,
        use_cache: bool = True,
        faithfullness: float = 1.0
    ) -> float:
        
        if use_cache:
            if graph_and_cache_1.graph != graph_and_cache_2.graph:    
                graph_1 = graph_and_cache_1.graph 
                cache_1_1 = graph_and_cache_1.cache1
                cache_1_2 = graph_and_cache_1.cache2
                
                graph_2 = graph_and_cache_2.graph
                cache_2_1 = graph_and_cache_2.cache1
                cache_2_2 = graph_and_cache_2.cache2
                
                graph_1_on_1 = metric.compare(
                    (graph_1, cache_1_1),
                    (graph_1, cache_1_2),
                    num_samples,
                    faithfullness
                )
                graph_2_on_2 = metric.compare(
                    (graph_2, cache_2_1),
                    (graph_2, cache_2_2),
                    num_samples,
                    faithfullness
                )
                graph_1_on_2 = metric.compare(
                    (graph_1, cache_1_1),
                    (graph_2, cache_2_2),
                    num_samples,
                    faithfullness
                )
            else:
                graph_1 = graph_and_cache_1.graph 
                cache_1_1 = graph_and_cache_1.cache1
                cache_1_2 = graph_and_cache_1.cache2
                
                graph_2 = graph_1
                cache_2_1 = metric.create_cache()
                cache_2_2 = metric.create_cache()
                
                graph_1_on_1 = metric.compare(
                    (graph_1, cache_1_1),
                    (graph_1, cache_1_2),
                    num_samples,
                    faithfullness
                )
                graph_2_on_2 = metric.compare(
                    (graph_2, cache_2_1),
                    (graph_2, cache_2_2),
                    num_samples,
                    faithfullness
                )
                graph_1_on_2 = metric.compare(
                    (graph_1, cache_1_1),
                    (graph_2, cache_2_2),
                    num_samples,
                    faithfullness
                )
        else:
            graph_1 = graph_and_cache_1.graph 
            graph_2 = graph_and_cache_2.graph
            
            graph_1_on_1 = metric.compare(
                graph_1,
                graph_1,
                num_samples,
                faithfullness
            )
            graph_2_on_2 = metric.compare(
                graph_2,
                graph_2,
                num_samples,
                faithfullness
            )
            graph_1_on_2 = metric.compare(
                graph_1,
                graph_2,
                num_samples,
                faithfullness
            )
        
        return math.sqrt(
            (graph_1_on_1 - graph_1_on_2) ** 2
            + (graph_2_on_2 - graph_1_on_2) ** 2
        ) / 2
        
    # caching is used to make the metric run faster after the first comparison using that graph
    # gc_0 = GraphAndCache.create(graph_0)
    gc_1 = GraphAndCache.create(graph_1)
    gc_2 = GraphAndCache.create(graph_2)
    gc_3 = GraphAndCache.create(graph_3)
    gc_4 = GraphAndCache.create(graph_4)
    gc_5 = GraphAndCache.create(graph_5)
    gc_6 = GraphAndCache.create(graph_6)
    gc_7 = GraphAndCache.create(graph_7)
    gc_8 = GraphAndCache.create(graph_8)
    gc_9 = GraphAndCache.create(graph_9)
    gc_10 = GraphAndCache.create(graph_10)
    gc_11 = GraphAndCache.create(graph_11)
    
    
    # the number of samples dictates the convergence of the metric
    # the more samples the more accurate the metric will be but the slower it will run (linearly)
    NUM_SAMPLES = 30
    
    # caching makes the metric run faster after the first comparison using that graph
    USE_CACHE = True
    
    # Faithfullness represents how much the metric will be accurate. 
    # Scaling this up will increase accuracy,
    # but will also linearly increase the computation time.
    FAITHFULLNESS: float = 0.5
    
    def compare(gc1: GraphAndCache, gc2: GraphAndCache) -> float:  
        return zero_base_compare(
            metric,
            gc1,
            gc2,
            NUM_SAMPLES,
            USE_CACHE,
            FAITHFULLNESS
        )
    
    print("Running tests on chemical compounds")
    # # run on graph 0 twice (hopefully its zero):
    # similiarity = compare(gc_0, gc_0) 
    # print(f"Simliarity metric between graph0 & itself: {similiarity}")
    
    # run on graph 1 twice (hopefully its zero):
    similiarity = compare(gc_1, gc_1) 
    print(f"Simliarity metric between graph1 & itself: {similiarity}")
    
    # run on graph 2 twice (hopefully its zero):
    similiarity = compare(gc_2, gc_2) 
    print(f"Simliarity metric between graph2 & itself: {similiarity}")
    
    # run on graph 3 twice (hopefully its zero):
    similiarity = compare(gc_3, gc_3) 
    print(f"Simliarity metric between graph3 & itself: {similiarity}")

    # run on graph 4 twice (hopefully its zero):
    similiarity = compare(gc_4, gc_4) 
    print(f"Simliarity metric between graph4 & itself: {similiarity}")
    
    # run on graph 5 twice (hopefully its zero):
    similiarity = compare(gc_5, gc_5) 
    print(f"Simliarity metric between graph5 & itself: {similiarity}")
    
    # run on graph 6 twice (hopefully its zero):
    similiarity = compare(gc_6, gc_6) 
    print(f"Simliarity metric between graph6 & itself: {similiarity}")
    
    # run on graph 7 twice (hopefully its zero):
    similiarity = compare(gc_7, gc_7) 
    print(f"Simliarity metric between graph7 & itself: {similiarity}")
    
    # run on graph 8 twice (hopefully its zero):
    similiarity = compare(gc_8, gc_8) 
    print(f"Simliarity metric between graph8 & itself: {similiarity}")
    
    # run on graph 9 twice (hopefully its zero):
    similiarity = compare(gc_9, gc_9) 
    print(f"Simliarity metric between graph9 & itself: {similiarity}")
    
    # run on graph 10 twice (hopefully its zero):
    similiarity = compare(gc_10, gc_10) 
    print(f"Simliarity metric between graph10 & itself: {similiarity}")
    
    # run on graph 11 twice (hopefully its zero):
    similiarity = compare(gc_11, gc_11) 
    print(f"Simliarity metric between graph11 & itself: {similiarity}")
    
    
    # run on the graph 1 and 2:
    similiarity = compare(gc_1, gc_2) 
    print(f"Simliarity metric between graph1 & graph2: {similiarity}")
    
    # run on the graph 1 and 3:
    similiarity = compare(gc_1, gc_3) 
    print(f"Simliarity metric between graph1 & graph3: {similiarity}")
    
    # run on the graph 2 and 3:
    similiarity = compare(gc_2, gc_3) 
    print(f"Simliarity metric between graph2 & graph3: {similiarity}")
    
    # run on the graph 1 and 4:
    similiarity = compare(gc_1, gc_4) 
    print(f"Simliarity metric between graph1 & graph4: {similiarity}")
    
    # run on the graph 1 and 5:
    similiarity = compare(gc_1, gc_5) 
    print(f"Simliarity metric between graph1 & graph5: {similiarity}")
    
    # run on the graph 1 and 6:
    similiarity = compare(gc_1, gc_6) 
    print(f"Simliarity metric between graph1 & graph6: {similiarity}")  
      
    # run on the graph 1 and 7:
    similiarity = compare(gc_1, gc_7) 
    print(f"Simliarity metric between graph1 & graph7: {similiarity}")  
    
    # run on the graph 1 and 8:
    similiarity = compare(gc_1, gc_8) 
    print(f"Simliarity metric between graph1 & graph8: {similiarity}")
    
    # run on the graph 1 and 9:
    similiarity = compare(gc_1, gc_9) 
    print(f"Simliarity metric between graph1 & graph9: {similiarity}")
    
    # run on the graph 1 and 10:
    similiarity = compare(gc_1, gc_10) 
    print(f"Simliarity metric between graph1 & graph10: {similiarity}")
    
    # run on the graph 1 and 11:
    similiarity = compare(gc_1, gc_11) 
    print(f"Simliarity metric between graph1 & graph11: {similiarity}")\
        
    # run on the graph 10 and 11:
    similiarity = compare(gc_10, gc_11) 
    print(f"Simliarity metric between graph10 & graph11: {similiarity}")

if __name__ == "__main__":
    test_stochastic_graph_similiarity_on_chemical_compounds()