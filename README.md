# GraphSimilarity
Created a solution for graph similiarity using a stochastic approach.

The algorithm doesn't seem to be in academia to the best of my knowledge; The closest things to this approaches is Weisfeiler-Lehman Graph Hashing (but not really?) 
This approach is fast but not optimized in lower languages or parallelized (Just a proof of concept after all that works really well). If enough interest is put here I might do it.

The nice thing about this algorithm, is that it embeds each graph into a characteristic function (fyi characteristic function has nothing to do with the characterstic polynomial of the matrix) where is can be represented as a 1 dimensional array of complex numbers, so the graphs can be embeded into this form for caching purposes, and then the similarity metric calculation is just and average of the difference vector magnitudes. If more characteric values are needed for the embedding (for higher precision), the new ones just need to be calculated (rather then the whole embedding redone); they are appended to the end of the vector. The calculation of the embedding procedure is parallelizable and the output vector size doesn't rely on graph size, only precision does. Another note, the characteristic function, although calculated using a random procedure, should converge with to the same value for a single graph in theory. In practice, the characteristic function converges to the same values really well from my test, given only 30 samples (a value used for convergence).


The approach should have a average case time complexity of O((V + E) * C), but the implementation is more like O((V^2) * C) I think (I could of optimized here). 
Just for reference, V is number of verticies of the graph, E is the number of edges, and C is the number of values calculated from the characteristic function used for the embedding 1-d vector.
After embedding each graph once, the time complexity for the similiarity value is just O(C) between any two graph embeddings.

For more information, please look at the test file, hopefully it's not that confusing.
 



