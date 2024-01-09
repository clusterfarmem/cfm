import igraph as ig


G = ig.Graph.Erdos_Renyi(21000, 0.001)


pr = G.pagerank(niter=1000, nthreads=8)

#print(pr)