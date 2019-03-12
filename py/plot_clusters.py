from plotter import Plotter
from recordings_clusters import RecordingsClusters
from bipartite_clusters import BipartiteGraph

b = BipartiteGraph()
b.make_clusters('mfcc')
b.calculate_filter(type='heat-prob')
p = Plotter(b.clusters)
# for row in b.similarity_matrix:
#     print([ round(_n, 4) for _n in row ])
p.plot_similarities(b.similarity_matrix)
# p.plot_clusters()
# p.plot_exemplars()
# p.plot_spanning_tree()
# p.plot_linkage_tree()
# p.plot_condensed_tree()
