from recordings_clusters import RecordingsClusters
from bipartite_clusters import BipartiteClusters
from data import SimilarityDb

class SimilarityModels:

    def __init__(self, max, use_tags, use_subset, use_mirex):
        self.max = max
        self.use_tags = use_tags
        self.use_subset = use_subset
        self.use_mirex = use_mirex
        self.features = ['mfcc', 'rhythm', 'chords']
        self.similarities = ['heat-prob', 'max-degree', 'collab']
        self.db = SimilarityDb()

    def run(self):
        for _ftr in self.features:
            self.calculate_model(_ftr)

    def calculate_model(self, feature):
        clusters = RecordingsClusters(self.use_tags)
        clusters.run(feature, use_subset=self.use_subset, use_mirex=self.use_mirex)
        print("cluster stats for", feature)
        clusters.print_cluster_stats()
        graph = BipartiteClusters(self.use_tags, clusters)
        for _sim in self.similarities:
            graph.calculate_artist_similarity(_sim, 1.0, self.max, False)
            self.db.save(graph.artist_similarities, feature, _sim)
