from bipartite_clusters import BipartiteClusters
import json

class MusicLynxDbBuilder:
    def __init__(self):
        self.clusterings = {
            'timbre': '20190228-171917-mfcc',
            'chords': '20190301-142607-chords',
            'rhythm': '20190301-143100-rhythm'
        }

    def run(self, similarity='heat-prob', lmb=1.0, num_similar=13):
        self.db = { }
        for _feature in self.clusterings:
            self.process_feature(_feature, similarity, lmb, num_similar)
        self.write_db(similarity)

    def process_feature(self, feature, similarity, lmb, num_similar):
        self.clusters = BipartiteClusters()
        self.clusters.assign_existing_clusters(self.clusterings[feature])
        self.clusters.calculate_artist_similarity(similarity, lmb, num_similar, False)
        for _id in self.clusters.artist_similarities:
            if not _id in self.db:
                self.db[_id] = { }
            _sim = self.clusters.artist_similarities[_id]
            self.db[_id][feature] = [ { 'name': self.clusters.rec_clusters.names[_a['id']], 'id': _a['id'], 'div': _a['similarity'] } for _a in _sim ]

    def write_db(self, similarity):
        with open('../data/ab_db_%s.json' % similarity, 'w') as wj:
            wj.write(json.dumps(self.db))
            wj.close()

if __name__ == "__main__":
    db = MusicLynxDbBuilder()
    db.run()
