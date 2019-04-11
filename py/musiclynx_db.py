from bipartite_clusters_m import BipartiteClusters
from data import ArtistData
import json
from couchdb import Server
import progressbar as bar

NUM_SAVED = 17

class MusicLynxDbBuilder:
    def __init__(self):
        srv = Server()
        self.cdbs = {
            'timbre': srv['ab_db_timbre'],
            'chords': srv['ab_db_chords'],
            'rhythm': srv['ab_db_rhythm'],
            'combined': srv['ab_db_combined']
        }
        self.clusterings = {
            'timbre': '20190228-172841-mfcc',
            'chords': '20190304-090730-chords',
            'rhythm': '20190318-145210-rhythm-H'
        }

    def run(self, similarity='heat-prob', lmb=1.0, num_similar=17):
        self.db = { }
        for _feature in self.clusterings:
            self.process_feature(_feature, similarity, lmb, num_similar)

    def process_feature(self, feature, similarity, lmb, num_similar):
        self.clusters = BipartiteClusters(False)
        self.clusters.assign_existing_clusters(self.clusterings[feature])
        self.clusters.rec_clusters.print_cluster_stats()
        results = self.clusters.calculate_artist_similarity(similarity, lmb, num_similar, False)
        print('saving artists ..')
        b = bar.ProgressBar(max_value=len(results))
        c = 0
        for _tup in results:
            _id = _tup[0]
            self.cdbs[feature].save({ '_id': _id, 'metric': similarity + '_' + str(lmb), 'similar': _tup[1] })
            c += 1
            b.update(c)
        b.finish()
        print('\n')

    def build_from_couch(self):
        data = ArtistData()
        ids = data.load_ids()
        self.db = { }
        for _id in ids:
            self.db[_id] = { }
            for _db_key in self.cdbs:
                _doc = self.cdbs[_db_key].get(_id)
                self.db[_id][_db_key] = _doc['similar'][:13]
            print(_id)
        self.write_db('max')

    def write_db(self, similarity):
        with open('../data/ab_db_%s.json' % similarity, 'w') as wj:
            wj.write(json.dumps(self.db))
            wj.close()

    def save(self, b, m):
        for _id in b.artist_similarities:
            _sim = b.artist_similarities[_id]
            self.cdbs['combined'].save({ '_id': _id, 'metric': m, 'similar': _sim })

for _id in b.artist_similarities:
    _sim = [ { 'id': _a['id'], 'similarity': _a['similarity'], 'degree': _a['degree'], 'ranking': _a['ranking'] } for _a in b.artist_similarities[_id] ]
    m.cdbs['combined'].save({ '_id': _id, 'metric': 'rank', 'similar': _sim })

if __name__ == "__main__":
    db = MusicLynxDbBuilder()
    db.run()
