from recordings_clusters import RecordingsClusters
from similarity import ArtistSimilarity, TagSimilarity
import numpy as np
import progressbar as bar

class BipartiteClusters:
    def __init__(self, use_tags=True):
        self.rec_clusters = RecordingsClusters(use_tags)

    def make_clusters(self, feature='mfcc', use_soft_clustering=True):
        self.rec_clusters.run(feature, use_soft_clustering=use_soft_clustering)

    def assign_existing_clusters(self, clustering_id):
        self.rec_clusters.load_clusters(clustering_id)

    def calculate_artist_similarity(self, type='max-degree', lmb=1.0, max_similarities = 0, include_self=True):
        print("calculating %s similarity" % type)
        self.similarity = ArtistSimilarity(self.rec_clusters.artists, self.rec_clusters.clusters)
        self.output = ""
        self.similarity_matrix = np.zeros((len(self.rec_clusters.artists), len(self.rec_clusters.artists)))
        self.ids = list(self.rec_clusters.artists.keys())
        self.artist_similarities = { }
        b = bar.ProgressBar(max_value=len(self.rec_clusters.artists))
        c = 0
        for _id in self.rec_clusters.artists:
            linked_artists = self.similarity.get_artists(_id, include_self)
            degree = self.similarity.get_artist_degree(_id)
            if type == 'collab':
                similar =  self.similarity.get_collaborative(linked_artists, degree)
            elif type == 'max-degree':
                similar = self.similarity.get_max_degree(linked_artists, degree)
            elif type == 'heat-prob':
                similar = self.similarity.get_heat_prob(linked_artists, degree, lmb)
            else:
                similar = linked_artists
            if max_similarities > 0:
                self.artist_similarities[_id] = sorted(similar, key=lambda x: x["similarity"], reverse=True)[:max_similarities]
            c += 1
            b.update(c)
            # x = self.ids.index(_id)
            # for _a in similar:
            #     y = self.ids.index(_a['id'])
            #     self.similarity_matrix[x, y] = _a['similarity']
            #     self.similarity_matrix[y, x] = _a['similarity']
        # self.save_output()
        b.finish()
        print('\n')

    def calculate_tag_similarity(self, type='max-degree', lmb=1.0, max_similarities = 0):
        self.similarity = TagSimilarity(self.rec_clusters.cluster_tags, self.rec_clusters.clusters)
        self.tag_names = list(self.get_unique_tag_names())
        self.similarity_matrix = np.zeros((len(self.tag_names), len(self.tag_names)))
        self.tag_similarities = { }
        for _name in self.tag_names:
            linked_tags = self.similarity.get_tags(_name)
            degree = self.similarity.get_tag_degree(_name)
            _type = 'ranking'
            if type == 'collab':
                similar =  self.similarity.get_collaborative(linked_tags, degree)
            elif type == 'max-degree':
                similar = self.similarity.get_max_degree(linked_tags, degree)
            elif type == 'heat-prob':
                similar = self.similarity.get_heat_prob(linked_tags, degree, lmb)
            else:
                similar = linked_tags
            if max_similarities > 0:
                self.tag_similarities[_name] = sorted(similar, key=lambda x: x["similarity"], reverse=True)[:max_similarities]
            x = self.tag_names.index(_name)
            for _a in similar:
                y = self.tag_names.index(_a['name'])
                self.similarity_matrix[x, y] = _a['similarity']
                self.similarity_matrix[y, x] = _a['similarity']

    def compare_artists(self, source_id, target_id, type, lmb):
        similarity = ArtistSimilarity(self.rec_clusters.artists, self.rec_clusters.clusters)
        linked_artist = similarity.compare(source_id, target_id)
        degree = self.similarity.get_artist_degree(source_id)
        if type == 'collab':
            similarity =  self.similarity.get_collaborative(linked_artist, degree)
        elif type == 'max-degree':
            similarity = self.similarity.get_max_degree(linked_artist, degree)
        elif type == 'heat-prob':
            similarity = self.similarity.get_heat_prob(linked_artist, degree, lmb)
        else:
            linked_artist[0]['similarity'] = linked_artist[0]['ranking']
        return similarity

    def compare_tags(self, source_tag, target_tag, type, lmb):
        similarity = TagSimilarity(self.rec_clusters.artists, self.rec_clusters.clusters)
        linked_tags = similarity.compare(source_tag, target_tag)
        degree = self.similarity.get_tag_degree(source_tag)
        if type == 'collab':
            similarity =  self.similarity.get_collaborative(linked_tags, degree)
        elif type == 'max-degree':
            similarity = self.similarity.get_max_degree(linked_tags, degree)
        elif type == 'heat-prob':
            similarity = self.similarity.get_heat_prob(linked_tags, degree, lmb)
        else:
            linked_tags[0]['similarity'] = linked_tags[0]['ranking']
        return similarity

    def get_unique_tag_names(self):
        all_tags = [ ]
        for _n in self.rec_clusters.cluster_tags:
            [ all_tags.append(_tag) for _tag in self.rec_clusters.cluster_tags[_n] ]
        return set(all_tags)

    def print_artists(self):
        for _id in self.artist_similarities:
            similar = self.artist_similarities[_id]
            print("\n-----\n\n", self.rec_clusters.names[_id], _id)
            [ print(self.rec_clusters.names[_artist["id"]], _artist["ranking"], round(_artist["similarity"], 4)) for _artist in similar ]

    def print_tags(self):
        for _name in self.tag_similarities:
            similar = self.tag_similarities[_name]
            print("\n-----\n\n", _name)
            [ print(_tag["name"], _tag["ranking"], round(_tag["similarity"], 4)) for _tag in similar ]

    def save_output(self):
        with open('../data/heat-prob.data', 'w') as wf:
            wf.write(self.output)
            wf.close()

# if __name__ == "__main__":
#     bg = BipartiteGraph()
#     bg.make_clusters()
#     bg.calculate_filter()
