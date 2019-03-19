from recordings_clusters import RecordingsClusters
from similarity import ArtistSimilarity, TagSimilarity
import numpy as np

class BipartiteClusters:
    def __init__(self):
        self.rec_clusters = RecordingsClusters()

    def make_clusters(self, feature='mfcc', use_soft_clustering=True):
        self.rec_clusters.run(feature, use_soft_clustering=use_soft_clustering)

    def assign_existing_clusters(self, clustering_id):
        self.rec_clusters.load_clusters(clustering_id)

    def calculate_artist_similarity(self, type='max-degree', lmb=1.0, max_similarities = 0, include_self=True):
        self.similarity = ArtistSimilarity(self.rec_clusters.artists, self.rec_clusters.clusters)
        self.output = ""
        self.similarity_matrix = np.zeros((len(self.rec_clusters.artists), len(self.rec_clusters.artists)))
        self.names = list(self.rec_clusters.artists.keys())
        self.artist_similarities = { }
        for _name in self.rec_clusters.artists:
            linked_artists = self.similarity.get_artists(_name, include_self)
            degree = self.similarity.get_artist_degree(_name)
            if type == 'collab':
                similar =  self.similarity.get_collaborative(linked_artists, degree)
            elif type == 'max-degree':
                similar = self.similarity.get_max_degree(linked_artists, degree)
            elif type == 'heat-prob':
                similar = self.similarity.get_heat_prob(linked_artists, degree, lmb)
            else:
                similar = linked_artists
            if max_similarities > 0:
                self.artist_similarities[_name] = sorted(similar, key=lambda x: x["similarity"], reverse=True)[:max_similarities]
            x = self.names.index(_name)
            for _a in similar:
                y = self.names.index(_a['name'])
                self.similarity_matrix[x, y] = _a['similarity']
                self.similarity_matrix[y, x] = _a['similarity']
        # self.save_output()

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

    def get_unique_tag_names(self):
        all_tags = [ ]
        for _n in self.rec_clusters.cluster_tags:
            [ all_tags.append(_tag) for _tag in self.rec_clusters.cluster_tags[_n] ]
        return set(all_tags)

    def print_artists(self):
        for _name in self.artist_similarities:
            similar = self.artist_similarities[_name]
            # self.output += "\n-----\n\n" + name + "\n"
            print("\n-----\n\n", _name)
            # [ print(_artist["name"], _artist["ranking"], round(_artist["heat-prob"], 4)) for _artist in similar if _artist["name"] != name ]
            [ print(_artist["name"], _artist["ranking"], round(_artist["similarity"], 4)) for _artist in similar ]
            # for _a in [ _artist for _artist in similar if _artist["name"] != name ]:
            #     self.output += _a["name"] + " " + str(_a["ranking"]) + " " + str(round(_a["heat-prob"], 4)) + "\n"

    def print_tags(self):
        for _name in self.tag_similarities:
            similar = self.tag_similarities[_name]
            # self.output += "\n-----\n\n" + name + "\n"
            print("\n-----\n\n", _name)
            # [ print(_artist["name"], _artist["ranking"], round(_artist["heat-prob"], 4)) for _artist in similar if _artist["name"] != name ]
            [ print(_tag["name"], _tag["ranking"], round(_tag["similarity"], 4)) for _tag in similar ]
            # for _a in [ _artist for _artist in similar if _artist["name"] != name ]:
            #     self.output += _a["name"] + " " + str(_a["ranking"]) + " " + str(round(_a["heat-prob"], 4)) + "\n"

    def save_output(self):
        with open('../data/heat-prob.data', 'w') as wf:
            wf.write(self.output)
            wf.close()

# if __name__ == "__main__":
#     bg = BipartiteGraph()
#     bg.make_clusters()
#     bg.calculate_filter()
