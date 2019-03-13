from recordings_clusters import RecordingsClusters
from filter import ArtistFilter, TagFilter
import numpy as np

class BipartiteGraph:
    def __init__(self):
        self.rec_clusters = RecordingsClusters()

    def make_clusters(self, feature='mfcc'):
        self.rec_clusters.run(feature)

    def assign_existing_clusters(self, rec_clusters):
        self.rec_clusters = rec_clusters

    def calculate_filter(self, type='max-degree', lmb=1.0):
        self.filter = ArtistFilter(self.rec_clusters.artists, self.rec_clusters.clusters)
        self.output = ""
        self.similarity_matrix = np.zeros((len(self.rec_clusters.artists), len(self.rec_clusters.artists)))
        self.names = list(self.rec_clusters.artists.keys())
        for _name in self.rec_clusters.artists:
            linked_artists = self.filter.get_artists(_name)
            degree = self.filter.get_artist_degree(_name)
            _type = 'ranking'
            if type == 'collab':
                _type = 'collab'
                similar =  self.filter.get_collaborative(linked_artists, degree)
            elif type == 'max-degree':
                _type = 'max-degree'
                similar = self.filter.get_max_degree(linked_artists, degree)
            elif type == 'heat-prob':
                _type = 'heat-prob'
                similar = self.filter.get_heat_prob(linked_artists, degree, lmb)
            else:
                similar = self.filter.get_ranking(linked_artists)
            x = self.names.index(_name)
            for _a in similar:
                y = self.names.index(_a['name'])
                self.similarity_matrix[x, y] = _a[_type]
                self.similarity_matrix[y, x] = _a[_type]
        # self.save_output()

    def calculate_tag_filter(self, type='max-degree', lmb=1.0):
        self.filter = TagFilter(self.rec_clusters.cluster_tags, self.rec_clusters.clusters)
        self.similarity_matrix = np.zeros((len(self.rec_clusters.artists), len(self.rec_clusters.artists)))
        self.tag_names = list(self.get_unique_tag_names())
        for _name in self.tag_names:
            linked_tags = self.filter.get_tags(_name)
            degree = self.filter.get_tag_degree(_name)
            _type = 'ranking'
            if type == 'collab':
                _type = 'collab'
                similar =  self.filter.get_collaborative(linked_tags, degree)
            elif type == 'max-degree':
                _type = 'max-degree'
                similar = self.filter.get_max_degree(linked_tags, degree)
            elif type == 'heat-prob':
                _type = 'heat-prob'
                similar = self.filter.get_heat_prob(linked_tags, degree, lmb)
            else:
                similar = self.filter.get_ranking(linked_tags)
            x = self.tag_names.index(_name)
            for _a in similar:
                y = self.tag_names.index(_a['name'])
                self.similarity_matrix[x, y] = _a[_type]
                self.similarity_matrix[y, x] = _a[_type]

    def get_unique_tag_names(self):
        all_tags = [ ]
        for _n in self.rec_clusters.cluster_tags:
            [ all_tags.append(_tag) for _tag in self.rec_clusters.cluster_tags ]
        return set(all_tags)

    def print_artist(self, name, similar):
        self.output += "\n-----\n\n" + name + "\n"
        print("\n-----\n\n", name)
        # [ print(_artist["name"], _artist["ranking"], round(_artist["heat-prob"], 4)) for _artist in similar if _artist["name"] != name ]
        [ print(_artist["name"], _artist["ranking"], round(_artist["heat-prob"], 4)) for _artist in similar ]
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
