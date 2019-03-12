from recordings_clusters import RecordingsClusters
from filter import ArtistFilter
import numpy as np

class BipartiteGraph:
    def __init__(self):
        self.clusters = RecordingsClusters()

    def make_clusters(self, feature='mfcc'):
        self.clusters.run(feature)

    def assign_existing_clusters(self, clusters):
        self.clusters = clusters

    def calculate_filter(self, type='max-degree', lmb=1.0):
        self.filter = ArtistFilter(self.clusters.artists, self.clusters.clusters)
        self.output = ""
        self.similarity_matrix = np.zeros((len(self.clusters.artists), len(self.clusters.artists)))
        self.names = list(self.clusters.artists.keys())
        for _name in self.clusters.artists:
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
