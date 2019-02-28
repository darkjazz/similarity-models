from recordings_clusters import RecordingsClusters
from filter import ArtistFilter

class BipartiteGraph:
    def __init__(self):
        self.clusters = RecordingsClusters()

    def make_clusters(self, feature='mfcc'):
        self.clusters.run(feature)

    def calculate_filter(self, type='max-degree', lmb=1.0):
        self.filter = ArtistFilter(self.clusters.artists, self.clusters.clusters)
        for _name in self.clusters.artists:
            linked_artists = self.filter.get_artists(_name)
            degree = self.filter.get_artist_degree(_name)
            if type == 'collab':
                similar =  self.filter.get_collaborative(linked_artists, degree)
            elif type == 'max-degree':
                similar = self.filter.get_max_degree(linked_artists, degree)
            elif type == 'heat-prob':
                similar = self.filter.get_heat_prob(linked_artists, degree, lmb)
            else:
                similar = self.filter.get_ranking(linked_artists)
            self.print_artist(_name, similar)

    def print_artist(self, name, similar):
        print("\n-----\n\n", name)
        [ print(_artist) for _artist in similar if _artist["name"] != name ]

if __name__ == "__main__":
    bg = BipartiteGraph()
    bg.make_clusters()
    bg.calculate_filter()
