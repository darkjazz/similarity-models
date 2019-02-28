import numpy as np

LIMIT = 13

class ArtistFilter:
    def __init__(self, artists, clusters, limit=LIMIT):
        self.artists = artists
        self.clusters = clusters
        self.limit = limit

    def get_artists(self, name):
        linked_artists = { }
        for _num in [ _n for _n in self.clusters if name in self.clusters[_n] ]:
            _cluster = self.clusters[_num]
            for _name in _cluster:
                if not _name in linked_artists:
                    linked_artists[_name] = { 'name': _name, 'ranking': 0, 'common_clusters': [ ] }
                linked_artists[_name]['ranking'] += _cluster[_name]
                linked_artists[_name]['common_clusters'].append(_num)
        return list(linked_artists.values())

    def get_cluster_degree(self, cluster):
        return sum(list(self.clusters[cluster].values()))

    def get_artist_degree(self, name):
        return sum([ (1.0 / _c['weight']) for _c in self.artists[name] ])

    def get_ranking(self, artists):
        return sorted(artists, key=lambda a: a['ranking'], reverse=True)[:self.limit]

    def get_collaborative(self, artists, degree):
        for artist in artists:
            artist_degree = self.get_artist_degree(artist['name'])
            artist['collab'] = float(artist['ranking']) / float(min(artist_degree, degree))
        return sorted(artists, key=lambda a: a['collab'], reverse=True)[:self.limit]

    def get_max_degree(self, artists, degree):
        for artist in artists:
            weighted = 0
            artist_degree = self.get_artist_degree(artist['name'])
            for cluster in artist['common_clusters']:
                weighted += float(artist['ranking']) / float(self.get_cluster_degree(cluster))
            artist['max-degree'] = 1.0 / float(max(artist_degree, degree)) * weighted
        return sorted(artists, key=lambda a: a['max-degree'], reverse=True)[:self.limit]

    def get_heat_prob(self, artists, degree, l=1.0):
        for artist in artists:
            weighted = 0
            artist_degree = self.get_artist_degree(artist['name'])
            for cluster in artist['common_clusters']:
                weighted += float(artist['ranking']) / float(self.get_cluster_degree(cluster))
            artist['heat-prob'] = 1.0 / (degree**(1.0-l)) * (artist_degree**l) * weighted
        return sorted(artists, key=lambda a: a['heat-prob'], reverse=True)[:self.limit]
