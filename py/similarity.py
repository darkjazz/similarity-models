import numpy as np

class Similarity:
    def __init__(self, clusters):
        self.clusters = clusters

    def get_cluster_degree(self, cluster):
        return len(list(self.clusters[cluster].keys()))

    def get_collaborative(self, objects, degree):
        for _object in objects:
            _object['similarity'] = float(_object['ranking']) / float(min(_object['degree'], degree))
        return objects

    def get_max_degree(self, objects, degree):
        for _object in objects:
            weighted = 0
            for _cluster in _object['common_clusters']:
                weighted += float(_object['ranking']) / float(self.get_cluster_degree(_cluster))
            _object['similarity'] = 1.0 / float(max(_object['degree'], degree)) * weighted
        return objects

    def get_heat_prob(self, objects, degree, l=1.0):
        for _object in objects:
            weighted = 0
            for _cluster in _object['common_clusters']:
                weighted += float(_object['ranking']) / float(self.get_cluster_degree(_cluster))
            _object['similarity'] = 1.0 / (degree**(1.0-l) * _object['degree']**l) * weighted
        return objects

class ArtistSimilarity(Similarity):
    def __init__(self, artists, clusters):
        super().__init__(clusters)
        self.artists = artists

    def get_artists(self, id, include_self):
        linked_artists = { }
        for _num in [ _n for _n in self.clusters if id in self.clusters[_n] ]:
            _cluster = self.clusters[_num]
            for _id in _cluster:
                if not _id in linked_artists:
                    linked_artists[_id] = { 'id': _id, 'ranking': 0, 'common_clusters': [ ] }
                linked_artists[_id]['ranking'] += _cluster[_id]
                linked_artists[_id]['common_clusters'].append(_num)
        for _id in linked_artists:
            linked_artists[_id]['degree'] = self.get_artist_degree(_id)
        if not include_self:
            del linked_artists[id]
        return list(linked_artists.values())

    def get_artist_degree(self, id):
        return sum([ _c['weight'] for _c in self.artists[id] ])

class TagSimilarity(Similarity):
    def __init__(self, cluster_tags, clusters):
        super().__init__(clusters)
        self.cluster_tags = cluster_tags

    def get_tag_degree(self, name):
        return sum([ 1 for _n in self.cluster_tags if name in self.cluster_tags[_n] ])

    def get_tags(self, name):
        linked_tags = { }
        for _num in [ _n for _n in self.cluster_tags if name in self.cluster_tags[_n] ]:
            _cluster = self.cluster_tags[_num]
            for _name in _cluster:
                if not _name in linked_tags:
                    linked_tags[_name] = { 'name': _name, 'ranking': 0, 'common_clusters': [ ] }
                linked_tags[_name]['ranking'] += _cluster[_name]
                linked_tags[_name]['common_clusters'].append(_num)
        for _name in linked_tags:
            linked_tags[_name]['degree'] = self.get_tag_degree(_name)
        return list(linked_tags.values())
