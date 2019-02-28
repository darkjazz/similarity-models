import json, random
import numpy as np
import hdbscan as hd
from data import ArtistData

FEATURE = 'mfcc'
MIN_CLUSTER_SIZE = 5

class RecordingsClusters:
	def __init__(self):
		self.data = ArtistData()

	def run(self, feature='mfcc'):
		self.select_data(feature)
		self.make_clusters()
		self.assign_clusters()
		self.collect_artists()

	def select_data(self, feature='mfcc'):
		artists = self.data.get_cluster_artists(feature=feature, use_subset=True)
		features = []
		self.tracks = []
		print("Collecting track features .. ")
		for _id in artists:
			artist = artists[_id]
			[ features.append(_ftr) for _ftr in artist['features'].tolist() ]
			[ self.tracks.append({ 'title': _title, 'artist': artist['name'], '_id': _id }) for _title in artist['recordings'] ]
			self.features = np.array(features)

	def make_clusters(self):
		print("Making clusters ..")
		self.clusterer = hd.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, metric='euclidean',
			p=1, min_samples=1, cluster_selection_method='leaf', leaf_size=5,
			prediction_data=True)
		result = self.clusterer.fit(self.features)
		self.soft_clusters = hd.all_points_membership_vectors(self.clusterer)
		print("Finished making clusters ..")

	def assign_clusters(self):
		print("Assigning clusters ..")
		print(len(self.tracks), len(self.features), len(self.soft_clusters))
		for i, cluster in enumerate([ np.argmax(x) for x in self.soft_clusters ]):
			# print(i, cluster)
			self.tracks[i]["cluster"] = cluster
		print("Clusters assigned ..")

	def collect_artists(self):
		print("Collecting artists ..")
		self.clusters = { }
		self.artists = { }
		[ self.add_artist_to_cluster(t) for t in self.tracks ]
		[ self.add_cluster_to_artists(n) for n in self.clusters ]

	def add_artist_to_cluster(self, track):
		if not track['cluster'] in self.clusters:
			self.clusters[track['cluster']] = { }
		if not track['artist'] in self.clusters[track['cluster']]:
			self.clusters[track['cluster']][track['artist']] = 1
		else:
			self.clusters[track['cluster']][track['artist']] += 1

	def add_cluster_to_artists(self, number):
		cluster = self.clusters[number]
		for _name in cluster:
			if not _name in self.artists:
				self.artists[_name] = [ ]
			self.artists[_name].append({ 'cluster': number, 'weight': 1.0 / cluster[_name] })

	def print_clusters(self):
		for _n in sorted(list(self.clusters.keys())):
			print("\n", "-------\n")
			print(_n, "\n")
			[ print(_name, self.clusters[_n][_name]) for _name in self.clusters[_n] ]

	def print_artists(self):
		for _name in self.artists:
			print(_name)
			artist = self.artists[_name]
			for cluster in artist:
				print(cluster)

if __name__ == "__main__":
	c = RecordingsClusters()
	c.run()
