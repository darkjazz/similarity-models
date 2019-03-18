import json, random
import numpy as np
import hdbscan as hd
from data import ArtistData, ClusterData, TagData
import time, uuid

FEATURE = 'mfcc'
MIN_CLUSTER_SIZE = 11

class RecordingsClusters:
	def __init__(self):
		self.data = ArtistData()
		self.tag_data = TagData()
		self.show_clusterings()
		self.use_soft_clustering = True

	def show_clusterings(self):
		for row in self.data.cdb.view("views/clusterings", group=True):
			print(row.key, row.value)

	def run(self, feature='mfcc', num_artists=-1, use_subset=True, save=False, use_soft_clustering=True):
		self.timestamp = time.strftime("%Y%m%d-%H%M%S")
		self.feature = feature
		self.use_soft_clustering = use_soft_clustering
		start = time.time()
		t = time.time()
		self.select_data(feature, num_artists, use_subset)
		print("processing time: %.3f seconds" % round(time.time() - t, 3))
		t = time.time()
		self.make_clusters(use_soft_clustering=use_soft_clustering)
		print("processing time: %.3f seconds" % round(time.time() - t, 3))
		t = time.time()
		self.assign_clusters()
		print("processing time: %.3f seconds" % round(time.time() - t, 3))
		t = time.time()
		self.collect_artists()
		print("processing time: %.3f seconds" % round(time.time() - t, 3))
		print("total processing time %.3f seconds" % round(time.time() - start, 3))
		if save:
			self.save_tracks()

	def select_data(self, feature, num_artists, use_subset):
		print("selecting recordings ...")
		artists = self.data.get_cluster_artists(feature, num_artists, use_subset)
		features = []
		self.tracks = []
		print("collecting track features ...")
		for _id in artists:
			artist = artists[_id]
			[ features.append(_ftr) for _ftr in artist['features'].tolist() ]
			[ self.tracks.append({ "title": _rec['title'], "artist": artist['name'], "artist_id": _id }) for _rec in artist['recordings'] ]
			self.features = np.array(features)
		print("%d feature vectors of %d tracks by %d artists" % (len(self.features), len(self.tracks), len(artists)))
		print("collecting tag data ..")
		self.tag_data.get_artist_tags()
		print("finished collecting tags ..")

	def make_clusters(self, min_size=11, metric='euclidean', use_soft_clustering=True):
		print("making clusters ..")
		self.use_soft_clustering = use_soft_clustering
		self.clusterer = hd.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, metric='euclidean',
			p=1, min_samples=1, cluster_selection_method='leaf', leaf_size=MIN_CLUSTER_SIZE*2,
			prediction_data=use_soft_clustering)
		result = self.clusterer.fit(self.features)
		if use_soft_clustering:
			self.soft_clusters = hd.all_points_membership_vectors(self.clusterer)
		print("finished making clusters ..")

	def assign_clusters(self):
		print("assigning clusters ..")
		if self.use_soft_clustering:
			for i, cluster in enumerate([ np.argmax(x) for x in self.soft_clusters ]):
				self.tracks[i]["cluster"] = cluster
				self.tracks[i]["weight"] = self.soft_clusters[i][cluster]
		else:
			for i in range(len(self.clusterer.labels_)):
				self.tracks[i]["cluster"] = self.clusterer.labels_[i]
				self.tracks[i]["weight"] = self.clusterer.probabilities_[i]
		print("clusters assigned ..")

	def save_tracks(self):
		print("saving tracks ..")
		for _track in self.tracks:
			savetrack = {
				"_id": str(uuid.uuid4()),
				"clustering": self.timestamp + "-" + self.feature,
				"cluster": str(_track["cluster"]),
				"title": _track["title"],
				"artist": _track["artist"],
				"artist_id": _track["artist_id"]
			}
			self.data.save_cluster_track(savetrack)
		print("finished saving ..")

	def collect_artists(self):
		print("collecting artists ..")
		self.clusters = { }
		self.artists = { }
		self.cluster_tags = { }
		[ self.add_artist_to_cluster(t) for t in self.tracks ]
		[ self.add_cluster_to_artists(n) for n in self.clusters ]
		[ self.add_tags_to_cluster(t) for t in self.tracks ]

	def add_artist_to_cluster(self, track):
		if not track['cluster'] in self.clusters:
			self.clusters[track['cluster']] = { }
		if not track['artist'] in self.clusters[track['cluster']]:
			self.clusters[track['cluster']][track['artist']] = track["weight"]
		else:
			self.clusters[track['cluster']][track['artist']] += track["weight"]
		self.add_tags_to_cluster(track)

	def add_tags_to_cluster(self, track):
		cluster = self.clusters[track['cluster']]
		artist_id = track["artist_id"]
		if not track['cluster'] in self.cluster_tags:
			self.cluster_tags[track['cluster']] = { }
		for _tag in self.tag_data.artist_tags[track['artist_id']]:
			if _tag != 'seen live':
				clean_tag = _tag.lower().replace("-", " ")
				if not clean_tag in self.cluster_tags[track['cluster']]:
					self.cluster_tags[track['cluster']][clean_tag] = track["weight"]
				else:
					self.cluster_tags[track['cluster']][clean_tag] += track["weight"]

	def add_cluster_to_artists(self, number):
		cluster = self.clusters[number]
		for _name in cluster:
			if not _name in self.artists:
				self.artists[_name] = [ ]
			self.artists[_name].append({ 'cluster': number, 'weight': cluster[_name] })

	def load_clusters(self, id):
		cluster_data = ClusterData()
		cluster_data.get_clusters(id)
		self.tracks = cluster_data.tracks
		self.tag_data.get_artist_tags()
		self.collect_artists()

	def print_clusters(self):
		for _n in sorted(list(self.clusters.keys())):
			print("\n", "-------\n")
			print(_n, "\n")
			print(sorted(self.cluster_tags[_n], key=self.cluster_tags[_n].get, reverse=True)[:3])
			print(self.cluster_tags[_n])
			[ print(_name, self.clusters[_n][_name]) for _name in self.clusters[_n] ]

	# def print_track_clusters(self):
	# 	if self.use_soft_clustering:
	# 		for i, cluster in enumerate([ np.argmax(x) for x in self.soft_clusters ]):
	# 			self.tracks[i]["cluster"] = cluster
	# 			self.tracks[i]["weight"] = self.soft_clusters[i][cluster]
	# 	else:
	#         for i in range(len(self.clusterer.labels_)):
	#             self.tracks[i]["cluster"] = self.clusterer.labels_[i]
	#             self.tracks[i]["weight"] = self.clusterer.probabilities_[i]

	def print_artists(self):
		for _name in self.artists:
			print(_name)
			artist = self.artists[_name]
			for cluster in artist:
				print(cluster)

	def print_cluster_stats(self):
		sizes = [ len(self.clusters[c]) for c in self.clusters ]
		num = len(self.clusters)
		max = np.max(sizes)
		min = np.min(sizes)
		mean = np.mean(sizes)
		median = np.median(sizes)
		print("num %d | max %d | min %d | mean %d | median %d" % (num, max, min, mean, median))

if __name__ == "__main__":
	c = RecordingsClusters()
	c.run()
