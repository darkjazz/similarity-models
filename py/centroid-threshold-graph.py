from data import ArtistData
import numpy as np
from sklearn.metrics import pairwise_distances
import time

NUM_ARTISTS = -1
USE_SUBSET = True

class MedianGraph:
	def __init__(self):
		self.data = ArtistData()

	def get_artists(self):
		self.artists = self.data.get_artists(NUM_ARTISTS, USE_SUBSET)
		self.ids = list(self.artists.keys())[:5]

	def iterate(self):
		self.shrink = self.ids.copy()
		for _id in self.ids:
			self.shrink.remove(_id)
			self.artists[_id]["sums"] = self.data.create_sums()
			times = []
			for _other in self.shrink:
				if not "sums" in self.artists[_other]:
					self.artists[_other]["sums"] = self.data.create_sums()
				t = time.time()
				for _ftr in self.artists[_id]["sums"]:
					a = self.artists[_id]["recordings"][_ftr]
					b = self.artists[_other]["recordings"][_ftr]
					centa = self.artists[_id]["aggregates"][_ftr]
					centb = self.artists[_other]["aggregates"][_ftr]
					print(self.artists[_id]["name"], self.artists[_id]["track_count"], self.artists[_other]["track_count"], self.artists[_other]["name"], _ftr)
					sum = self.make_graph(a, b, centa, centb)
					self.artists[_id]["sums"][_ftr].append({ "id": _other, "name": self.artists[_other]["name"], "sum": sum })
					self.artists[_other]["sums"][_ftr].append({ "id": _id, "name": self.artists[_id]["name"], "sum": sum })
				ti = time.time() - t
				times.append(ti)
			if not times:
				times = [0]
			left = len(self.shrink) * len(self.ids) * np.mean(times) / 60.0
			print(_id, self.artists[_id]["name"], round(left, 2), " minutes left")

	def make_graph(self, a, b, centa, centb):
		distance_array = pairwise_distances(a, b)
        median = np.median(distance_array)
		print(np.where())
		# [ print(x) for x in [ y for y in distance_array.tolist() ] ]

if __name__ == "__main__":
	m = MedianGraph()
	m.get_artists()
	m.iterate()
