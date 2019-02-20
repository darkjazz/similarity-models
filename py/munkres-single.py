import numpy as np
from sklearn.metrics import pairwise_distances
from munkres import Munkres
from data import ArtistData
import time

NUM_ARTISTS = -1
USE_SUBSET = True

class WeightMatcher:
	def __init__(self):
		self.munkres = Munkres()
		self.data = ArtistData()

	def get_artists(self):
		self.artists = self.data.get_artists(NUM_ARTISTS, USE_SUBSET)
		self.ids = list(self.artists.keys())

	def iterate(self):
		self.shrink = self.ids.copy()
		for _id in self.ids:
			self.shrink.remove(_id)
			self.artists[_id]["sums"] = self.create_sums()
			times = []
			for _other in self.shrink:
				if not "sums" in self.artists[_other]:
					self.artists[_other]["sums"] = self.create_sums()
				t = time.time()
				for _ftr in self.artists[_id]["sums"]:
					sum = self.assign_sum_pairwise(self.artists[_id]["recordings"][_ftr], self.artists[_other]["recordings"][_ftr])
					self.artists[_id]["sums"][_ftr].append({ "id": _other, "name": self.artists[_other]["name"], "sum": sum })
					self.artists[_other]["sums"][_ftr].append({ "id": _id, "name": self.artists[_id]["name"], "sum": sum })
				ti = time.time() - t
				times.append(ti)
			if not times:
				times = [0]
			left = len(self.shrink) * len(self.ids) * np.mean(times) / 60.0
			print(_id, self.artists[_id]["name"], round(left, 2), " minutes left")

	def calculate_time_left(self, interval):
		total = len(self.ids) * (len(self.ids) - 1)
		left = len(self.shrink) * len(self.ids)
		total_time = interval * float(total)
		time_left = interval * float(left)
		return time_left

	def create_sums(self):
		return { "mfcc": [], "chords": [], "rhythm": [] }

	def assign_sum_pairwise(self, a, b):
		distance_array = pairwise_distances(a, b)
		indexes = self.munkres.compute(distance_array)
		return np.sqrt(np.sum([ distance_array[x][y] for x, y in indexes ]))

	def save(self):
		self.data.write_db(self.artists)

if __name__ == "__main__":
	a = time.time()
	w = WeightMatcher()
	w.get_artists()
	w.iterate()
	w.save()
	b = time.time()
	print(b - a)
