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
		self.sums = []
		for _id in self.ids:
			self.shrink.remove(_id)
			times = []
			for _other in self.shrink:
				t = time.time()
				sums = { }
				for _ftr in self.data.create_sums():
					sums[_ftr] = self.assign_sum_pairwise(self.artists[_id]["recordings"][_ftr], self.artists[_other]["recordings"][_ftr])
				self.sums.append({ '_id': _id, '_od': _other, 'sums': sums })
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

	def assign_sum_pairwise(self, a, b):
		distance_array = pairwise_distances(a, b)
		return np.mean(distance_array)
		# indexes = self.munkres.compute(distance_array.copy())
		# return np.sqrt(np.sum([ distance_array[x][y] for x, y in indexes ]))

	def save(self):
		self.data.write_db(self.artists, self.sums)

if __name__ == "__main__":
	a = time.time()
	w = WeightMatcher()
	w.get_artists()
	w.iterate()
	w.save()
	b = time.time()
	print(b - a)
