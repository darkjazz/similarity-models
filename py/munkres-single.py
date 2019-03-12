import numpy as np
from sklearn.metrics import pairwise_distances
from munkres import Munkres
from data import ArtistData
import time
from plotter import Plotter
from recordings_clusters import RecordingsClusters

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

	def find_outlier(self):
		self.sums

	def calculate_time_left(self, interval):
		total = len(self.ids) * (len(self.ids) - 1)
		left = len(self.shrink) * len(self.ids)
		total_time = interval * float(total)
		time_left = interval * float(left)
		return time_left

	def assign_sum_pairwise(self, a, b):
		# max_length = min(len(a), len(b))
		distance_array = pairwise_distances(a, b)
		# indexes = self.munkres.compute(distance_array.copy())
		# return np.sqrt(np.sum([ distance_array[x][y] for x, y in indexes ]))
		return np.mean(distance_array)

	def save(self):
		self.data.write_db(self.artists, self.sums)

	def plot(self, feature):
		max_dist = 0
		size = len(self.ids)
		matrix = np.zeros((size, size))
		for _sum in self.sums:
			x = self.ids.index(_sum['_id'])
			y = self.ids.index(_sum['_od'])
			matrix[x, y] = _sum['sums'][feature]
			matrix[y, x] = _sum['sums'][feature]
			if feature == 'mfcc':
				if _sum['sums'][feature] > max_dist:
					max_dist = _sum['sums'][feature]
					print(max_dist, self.ids[x], self.ids[y])
		p = Plotter(RecordingsClusters())
		p.plot_similarities(matrix)

if __name__ == "__main__":
	a = time.time()
	w = WeightMatcher()
	w.get_artists()
	w.iterate()
	w.plot('mfcc')
	w.plot('rhythm')
	w.plot('chords')
	# w.save()
	b = time.time()
	print(b - a)
