import couchdb, json
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances
import time

MAX_NEAREST = 13
MAX_RECS = 11
DB_PATH = "../data/ab_db.json"
NUM_ARTISTS = 1000

class WeightMatcher:
	def __init__(self):
		srv = couchdb.Server()
		self.sdb = srv["ab_11_plus"]

	def get_artists(self):
		self.artists = { }
		for _id in self.load_ids():
			doc = self.sdb.get(_id)
			self.artists[_id] = self.select_recordings(doc)
		self.ids = list(self.artists.keys())
		print("Artists loaded, assigning sums ... ")

	def select_recordings(self, doc):
		recordings = list(doc["recordings"].values())
		features = { }
		for _ftr in self.create_sums():
			features[_ftr] = np.matrix([ r[_ftr] for r in self.select(recordings, _ftr) ])
		doc["recordings"] = features
		return doc

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

	def print_nearest(self):
		for _id in self.artists:
			artist = self.artists[_id]
			print(artist["name"])
			print(sorted(artist["sums"], key=lambda x: x["sum"])[:MAX_NEAREST])

	def select(self, recordings, feature):
		return sorted(recordings, key=lambda x: x["centroid_distances"][feature])[:MAX_RECS]

	def assign_sum_pairwise(self, a, b):
		distance_array = pairwise_distances(a, b)
		row_ind, col_ind = linear_sum_assignment(distance_array)
		return np.sqrt(distance_array[row_ind, col_ind].sum())

	def assign_sum(self, artistA, artistB):
		source, target = sorted([artistA, artistB], key=lambda x: x["track_count"])
		a = np.matrix([ r["mfcc"].append(r["chords"]) for r in self.select(target["recordings"], source["track_count"]) ])
		b = np.matrix([ r["mfcc"].append(r["chords"]) for r in source["recordings"] ])
		aD = np.repeat(np.matrix.flatten(np.matrix.sum(np.square(a), axis=1)), len(b), 0)
		bD = np.repeat(np.matrix.flatten(np.matrix.sum(np.square(b), axis=1)), len(a), 0).T
		dM = aD + bD - (2 * b * a.T)
		distance_array = np.array(dM)
		row_ind, col_ind = linear_sum_assignment(distance_array)
		return np.sqrt(distance_array[row_ind, col_ind].sum())

	def collect_db(self):
		self.db = {}
		for _id in self.artists:
			sums = {}
			for _ftr in self.artists[_id]["sums"]:
				sums[_ftr] = sorted(self.artists[_id]["sums"][_ftr], key=lambda x: x["sum"])[:MAX_NEAREST]
			self.db[_id] = sums

	def write_db(self):
		self.collect_db()
		with open(DB_PATH, "w") as write_json:
			write_json.write(json.dumps(self.db))
			write_json.close()

	def load_ids(self, limit=None):
		str = ""
		with open('../data/ab_11.id', 'r') as rf:
			str = rf.read()
		ids = str.split("\n")[:-1]
		if limit is None:
			return ids
		else:
			return ids[:limit]

if __name__ == "__main__":
	a = time.time()
	w = WeightMatcher()
	w.get_artists()
	w.iterate()
	w.write_db()
	b = time.time()
	print(b - a)
