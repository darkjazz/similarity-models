import couchdb, json
import numpy as np

MAX_RECS = 11
MAX_NEAREST = 13
DB_PATH = "../data/ab_db.json"

class ArtistData:
	def __init__(self):
		srv = couchdb.Server()
		self.sdb = srv["ab_11_plus"]

	def get_artists(self, limit=0, use_subset=False):
		self.artists = { }
		if use_subset:
			ids = self.load_subset()
		else:
			if limit > 0:
				ids = self.load_ids(limit)
			else:
				ids = self.load_ids()
		for _id in ids:
			doc = self.sdb.get(_id)
			if not doc is None:
				self.artists[_id] = self.select_recordings(doc)
		return self.artists

	def select_recordings(self, doc):
		if not doc is None:
			recordings = list(doc["recordings"].values())
			features = { }
			for _ftr in doc["aggregates"]:
				features[_ftr] = np.matrix([ r[_ftr] for r in self.select(recordings, _ftr) ])
			doc["recordings"] = features
		return doc

	def select(self, recordings, feature):
		return sorted(recordings, key=lambda x: x["centroid_distances"][feature])[:MAX_RECS]

	def load_ids(self, limit=None):
		str = ""
		with open('../data/ab_11.id', 'r') as rf:
			str = rf.read()
		ids = str.split("\n")[:-1]
		if limit is None:
			return ids
		else:
			return ids[:limit]

	def load_subset(self):
		with open("../data/ab_subset.json") as js:
			ids = [ j['id'] for j in json.load(js) ]
			js.close()
		return ids

	# def inspect_db(self):
	def collect_db(self, artists):
		self.db = {}
		for _id in artists:
			sums = {}
			for _ftr in artists[_id]["sums"]:
				sums[_ftr] = sorted(artists[_id]["sums"][_ftr], key=lambda x: x["sum"])[:MAX_NEAREST]
			self.db[_id] = sums

	def write_db(self, artists):
		self.collect_db(artists)
		with open(DB_PATH, "w") as write_json:
			write_json.write(json.dumps(self.db))
			write_json.close()
