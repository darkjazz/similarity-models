import couchdb, json
import numpy as np

MAX_RECS = 23
MAX_NEAREST = 13
DB_PATH = "../data/ab_db_%s.json"

class ArtistData:
	def __init__(self):
		srv = couchdb.Server()
		self.sdb = srv['ab_o11']
		self.tdb = srv['ab_sums']

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
				features[_ftr] = np.array([ r[_ftr] for r in self.select(recordings, _ftr) ])
			doc["recordings"] = features
		return doc

	def select(self, recordings, feature):
		return sorted(recordings, key=lambda x: x["centroid_distances"][feature])[:MAX_RECS]

	def get_cluster_artists(self, feature='mfcc', limit=0, use_subset=False):
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
				self.artists[_id] = self.select_cluster_recordings(doc, feature)
		return self.artists

	def select_cluster_recordings(self, doc, feature):
		if not doc is None:
			recordings = list(doc["recordings"].values())
			recordings = self.select(recordings, feature)
			features = np.array([ r[feature] for r in recordings  ])
			doc["features"] = features
			doc["recordings"] = recordings
		return doc

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

	def inspect_db(self, feature, limit=5):
		with open(DB_PATH, "r") as js:
			self.db = json.load(js)
			js.close()
		for _id in self.db:
			artist = self.db[_id]
			doc = self.sdb.get(_id)
			print("\n\n", doc["name"])
			print("----")
			for similar in artist[feature][:limit]:
				print(similar)

	def collect_db(self, feature):
		self.db = { }
		self.ids = self.load_ids()
		rows = [ ]
		for _id in self.ids:
			sums = self.get_doc(_id, feature)
			for _sum in sums:
				rows.append({ '_id': _id, '_od': _sum['_id'], 'sum': _sum['sum'] })
			print(_id)
		for _sum in rows:
			if not _sum['_id'] in self.db:
				self.db[_sum['_id']] = [ ]
			if not _sum['_od'] in self.db:
				self.db[_sum['_od']] = [ ]
			self.db[_sum['_id']].append({ '_id': _sum['_od'], 'sum': _sum['sum'] })
			self.db[_sum['_od']].append({ '_id': _sum['_id'], 'sum': _sum['sum'] })
		for _id in self.db:
			self.db[_id] = sorted(self.db[_id], key=lambda x: x["sum"])[:MAX_NEAREST]
			print(_id)

	def get_doc(self, id, feature):
		doc = self.tdb.get(id)
		return doc['sums'][feature]

	def create_sums(self):
		return { "mfcc": [], "chords": [], "rhythm": [] }

	def write_db(self, feature):
		self.collect_db(feature)
		with open(DB_PATH % feature, "w") as write_json:
			write_json.write(json.dumps(self.db))
			write_json.close()

	def save(self, artists, sums):
		artists = self.collect_db(artists, sums)
		for _id in self.db:
			sums = self.db[_id]
			sums['_id'] = _id
			self.tdb.save(sums)

	def get_statistics(self):
		track_counts = []
		for _id in self.sdb:
			doc = self.sdb.get(_id)
			track_counts.append(doc["track_count"])
		print("Total tracks", np.sum(track_counts))
		print("Mean # tracks", np.mean(track_counts))
		print("Median # tracks", np.median(track_counts))

	def get_subset_statistics(self):
		track_counts = []
		artist_count = 0
		self.ids = self.load_subset()
		for _id in self.ids:
			doc = self.sdb.get(_id)
			if not doc is None:
				artist_count += 1
				track_counts.append(doc["track_count"])
		print("Total artists", np.sum(artist_count))
		print("Total tracks", np.sum(track_counts))
		print("Mean # tracks", np.mean(track_counts))
		print("Median # tracks", np.median(track_counts))
