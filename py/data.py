import couchdb, json
import numpy as np

MAX_RECS = 23
MAX_NEAREST = 13
DB_PATH = "../data/ab_db.json"

class ArtistData:
	def __init__(self):
		srv = couchdb.Server()
		self.sdb = srv['ab_11_plus']
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

	def collect_db(self, artists, sums):
		self.db = { }
		for _item in sums:
			if 'sums' not in artists[_item['_id']]:
				artists[_item['_id']]['sums'] = self.create_sums()
			if 'sums' not in artists[_item['_od']]:
				artists[_item['_od']]['sums'] = self.create_sums()
			for _ftr in _item['sums']:
				artists[_item['_id']]['sums'][_ftr].append({
					'id': _item['_od'],
					'name': artists[_item['_od']]['name'],
					'sum': _item['sums'][_ftr]
				})
				artists[_item['_od']]['sums'][_ftr].append({
					'id': _item['_id'],
					'name': artists[_item['_id']]['name'],
					'sum': _item['sums'][_ftr]
				})
		for _id in artists:
			sums = {}
			for _ftr in artists[_id]["sums"]:
				sums[_ftr] = sorted(artists[_id]["sums"][_ftr], key=lambda x: x["sum"])[:MAX_NEAREST]
			self.db[_id] = sums
		return artists

	def create_sums(self):
		return { "mfcc": [], "chords": [], "rhythm": [] }

	def write_db(self, artists, sums):
		artists = self.collect_db(artists, sums)
		with open(DB_PATH, "w") as write_json:
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
