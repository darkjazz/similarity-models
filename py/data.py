import couchdb, json
import numpy as np
import progressbar as pb
import httplib2
from urllib.parse import quote
import time

MAX_RECS = 23
MAX_NEAREST = 13
DB_PATH = "../data/ab_db_%s.json"
SERVER_URI = 'http://127.0.0.1:8080/lastfm/get_top_tags/'

class ArtistData:
	def __init__(self):
		srv = couchdb.Server()
		self.sdb = srv['ab_o11']
		self.tdb = srv['ab_sums']
		self.cdb = srv['track_clusters']
		self.gdb = srv["graph_match"]

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

	def get_artist_name(self, id):
		name = ""
		for _row in self.sdb.view('views/name_by_id', key=id):
			name = _row.value
		return name

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

	def save_clustering(self, id):
		with open('../data/clusterings.id', 'a') as wf:
			wf.write(id+'\n')
			wf.close()

	def save_cluster_track(self, track):
		print(track)
		self.cdb.save(track)

	def export_subset(self):
		self.get_artists(0, True)
		subset = { }
		for _id in self.artists:
			doc = self.artists[_id]
			doc['recordings']['mfcc'] = list([ list(_r) for _r in doc['recordings']['mfcc'] ])
			doc['recordings']['chords'] = list([ list(_r) for _r in doc['recordings']['chords'] ])
			doc['recordings']['rhythm'] = list([ list(_r) for _r in doc['recordings']['rhythm'] ])
			subset[_id] = doc
		with open('../data/ab_subset_data.json', 'w') as w:
			w.write(json.dumps(subset))
			w.close()

	def import_subset(self):
		with open('../data/ab_subset_data.json', 'r') as r:
			self.artists = json.load(r)
			r.close()

class ClusterData:
	def __init__(self):
		srv = couchdb.Server()
		self.db = srv["track_clusters"]

	def get_clusters(self, clustering):
		self.tracks = []
		if clustering is None:
			print('yo, wheres the clustering id, fr fcks sake')
		else:
			for row in self.db.view('views/track_by_clustering', key=clustering):
				self.tracks.append(row.value)
			print('%d tracks loaded ..' % len(self.tracks))

class DbMerger:
	def __init__(self):
		srv = couchdb.Server()
		self.qdb = srv['ab_features_o15']
		self.check_sum = 18330
		self.ad = ArtistData()
		self.map = {
			'mfcc': 'timbre', 'chords': 'tonality', 'rhythm': 'rhythm'
		}

	def merge_ab_db(self):
		self.load_files()
		self.load_names()
		self.merge()
		self.write_db()

	def merge(self):
		print("merging data ...")
		self.db = { }
		for _id in self.data['mfcc']:
			self.db[_id] = self.create_sums()
		for _ftr in self.ad.create_sums():
			print("working on %s feature" % _ftr)
			bar = pb.ProgressBar(max_value=len(self.db))
			i = 0
			for _id in self.data[_ftr]:
				for _artist in self.data[_ftr][_id]:
					name = self.names[_artist['_id']]
					id = _artist['_id']
					sum = _artist['sum']
					self.db[_id][self.map[_ftr]].append({ "id": id, "name": name, "div": sum })
				i += 1
				bar.update(i)

	def create_sums(self):
		return { 'timbre': [], 'tonality': [], 'rhythm': [] }

	def load_names(self):
		self.names = { }
		for row in self.qdb.view('views/name_by_mbid'):
			self.names[row.key] = row.value

	def load_files(self):
		print("loading files ...")
		self.data = { }
		for _ftr in self.ad.create_sums():
			with open(DB_PATH % _ftr, 'r') as js:
				self.data[_ftr] = json.load(js)
				js.close()

	def write_db(self):
		print("writing db ...")
		with open(DB_PATH.replace("_%s", ""), "w") as write_json:
			write_json.write(json.dumps(self.db))
			write_json.close()
		print("finished.")

class TagData:
	def __init__(self):
		srv = couchdb.Server()
		self.db = srv["ab_o11"]
		self.uri = SERVER_URI
		# self.load_subset()

	def load_subset(self):
		with open("../data/ab_subset.json") as js:
			self.ids = json.load(js)
			js.close()

	def get_artist_tags(self):
		self.artist_tags = { }
		for row in self.db.view("views/tags_by_id"):
			self.artist_tags[row.key] = row.value
		print('loaded tags for %d artists' % len(self.artist_tags))

	def get_name(self, id):
		name = ""
		for _row in self.db.view("views/name_by_id", key=id):
			name = _row.value
		return name

	def get_tags(self):
		for _id in self.ids:
			name = self.get_name(_id)
			uri = SERVER_URI + _id + '/' + quote(name)
			re, co = httplib2.Http().request(uri)
			if re.status == 200:
				res = json.loads(co)
				if isinstance(res, list):
					tags = [ ]
					for _tag in res[:3]:
						tags.append(_tag['name'])
					doc = self.db.get(_id)
					# print(doc)
					doc['tags'] = tags
					self.db.delete(doc)
					del doc['_rev']
					self.db.save(doc)
					print(_id, tags)
					time.sleep(1.1)

class DbRebuilder:
	def __init__(self, source, target):
		srv = couchdb.Server()
		self.source = srv[source]
		self.target = srv[target]

	def run(self):
		for _id in self.source:
			doc = self.source.get(_id)
			del doc['_rev']
			self.target.save(doc)
			print(_id)
