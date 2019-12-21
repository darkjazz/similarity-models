import couchdb, json
import numpy as np
import progressbar as pb
import httplib2
from urllib.parse import quote
import time
from pymongo import MongoClient, ReturnDocument

MAX_RECS = 37
MAX_NEAREST = 17
DB_PATH = "../data/ab_db_%s.json"
SERVER_URI = 'http://127.0.0.1:8080/lastfm/get_top_tags/'
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB = "deezer"

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

	def get_cluster_artists(self, feature='mfcc', limit=0, use_subset=False, use_mirex=False):
		self.artists = { }
		if use_subset:
			ids = self.load_subset()
		elif use_mirex:
			self.mirex_data = self.load_mirex_data()
			ids = list(self.mirex_data.keys())
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

	def load_mirex_data(self):
		with open('../data/dataset-artist-similarity/mirex_gold.txt') as rf:
			rows = rf.read().split('\n')
		mirex_data = { _id: _row.split(' ') for _id, _row in
			[ (_row.split('\t')[0], _row.split('\t')[1]) for _row in rows[:-1] ]
		}
		return mirex_data

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
		self.tdb = srv["ab_token"]
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

	def get_artist_tokens(self, nlp):
		self.artist_tokens = { }
		bar = pb.ProgressBar(max_value=len(self.db.view("views/tags_by_id")))
		for _i, row in enumerate(self.db.view("views/tags_by_id")):
			_tokens = nlp(" ".join(row.value))
			if _tokens.vector_norm:
				self.artist_tokens[row.key] = _tokens
			bar.update(_i)
		bar.finish()
		print('loaded tokens for %d artists' % len(self.artist_tokens))

	def get_name(self, id):
		name = ""
		for _row in self.db.view("views/name_by_id", key=id):
			name = _row.value
		return name

	def write_db(self, rows):
		sims = { }
		print("building database ..")
		bar = pb.ProgressBar(max_value=len(rows))
		for _i, _sum in enumerate(rows):
			if not _sum['_id'] in sims:
				sims[_sum['_id']] = [ ]
			if not _sum['_od'] in sims:
				sims[_sum['_od']] = [ ]
			sims[_sum['_id']].append({ '_id': _sum['_od'], 'sim': _sum['sim'] })
			sims[_sum['_od']].append({ '_id': _sum['_id'], 'sim': _sum['sim'] })
			bar.update(_i)
		bar.finish()
		bar = pb.ProgressBar(max_value=len(sims))
		for _i, _id in enumerate(sims):
			_similar = sorted(sims[_id], key=lambda x: x["sim"], reverse=True)[:MAX_NEAREST]
			for _artist in _similar:
				_artist["name"] = self.get_name(_artist['_id'])
			_doc = { "_id": _id, "name": self.get_name(_id), "similar": _similar }
			self.tdb.save(_doc)
			bar.update(_i)
		bar.finish()

	def get_tags(self):
		for _id in self.db:
			name = self.get_name(_id)
			uri = SERVER_URI + _id + '/' + quote(name)
			re, co = httplib2.Http().request(uri)
			if re.status == 200:
				res = json.loads(co)
				if isinstance(res, list):
					tags = [ ]
					for _tag in res:
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

class DeezerDb:
	def __init__(self):
		self.mongocli = MongoClient(MONGO_URI)
		self.mongodb = self.mongocli[MONGO_DB]
		self.coll = self.mongodb['artists']

	def compare(self):
		self.map = { }
		o11 = ArtistData()
		c = 0
		for _id in o11.sdb:
			dzr_id = None
			re = self.coll.find_one({ '_id': _id })
			if re and 'deezer_id' in re:
				dzr_id = re['deezer_id']
				self.map[_id] = dzr_id
				print(_id, dzr_id)
				c += 1
		print(c)

class AudioGraphBuilder:
	def __init__(self):
		srv = couchdb.Server()
		self.cdb = srv['ab_db_eval']
		self.tag_data = TagData()
		self.data = ArtistData()
		self.db_path = '../data/ag_db.json'
		self.occ_db_path = '../data/ag_occ_db.json'
		self.db_names = {
			"ab_db_chords_heat-prob-0": "Chords Heat",
			"ab_db_combined_heat-prob-0": "Audio Heat",
			"ab_db_rhythm_heat-prob-0": "Rhythm Heat",
			"ab_db_timbre_heat-prob-0": "Timbre Heat",
			"ab_db_chords_heat-prob-1": "Chords Inference",
			"ab_db_combined_heat-prob-1": "Audio Inference",
			"ab_db_rhythm_heat-prob-1": "Rhythm Inference",
			"ab_db_timbre_heat-prob-1": "Timbre Inference",
			"ab_db_chords_max": "Chords Maximum",
			"ab_db_combined_max": "Audio Maximum",
			"ab_db_rhythm_max": "Rhythm Maximum",
			"ab_db_timbre_max": "Timbre Maximum",
			"ab_db_chords_rank": "Chords Rank",
			"ab_db_combined_rank": "Audio Rank",
			"ab_db_rhythm_rank": "Rhythm Rank",
			"ab_db_timbre_rank": "Timbre Rank"
		}

	def make_db(self, num_artists = 7):
		self.ag_db = { }
		bar = pb.ProgressBar(max_value=len(self.cdb))
		for _i, _id in enumerate(self.cdb):
			doc = self.cdb[_id]
			if "tags" in self.tag_data.db[_id]:
				tags = self.tag_data.db[_id]["tags"][:3]
			_artist = { 'id': doc['_id'], 'name': self.data.get_artist_name(_id) }
			_categories = { }
			for _category_name in self.db_names:
				_artists = [ { 'id': _a['id'], 'name': self.get_name(_a) }
					for _a in doc['categories'][_category_name][:num_artists] ]
				_categories[self.db_names[_category_name]] = _artists
			if "seen live" in tags:
				tags.remove("seen live")
			if tags:
				_artist["tags"] = [ _t.lower() for _t in tags ]
			_artist['categories'] = _categories
			self.ag_db[_id] = _artist
			bar.update(_i)
		bar.finish()
		self.write_db()

	def get_name(self, artist):
		if 'name' in artist:
			return artist['name']
		else:
			return self.data.get_artist_name(artist['id'])

	def make_occurrence_db(self):
		self.read_db()
		self.occ_db = { 'all': { } }
		bar = pb.ProgressBar(max_value=len(self.cdb))
		for _i, _id in enumerate(self.ag_db):
			self.add_artists_occurrences(self.ag_db[_id]['categories'])
			bar.update(_i)
		bar.finish()
		self.write_occ_db()

	def add_artists_occurrences(self, _categories):
		for _name in self.db_names:
			if not _name in self.occ_db:
				self.occ_db[_name] = { }
			for _a in _categories[self.db_names[_name]]:
				if not _a['id'] in self.occ_db[_name]:
					self.occ_db[_name][_a['id']] = 1
				else:
					self.occ_db[_name][_a['id']] += 1
				if not _a['id'] in self.occ_db['all']:
					self.occ_db['all'][_a['id']] = 1
				else:
					self.occ_db['all'][_a['id']] += 1

	def get_artist_occurrences(self):
		for _name in self.occ_db:
			occ = list(self.occ_db[_name].values())
			# ids = list(self.occ_db[_name].keys())
			print(_name)
			print(np.min(occ), np.max(occ), np.mean(occ), np.median(occ))
			threshold = sorted(occ, reverse=True)[11]
			top = { _k:_v for _k, _v in self.occ_db[_name].items() if _v >= threshold }
			for _it in sorted(top.items(), key=lambda kv: kv[1], reverse=True):
				print(self.ag_db[_it[0]]['name'], _it[0], _it[1])
			print("----\n")

	def write_db(self):
		with open(self.db_path, 'w') as wf:
			wf.write(json.dumps(self.ag_db))
			wf.close()

	def read_db(self):
		with open(self.db_path, 'r') as rf:
			self.ag_db = json.load(rf)
			rf.close()

	def write_occ_db(self):
		with open(self.occ_db_path, 'w') as wf:
			wf.write(json.dumps(self.occ_db))
			wf.close()

	def read_occ_db(self):
		with open(self.occ_db_path, 'r') as rf:
			self.occ_db = json.load(rf)
			rf.close()

class SimilarityDb:
	def __init__(self):
		self.artist_data = ArtistData()
		self.db = couchdb.Server()['similarity_models']
		self.load_mirex_data()

	def load_mirex_data(self):
		self.mirex_data = self.artist_data.load_mirex_data()
		self.names = { }
		for _id in self.mirex_data:
			self.names[_id] = self.artist_data.get_artist_name(_id)

	def init_db(self):
		self.load_mirex_data()
		for _id in self.mirex_data:
			doc = { '_id': _id }
			similar = [ { 'id': _oid, 'name': self.names[_oid] } for _oid in self.mirex_data[_id] ]
			doc['mirex'] = similar
			self.db.save(doc)

	def save(self, artist_similarities, feature, similarity):
		for _id in artist_similarities:
			doc = self.db.get(_id)
			similar = [ { 'id': _a['id'], 'name': self.names[_a['id']] } for _a in artist_similarities[_id] ]
			doc['%s-%s' % (feature, similarity)] = similar
			self.db.save(doc)
