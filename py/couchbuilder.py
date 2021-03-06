import couchdb
import numpy as np
from scipy.spatial.distance import euclidean
from copy import copy

class CouchBuilder:
	def __init__(self):
		srv = couchdb.Server()
		self.tdb = srv["ab_o11"]
		# self.sdb = srv["ab_features_o15"]
		self.mdb = srv["similarity_models"]
		print("Connected to databases")

	def convert_recordings(self, recordings):
		obj = {}
		for _title in recordings:
			if _title not in obj:
				obj[_title] = self.get_feature_vectors(recordings[_title])
		return obj

	def get_feature_vectors(self, recording):
		mfcc = eval(recording["mfcc"]["mean"])
		dev = np.sqrt(np.array(eval(recording["mfcc"]["cov"])).diagonal()).tolist()
		mfcc.extend(dev)
		chords = eval(recording["chords_histogram"])
		bps = eval(recording["bpm"]) / 60.0
		onset_rate = eval(recording["onset_rate"])
		return { "title": recording["title"], "mfcc": mfcc, "chords": chords, "rhythm": [bps, onset_rate] }

	def aggregate_features(self, recordings):
		aggr = {}
		mtx = np.matrix([ recordings[ti]["mfcc"] for ti in recordings ])
		mean = np.mean(mtx, 0)
		median = np.median(mtx, 0)
		aggr["mfcc"] = { "mean": mean.tolist()[0], "median": median.tolist()[0] }
		mtx = np.matrix([ recordings[ti]["chords"] for ti in recordings ])
		mean = np.mean(mtx, 0)
		median = np.median(mtx, 0)
		aggr["chords"] = { "mean": mean.tolist()[0], "median": median.tolist()[0] }
		mtx = np.matrix([ recordings[ti]["rhythm"] for ti in recordings ])
		mean = np.mean(mtx, 0)
		median = np.median(mtx, 0)
		aggr["rhythm"] = { "mean": mean.tolist()[0], "median": median.tolist()[0] }
		return aggr

	def update_recordings(self, doc):
		agg = doc["aggregates"]
		for _title in doc["recordings"]:
			rec = doc["recordings"][_title]
			doc["recordings"][_title]["centroid_distances"] = self.calculate_distances(rec, agg)
		return doc

	def calculate_distances(self, recording, aggregates):
		distances = { }
		for _ftr in aggregates:
			distances[_ftr] = euclidean(recording[_ftr], aggregates[_ftr]["mean"])
		return distances

	def get_doc(self, mbid):
		doc = None
		for row in self.sdb.view("views/artist_by_mbid", key=mbid):
			doc = row.value
			doc["recordings"] = self.convert_recordings(doc["recordings"])
		if not doc is None and len(doc["recordings"].keys()) < doc["track_count"]:
			doc["track_count"] = len(doc["recordings"].keys())
		return doc

	def build_db(self):
		count = 0
		for row in self.sdb.view("views/artist_by_mbid"):
			doc = row.value
			_id = row.key
			# print(_id)
			doc['recordings'] = self.convert_recordings(doc['recordings'])
			if not doc is None and len(doc["recordings"].keys()) != doc["track_count"]:
				doc["track_count"] = len(doc["recordings"].keys())
			if doc["track_count"] > 10:
				doc["aggregates"] = self.aggregate_features(doc["recordings"])
				doc = self.update_recordings(doc)
				doc["_id"] = _id
				self.tdb.save(doc)
				count += 1
				print(_id, count)

	def calculate_combined(self):
		for _id in self.mdb:
			if len(_id) == 36:
				print(_id)
				doc = self.tdb.get(_id)
				if doc:
					for _ti in doc['recordings']:
						_rec = doc['recordings'][_ti]
						fvec = copy(_rec['mfcc'])
						fvec.extend(_rec['chords'])
						fvec.extend(_rec['rhythm'])
						# _rec['mfcc'] = _rec['mfcc'][:26]
						doc['recordings'][_ti]['combined'] = list(fvec)
					mtx = np.array( [ doc['recordings'][_ti]['combined'] for _ti in doc['recordings'] ])
					doc['aggregates']['combined'] = { }
					doc['aggregates']['combined']['mean'] = list(np.mean(mtx, 0))
					doc['aggregates']['combined']['median'] = list(np.median(mtx, 0))
					_aggr = np.mean(mtx, 0)
					for _ti in doc['recordings']:
						_rec = doc['recordings'][_ti]
						doc['recordings'][_ti]['centroid_distances']['combined'] = euclidean(_rec['combined'], _aggr)
					re = self.tdb.save(doc)
					print(re)

	def export_ids(self):
		id_str = ""
		for _id in self.tdb:
			id_str += _id + "\n"
		with open('../data/ab_11.id', 'w') as ws:
			ws.write(id_str)

if __name__ == "__main__":
	c = CouchBuilder()
	c.build_db()
