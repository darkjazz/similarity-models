import couchdb, json
import numpy as np
from munkres import Munkres
from sklearn.metrics import pairwise_distances
import multiprocessing as mp
import time

def select(recordings, feature):
	return sorted(recordings, key=lambda x: x["centroid_distances"][feature])[:MAX_RECS]

def create_sums():
	return { "mfcc": [], "chords": [], "rhythm": [] }

def select_recordings(doc):
	recordings = list(doc["recordings"].values())
	features = { }
	for _ftr in create_sums():
		features[_ftr] = np.matrix([ r[_ftr] for r in select(recordings, _ftr) ])
	doc["recordings"] = features
	return doc

def process_artist(_other, _id, artists):
	sums = { "_id": _id, "_od": _other }
	for _ftr in create_sums():
		sum = assign_sum_pairwise(artists[_id]["recordings"][_ftr], artists[_other]["recordings"][_ftr])
		sums[_ftr] = sum
	return sums

def append_result(result):
	global results
	for r in result:
		results.append(r)

def assign_sum_pairwise(a, b):
	distance_array = pairwise_distances(a, b)
	indexes = self.munkres.compute(distance_array)
	return np.sqrt(np.sum([ distance_array[x][y] for x, y in indexes ]))
	
def load_ids(limit=None):
	str = ""
	with open('../data/ab_11.id', 'r') as rf:
		str = rf.read()
	_ids = str.split("\n")[:-1]
	if limit is None:
		return _ids
	else:
		return _ids[:limit]

if __name__ == '__main__':

	a = time.time()

	MAX_NEAREST = 13
	MAX_RECS = 11
	DB_PATH = "../data/ab_db.json"


	srv = couchdb.Server()
	sdb = srv["ab_11_plus"]
	pool = mp.Pool(mp.cpu_count())

	results = []

	artists = { }
	i = load_ids(100)
	for _id in i:
		doc = sdb.get(_id)
		artists[_id] = select_recordings(doc)
	ids = list(artists.keys())
	print("Artists loaded, assigning sums ... ")

	shrink = ids.copy()
	for _id in ids:
		shrink.remove(_id)
		result = [ pool.apply(process_artist, args = (_other, _id, artists)) for _other in shrink ]
		append_result(result)
		print(_id, artists[_id]["name"])

	pool.close()
	pool.join()

	all_sums = { }

	for r in results:
		_id = r['_id']
		_od = r['_od']
		if not _id in all_sums:
			all_sums[_id] = create_sums()
		if not _od in all_sums:
			all_sums[_od] = create_sums()
		for _ftr in create_sums():
			all_sums[_id][_ftr].append({ '_id': _od, 'name': artists[_od]["name"], 'sum': r[_ftr] })

	db = { }

	for _id in all_sums:
		top_sums = { }
		print(_id)
		for _ftr in all_sums[_id]:
			top_sums[_ftr] = sorted(all_sums[_id][_ftr], key=lambda x: x["sum"])[:MAX_NEAREST]
		db[_id] = top_sums

	with open(DB_PATH, "w") as write_json:
		write_json.write(json.dumps(db))
		write_json.close()

	b = time.time()

	print(b - a)
