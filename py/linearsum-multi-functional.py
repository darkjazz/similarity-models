import couchdb, json
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances
import multiprocessing as mp

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
		results.append(r.get())

def assign_sum_pairwise(a, b):
	distance_array = pairwise_distances(a, b)
	row_ind, col_ind = linear_sum_assignment(distance_array)
	return np.sqrt(distance_array[row_ind, col_ind].sum())

if __name__ == '__main__':

	MAX_NEAREST = 13
	MAX_RECS = 7
	DB_PATH = "../data/ab_db.json"

	TEST_IDS = [
	"0003fd17-b083-41fe-83a9-d550bd4f00a1",
	"0004537a-4b12-43eb-a023-04009e738d2e",
	"0005682c-3083-415e-ae4c-debd7be3e47e",
	"0006c824-595a-45af-9374-238ce585fa3a",
	"0008af7d-2aa1-4b4d-80af-b3b64ee3cac6",
	"000ba849-700e-452e-8858-0db591587e4a",
	"000e6dda-32fa-4cdf-8f65-4825e13c5f6f",
	"000fc734-b7e1-4a01-92d1-f544261b43f5",
	"000fecd9-ae03-49bc-9a08-636dde5d405d",
	"00112dec-e09a-462d-86f3-1e6c64ee65a1"
	]

	srv = couchdb.Server()
	sdb = srv["ab_11_plus"]
	pool = mp.Pool(mp.cpu_count())

	results = []

	artists = { }
	for _id in sdb:
		doc = sdb.get(_id)
		artists[_id] = select_recordings(doc)
	ids = list(artists.keys())[:1000]
	print("Artists loaded, assigning sums ... ")

	shrink = ids.copy()
	for _id in ids:
		shrink.remove(_id)
		result = [ pool.apply_async(process_artist, args = (_other, _id, artists)) for _other in shrink ]
		append_result(result)
		print(_id, artists[_id]["name"])

	pool.close()
	pool.join()

	for r in results:
		print(r)
