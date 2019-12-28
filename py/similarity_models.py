from recordings_clusters import RecordingsClusters
from bipartite_clusters import BipartiteClusters
from data import SimilarityDb, TagData
from plotter import Plotter
import numpy as np
from copy import copy
import json

class SimilarityModels:

	def __init__(self, max=10, use_tags=False, use_subset=False, use_mirex=True):
		self.max = max
		self.use_tags = use_tags
		self.use_subset = use_subset
		self.use_mirex = use_mirex
		self.features = ['mfcc', 'rhythm', 'chords', 'combined']
		self.similarities = ['rank', 'heat-prob', 'max-degree', 'collab']
		self.db = SimilarityDb()

	def collect_categories(self):
		self.categories = [ ]
		for _ftr in self.features:
			self.categories.extend([ '%s-%s' % (_ftr, _sim) for _sim in self.similarities ])
		self.categories.extend(['mirex', 'token', 'lastfm'])

	def run(self):
		# self.db.init_db()
		for _ftr in self.features:
			self.calculate_model(_ftr)
		self.add_token_similarities()
		self.evaluate()
		self.sum_evaluation()
		self.write_csv()

	def calculate_model(self, feature):
		self.clusters = RecordingsClusters(self.use_tags)
		self.clusters.run(feature, use_subset=self.use_subset, use_mirex=self.use_mirex)
		print("cluster stats for", feature)
		self.clusters.print_cluster_stats()
		self.graph = BipartiteClusters(self.use_tags, self.clusters)
		for _sim in self.similarities:
			self.graph.calculate_artist_similarity(_sim, 1.0, self.max, False)
			self.db.save(graph.artist_similarities, feature, _sim)

	def add_token_similarities(self):
		tag_data = TagData()
		for _id in tag_data.tdb:
			doc = self.db.db.get(_id)
			similar = [ { 'id': _a['_id'], 'name': _a['name'] }
				for _a in tag_data.tdb[_id]['similar'][:self.max]
			]
			doc['token'] = similar
			self.db.db.save(doc)

	def evaluate(self):
		self.collect_categories()
		for _id in self.db.db:
			artist = self.db.db.get(_id)
			# if sum([1 for _c in self.categories if _c in artist]) == len(self.categories):
			if 'chords-collab' in artist:
				self.evaluate_artist(artist)
		print("evaluation completed..")

	def evaluate_artist(self, artist):
		categoriesA = copy(self.categories)
		categoriesB = copy(categoriesA)
		size = len(categoriesA)
		similarity_matrix = np.full((size, size), -1.0)
		for categoryA in categoriesA:
			categoriesB.pop(0)
			for categoryB in categoriesB:
				idsA = (a["id"] for a in artist[categoryA])
				idsB = (b["id"] for b in artist[categoryB])
				count = len(set(idsA) & set(idsB))
				similarity = round(float(count) / float(self.max), 4)
				similarity_matrix[categoriesA.index(categoryA)][categoriesA.index(categoryB)] = similarity
				similarity_matrix[categoriesA.index(categoryB)][categoriesA.index(categoryA)] = similarity
		artist["eval"] = {
			"similarity_matrix": similarity_matrix.tolist(),
			"columns": categoriesA
		}
		self.db.db.save(artist)

	def sum_evaluation(self):
		self.collect_categories()
		size = len(self.categories)
		self.similarity_sum = np.full((size, size), 0.0)
		self.similarity_count = np.full((size, size), 0.0)
		for _id in self.db.db:
			if "eval" in self.db.db[_id]:
				matrix = self.db.db[_id]["eval"]["similarity_matrix"]
				for y in range(len(matrix)):
					rowsum = np.array(matrix[y])
					np.place(rowsum, rowsum==-1.0, [np.nan])
					self.similarity_sum[y] = np.nansum(np.vstack((self.similarity_sum[y], rowsum)), axis=0 )
					rowcount = np.array(matrix[y])
					np.place(rowcount, rowcount>-1.0, [1])
					np.place(rowcount, rowcount==-1.0, [0])
					self.similarity_count[y] = np.nansum(np.vstack((self.similarity_count[y], rowcount)), axis=0 )
		self.similarity_mean = np.divide(self.similarity_sum, self.similarity_count, out=np.zeros_like(self.similarity_sum), where=self.similarity_count!=0)
		self.json = {}
		self.json["sum"] = self.similarity_sum.tolist()
		self.json["count"] = self.similarity_count.tolist()
		self.json["mean"] = self.similarity_mean.tolist()
		with open("../data/eval_mirex.json", "w") as write_json:
			write_json.write(json.dumps(self.json))
			write_json.close()
		print("Wrote data to ../data/eval_mirex.json")

	def load_matrix(self):
		self.collect_categories()
		with open("../data/eval_mirex.json", 'r') as json_file:
			self.eval_sum = json.load(json_file)
		self.mean = np.array(self.eval_sum["mean"])

	def write_csv(self):
		self.collect_categories()
		with open("../data/eval_mirex.json", 'r') as json_file:
			self.eval_sum = json.load(json_file)
		mean = self.eval_sum["mean"]
		# print(np.shape(mean))
		csv = ""
		selected_names = self.categories
		all_names = self.categories
		for _name in selected_names:
			csv += "|" + _name
		csv += "\n"
		for _name in selected_names:
			csv += _name
			y = all_names.index(_name)
			for _other in selected_names:
				x = all_names.index(_other)
				if not np.isnan(mean[y][x]):
					val = str(round(mean[y][x], 4))
				else:
					val = "0.0"
				csv += "|" + val
			csv += "\n"

		with open("../data/eval_mirex.csv", "w") as write_csv:
			write_csv.write(csv)
			write_csv.close()

		print("Wrote ../data/eval_mirex.csv")

	def plot_distance_matrix(self, n_iter, perplexity):
		self.load_matrix()
		plotter = Plotter(None)
		plotter.plot_similarities_labeled(1.0-self.mean, self.categories, n_iter, perplexity)
