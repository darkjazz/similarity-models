import json, couchdb, time
import progressbar as pb
import numpy as np
from copy import copy

features = ['chords', 'combined', 'rhythm', 'timbre']
similarities = ['heat-prob-0', 'heat-prob-1', 'max', 'rank']
LIMIT = 17

INCL = {
	'ch_h0': True,
	'co_h0': True,
	'rh_h0': True,
	'ti_h0': True,
	'ch_h1': True,
	'co_h1': True,
	'rh_h1': True,
	'ti_h1': True,
	'ch_mx': True,
	'co_mx': True,
	'rh_mx': True,
	'ti_mx': True,
	'ch_rk': True,
	'co_rk': True,
	'rh_rk': True,
	'ti_rk': True,
	'lfm': True,
	'coll': True,
	'glob': True,
	'max': True,
	'ti_gm': True,
	'rh_gm': True,
	'co_gm': True
}
# MASK = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
# ABBR = ['ch_h0', 'co_h0', 'rh_h0', 'ti_h0', 'ch_h1', 'co_h1', 'rh_h1', 'ti_h1', 'ch_mx', 'co_mx', 'rh_mx', 'ti_mx', 'ch_rk', 'co_rk', 'rh_rk', 'ti_rk', 'collab', 'lfm']
DBP_CAT = ['lastfm', 'collab', 'global', 'max-degree', 'timbre-gmm', 'rhythm-gmm', 'tonality-gmm']

class Eval:
	def __init__(self):
		srv = couchdb.Server()
		self.dbs = { }
		self.eval_db = srv['eval_db']
		self.ab_eval_db = srv['ab_db_eval']
		for _sim in similarities:
			for _f in features:
				_name = 'ab_db_%s_%s' % (_f, _sim)
				self.dbs[_name] = srv[_name]
		self.artists = { }

	def run(self):
		bar = pb.ProgressBar(max_value=18330)
		c = 0
		for _id in self.dbs['ab_o11']:
			if len(_id) == 36:
				self.get_lists(_id)
				self.evaluate_artist(self.artists[_id])
				c += 1
				bar.update(c)
		bar.finish()

	def get_lists(self, _id):
		self.artists[_id] = { '_id': _id, 'categories': { } }
		for _name in self.dbs:
			_doc = self.dbs[_name].get(_id)
			self.artists[_id]['categories'][_name] = _doc['similar']
		_doc = None
		_doc = self.eval_db.get(_id)
		if not _doc is None:
			for _category in DBP_CAT:
				if _category in _doc['categories']:
					self.artists[_id]['categories'][_category] = _doc['categories'][_category]

	def evaluate_artist(self, artist):
		categoriesA = list(artist['categories'].keys())
		categoriesB = copy(categoriesA)
		size = len(categoriesA)
		for _category in DBP_CAT:
			if not _category in categoriesA:
				size += 1
		diversity_matrix = np.full((size, size), -1.0)
		for categoryA in categoriesA:
			categoriesB.pop(0)
			for categoryB in categoriesB:
				idsA = (a["id"] for a in artist['categories'][categoryA])
				idsB = (b["id"] for b in artist['categories'][categoryB])
				count = len(set(idsA) & set(idsB))
				diversity = round(1.0 - (float(count) / float(LIMIT)), 4)
				diversity_matrix[categoriesA.index(categoryA)][categoriesA.index(categoryB)] = diversity
				diversity_matrix[categoriesA.index(categoryB)][categoriesA.index(categoryA)] = diversity
		artist["diversity_matrix"] = diversity_matrix.tolist()
		artist['columns'] = categoriesA
		self.save_eval(artist)

	def save_eval(self, artist):
		self.ab_eval_db.save(artist)
		# print(artist["_id"])

	def make_tag_table(self, limit=5):
		with open('../data/tag_stats.stats', 'r') as rf:
			tag_data = json.load(rf)
			rf.close()
		tbl = " ".join([ "c" for _h in tag_data ])
		tbl = "\\begin{center}\n\\begin{tabular}{||" + tbl + "||}\n\t\hline\n\t"
		tbl += " & ".join([ _ftr for _ftr in tag_data ])
		tbl += " \\\ \n\t\hline\n"
		for _i in range(limit):
			tbl += " & ".join([ _tag['tag'] + " (" + str(_tag['degree']) + ")" for _tag in [ tag_data[_ftr][_i] for _ftr in tag_data ]])
			tbl += " \\\ \n\t\hline\n"
		tbl += "\end{tabular}\n\end{\center}"
		print(tbl)

class EvalSum:
	def __init__(self):
		couch_server = couchdb.Server()
		self.eval_db = couch_server['ab_db_eval']
		size = 23
		self.diversity_sum = np.full((size, size), 0.0)
		self.diversity_count = np.full((size, size), 0.0)

	def run(self):
		bar = pb.ProgressBar(max_value=18330)
		c = 0
		for _id in self.eval_db:
			matrix = self.eval_db[_id]["diversity_matrix"]
			for y in range(len(matrix)):
				rowsum = np.array(matrix[y])
				np.place(rowsum, rowsum==-1.0, [np.nan])
				self.diversity_sum[y] = np.nansum(np.vstack((self.diversity_sum[y], rowsum)), axis=0 )
				rowcount = np.array(matrix[y])
				np.place(rowcount, rowcount>-1.0, [1])
				np.place(rowcount, rowcount==-1.0, [0])
				self.diversity_count[y] = np.nansum(np.vstack((self.diversity_count[y], rowcount)), axis=0 )
			c+= 1
			bar.update(c)
		bar.finish()
		self.diversity_mean = np.divide(self.diversity_sum, self.diversity_count)
		self.json = {}
		self.json["sum"] = self.diversity_sum.tolist()
		self.json["count"] = self.diversity_count.tolist()
		self.json["mean"] = self.diversity_mean.tolist()
		with open("../data/eval_all.json", "w") as write_json:
			write_json.write(json.dumps(self.json))
			write_json.close()
		print(self.diversity_sum)
		print(self.diversity_count)
		print(self.diversity_mean)

	def load(self):
		with open("../data/eval_all.json", 'r') as json_file:
			self.eval_sum = json.load(json_file)

	def write_csv(self):
		self.load()
		mean = self.eval_sum["mean"]
		print(np.shape(mean))
		csv = ""
		selected_names = [ _k for (_k,_v) in INCL.items() if _v ]
		all_names = list(INCL.keys())
		print(selected_names, INCL.items())
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

		with open("../data/eval.csv", "w") as write_csv:
			write_csv.write(csv)
			write_csv.close()

	def write_latex_table(self, ftr):
		self.load()
		mean = self.eval_sum["mean"]
		print(mean)
		latex = "\\begin{center}\n\\begin{tabular}{||" + self.write_table_header() + "||}\n\t\hline\n\t..."
		for i, _abbr in enumerate(ABBR):
			if _abbr.find(ftr) == 0 or _abbr == 'lfm':
				MASK[i] = True
			else:
				MASK[i] = False
		for name in ABBR:
			if MASK[ABBR.index(name)]:
				latex += " & " + name.lower()
		latex += " \\\ \n\t\hline\n"
		for y in range(len(ABBR)):
			if MASK[y]:
				latex += ABBR[y].lower()
				for x in range(len(mean[y])):
					if MASK[x]:
						if not np.isnan(mean[y][x]):
							val = str(round(mean[y][x], 3))
						else:
							val = "0.0"
						latex += " & " + val
				latex += " \\\ \n\t\hline\n"
		latex += "\end{tabular}\n\end{\center}"
		print(latex)

	def write_table_header(self):
		header = "c"
		for check in MASK:
			if check:
				header += " c"
		return header
