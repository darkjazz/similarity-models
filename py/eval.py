import json, couchdb
import progressbar as pb
import numpy as np

features = ['chords', 'combined', 'rhythm', 'timbre']
similarities = ['heat-prob-0', 'heat-prob-1', 'max', 'rank']

class Eval:
    def __init__(self):
        srv = couchdb.Server()
        self.dbs = { }
        self.dbs['eval_db'] = srv['eval_db']
        self.eval_db = srv['ab_db_eval']
        for _sim in similarities:
            for _f in features:
                _name = 'ab_db_%s_%s' % (_f, _sim)
                self.dbs[_name] = srv[_name]
        self.artists = { }

    def run(self):
        bar = ProgressBar(max_value=18330)
        c = 0
        for _id in self.dbs['ab_db_timbre_rank']:
            self.get_lists(_id)
            self.evaluate_artist(self.artists[_id])
            c += 1
            bar.update(c)
        bar.finish()

    def get_lists(self, _id):
        self.artists[_id] = { }
        for _name in self.dbs:
            _doc = self.dbs[_name].get(_id)
            self.artists[_id][_name] = _doc['similar']

    def evaluate_artist(self, artist):
        categoriesA = artist.keys()
        categoriesB = copy(categoriesA)
        diversity_matrix = np.full((len(INDEX), len(INDEX)), -1.0)
        for categoryA in categoriesA:
            categoriesB.pop(0)
            for categoryB in categoriesB:
                idsA = (a["id"] for a in artist[categoryA])
                idsB = (b["id"] for b in artist[categoryB])
                count = len(set(idsA) & set(idsB))
                diversity = round(1.0 - (float(count) / float(LIMIT)), 4)
                diversity_matrix[INDEX.index(categoryA)][INDEX.index(categoryB)] = diversity
                diversity_matrix[INDEX.index(categoryB)][INDEX.index(categoryA)] = diversity
        artist["diversity_matrix"] = diversity_matrix.tolist()
        self.save_eval(artist)

    def save_eval(self, artist):
        self.eval_db.save(artist)
        print "Saved ", artist["_id"], artist["name"]

class EvalSum:
    def __init__(self):
        couch_server = couchdb.Server()
        self.eval_db = couch_server[EVAL_DB]
        self.diversity_sum = np.full((len(INDEX), len(INDEX)), 0.0)
        self.diversity_count = np.full((len(INDEX), len(INDEX)), 0.0)

    def run(self):
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
        self.diversity_mean = np.divide(self.diversity_sum, self.diversity_count)
        self.json = {}
        self.json["sum"] = self.diversity_sum.tolist()
        self.json["count"] = self.diversity_count.tolist()
        self.json["mean"] = self.diversity_mean.tolist()
        with open("eval_intersection.js", "w") as write_json:
			write_json.write(json.dumps(self.json))
			write_json.close()
        print self.diversity_sum
        print self.diversity_count
        print self.diversity_mean

    def load(self):
        with open("eval_intersection.js") as json_file:
            self.eval_sum = json.load(json_file)

    def writeCSV(self):
        self.load()
        mean = self.eval_sum["mean"]
        csv = ""
        for name in ABBR:
            csv += "|" + name
        csv += "\n"
        for y in range(len(ABBR)):
            csv += ABBR[y]
            for x in range(len(mean[y])):
                if not np.isnan(mean[y][x]):
                    val = str(round(mean[y][x], 4))
                else:
                    val = "0.0"
                csv += "|" + val
            csv += "\n"
        with open("eval_intersection.csv", "w") as write_csv:
			write_csv.write(csv)
			write_csv.close()

    def writeLatexTable(self):
        self.load()
        mean = self.eval_sum["mean"]
        print mean
        latex = "\\begin{center}\n\\begin{tabular}{||" + self.writeTableHeader() + "||}\n\t\hline\n\t..."
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
                            val = str(round(mean[y][x], 4))
                        else:
                            val = "0.0"
                        latex += " & " + val
                latex += " \\\ \n\t\hline\n"
        latex += "\end{tabular}\n\end{\center}"
        print latex

    def writeTableHeader(self):
        header = "c"
        for check in MASK:
            if check:
                header += " c"
        return header
