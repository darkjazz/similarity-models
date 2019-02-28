import json
import numpy as np
import hdbscan as hd
from categories import CategoryFinder
from data import ArtistData

class ArtistClusters:
    def __init__(self):
        self.data = ArtistData()

    def make_clusters(self):
        self.aggregates = []
        self.artists = []
        self.track_count = 0
        for _id in self.data.subdb:
            doc = self.data.subdb.get(_id)
            self.aggregates.append(doc["aggregates"]["mfcc"]["median"])
            self.artists.append({ "name": doc["name"] })
            self.track_count += doc["track_count"]
        data = np.array(self.aggregates)
        self.clusterer = hd.HDBSCAN(min_cluster_size=3, metric='euclidean',
            p=1, min_samples=1, cluster_selection_method='leaf', leaf_size=5,
            prediction_data=True)
        result = self.clusterer.fit(data)
        self.soft_clusters = hd.all_points_membership_vectors(self.clusterer)

    def classify_tracks(self):
        finder = CategoryFinder()
        finder.parse_categories("Classical", 5)
        sum_pct = 0
        for _id in finder.artists:
            artist = finder.artists[_id]
            recs = []
            for recording in artist['recordings']:
                recs.append(self.data.get_feature_vector(recording)['mfcc'])
            # test_labels, strengths = hd.approximate_predict(self.clusterer, recs)
            memb_vec = hd.membership_vector(self.clusterer, np.array(recs))
            classified = [ np.argmax(v) for v in memb_vec ]
            pct = [ c for c in classified if c == 0 or c == 1 ]
            sum_pct += (float(len(pct)) / len(classified))
            print(artist['name'], float(len(pct)) / len(classified))
        print(sum_pct / len(finder.artists))

    def classify_artists(self):
        finder = CategoryFinder()
        finder.parse_categories("Classical", 5)
        aggrs = []
        names = []
        for _id in finder.artists:
            artist = self.data.get_doc(_id)
            artist["aggregates"] = self.data.aggregate_features(artist["recordings"])
            aggrs.append(artist["aggregates"]["median"])
            names.append(artist["name"])
        # test_labels, strengths = hd.approximate_predict(self.clusterer, aggrs)
        memb_vec = hd.membership_vector(self.clusterer, np.array(aggrs))
        classified = [ np.argmax(v) for v in memb_vec ]
        matched = [ 1 for a in classified if a == 0 or a == 1 ]
        for i, name in enumerate(names):
            # print(name, test_labels[i], strengths[i])
            print(name, classified[i])
        print(len(matched) / float(len(classified)))

    def classify_test(self):
        with open('../data/classical-artists.ids') as f:
            flat = f.read()
            f.close()
        aggrs = []
        names = []
        for _id in flat.split("\n"):
            doc = self.get_doc(_id)
            if not doc is None:
                if len(doc["recordings"].keys()) < doc["track_count"]:
                    doc["track_count"] = len(doc["recordings"].keys())
                aggrs.append(self.aggregate_features(doc["recordings"])["median"])
                names.append(doc["name"])
        test_labels, strengths = hd.approximate_predict(self.clusterer, aggrs)
        for i, name in enumerate(names):
            print(name, test_labels[i], strengths[i])

    def print_clusters(self):
        self.threshold = np.min([ np.max(x) for x in self.soft_clusters ])
        if not self.soft_clusters is None:
            for i, cluster in enumerate([ np.argmax(x) for x in self.soft_clusters ]):
                self.artists[i]["cluster"] = cluster
        else:
            for i in range(len(self.clusterer.labels_)):
                self.artists[i]["cluster"] = self.clusterer.labels_[i]
                self.artists[i]["weight"] = self.clusterer.probabilities_[i]
        for i in range(max(self.clusterer.labels_)):
            print("Cluster %s \n--------" % i)
            for name in [a["name"] for a in self.artists if a["cluster"] == i]:
                print(name)
            print("\n\n")

g = ArtistClusters()
g.make_clusters()
g.classify_artists()
# g.classify_tracks()
g.print_clusters()
