import matplotlib.pyplot as pl
import seaborn as sb
from sklearn.manifold import TSNE
import numpy as np
import json

class Plotter:
    def __init__(self, clusters):
        self.track_clusters = clusters

    def plot_clusters(self):
        self.clusterer = self.track_clusters.clusterer
        tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=5000)
        emb = tsne.fit_transform(self.track_clusters.features)
        color_palette = sb.color_palette('Paired', max(self.clusterer.labels_) + 1)
        cluster_colors = [color_palette[np.argmax(x)] for x in self.track_clusters.soft_clusters]
        pl.scatter(*emb.T,s=13, linewidth=0, c=cluster_colors, alpha=0.75)
        pl.show()

    def plot_exemplars(self):
        self.clusterer = self.track_clusters.clusterer
        exemplars = [ ]
        exemplar_labels = [ ]
        for i, _array in enumerate(self.clusterer.exemplars_):
            [ exemplars.append(a) for a in _array ]
            [ exemplar_labels.append(i) for a in _array ]
        # print(exemplars)
        # print(exemplar_labels)
        tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=2000)
        emb = tsne.fit_transform(exemplars)
        color_palette = sb.color_palette('Paired', max(exemplar_labels) + 1)
        cluster_colors = [color_palette[c] for c in exemplar_labels]
        pl.scatter(*emb.T,s=23, linewidth=0, c=cluster_colors, alpha=0.75)
        pl.show()

    def plot_similarities(self, matrix):
        tsne = TSNE(n_components=2, verbose=1, metric='precomputed')
        emb = tsne.fit_transform(matrix)
        # print(np.shape(emb))
        pl.scatter(*emb.T,s=17, linewidth=0, alpha=0.75)
        pl.show()

    def plot_similarities_labeled(self, matrix, labels, n_iter, perplexity):
        tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=n_iter, metric='precomputed')
        emb = tsne.fit_transform(matrix)
        # print(np.shape(emb))
        pl.scatter(*emb.T,s=57, linewidth=0, alpha=0.75)
        x = emb.T[0]
        y = emb.T[1]
        for i, txt in enumerate(labels):
            pl.text(x[i] - 13, y[i] + 11, txt, fontsize=11)
        pl.show()

    def plot_spanning_tree(self):
        self.clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis', edge_alpha=0.6, node_size=10, edge_linewidth=2)
        pl.show()

    def plot_linkage_tree(self):
        self.clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
        pl.show()

    def plot_condensed_tree(self):
        self.clusterer.condensed_tree_.plot()
        pl.show()

    def plot_histogram(self, ftr):
        with open('../data/' + ftr + '_artist_degrees.deg', 'r') as rf:
            x = json.load(rf)
            rf.close()
        pl.hist(x, 23, facecolor='b', alpha=0.7)
        pl.xlabel('Degree')
        pl.ylabel('# of artists')
        pl.title('Artist degree in %s clusters' % ftr)
        pl.grid(True)
        pl.show()
