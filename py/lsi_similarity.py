from lsi.model import Model
import lsi.config
from lsi.cleaner import TagCleaner
from data import TagData
import progressbar as pb

NUM_TAGS = 5

class LSI:
    def __init__(self):
        self.data = TagData()
        self.model = Model(lsi.config.DICTIONARY_FILENAME, lsi.config.CORPUS_FILENAME)

    def load_tags(self):
        cleaner = TagCleaner()
        self.tags = { }
        for _row in self.data.db.view('views/tags_by_id'):
            self.tags[_row.key] = [ cleaner.clean(_t) for _t in _row.value[:NUM_TAGS] ]
        self.ids = list(self.tags.keys())

    def compare(self):
        self.shrink = self.ids.copy()
        self.model.load()
        self.lsi = [ ]
        bar = pb.ProgressBar(max_value=len(self.ids)*(len(self.ids)-1))
        c = 0
        for _id in self.ids:
            self.shrink.remove(_id)
            for _other in self.shrink:
                sim = self.calculate_similarity(self.tags[_id], self.tags[_other])
                self.lsi.append({ 'id': _id, 'od': _other, 'sim': sim })
                c += 1
                bar.update(c)
        bar.finish()

    def calculate_similarity(self, a, b):
        sim = 0.0
        num = 0
        for _tagA in a:
            for _tagB in b:
                pct = self.model.get_pairwise_similarity(_tagA, _tagB)
                # print(_tagA, _tagB, pct)
                if not pct is None:
                    sim += pct
                    num += 1
        if num == 0:
            return -1
        else:
            return (sim / num)
