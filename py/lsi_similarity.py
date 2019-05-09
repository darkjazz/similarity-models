from lsi.model import Model
import lsi.config
from lsi.cleaner import TagCleaner
from data import TagData


class LSI:
    def __init__(self):
        self.data = TagData()
        self.model = Model(lsi.config.DICTIONARY_FILENAME, lsi.config.CORPUS_FILENAME)

    def compare(self):
        cleaner = TagCleaner()
        self.model.load()
        for _id in self.data.db:
