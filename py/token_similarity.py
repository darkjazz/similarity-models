import spacy
from data import TagData
import progressbar as pb

class TokenSimilarity:
	def __init__(self):
		self.nlp = spacy.load("en_core_web_lg")
		self.data = TagData()

	def run(self):
		self.data.get_artist_tokens(self.nlp)
		self.ids = list(self.data.artist_tokens.keys())
		shrink = self.ids.copy()
		self.rows = []
		bar = pb.ProgressBar(max_value=len(self.ids))
		for _i, _id in enumerate(self.ids):
			shrink.remove(_id)
			_tokenA = self.data.artist_tokens[_id]
			for _other in shrink:
				_tokenB = self.data.artist_tokens[_other]
				_sim = tokenA.similarity(tokenB)
				self.rows.append({ '_id': _id, '_od': _other, 'sim': _sim })
			bar.update(_i)
		bar.finish()

	def save(self):
		self.data.write_db(self.rows)
