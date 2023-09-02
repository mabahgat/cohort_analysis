from config import global_config
from abc import ABC, abstractmethod
from typing import List

from liwc import Liwc


# abstract class for lexicons to label words with categories
class Lexicon(ABC):

	def __init__(self, name: str):
		self.name = name

	@abstractmethod
	def label(self, word: str) -> List[str]:
		pass

	def full_label(self, word: str) -> List[str]:
		labels = self.label(word)
		return [f'{self.name}:{l}' for l in labels]

	def get_name(self):
		return self.name


# LIWC 2015 lexicon
class LIWC2015Lexicon(Lexicon):
	def __init__(self):
		self.liwc = Liwc(global_config.liwc2015.dic)
		super().__init__('liwc2015')

	def label(self, word):
		return self.liwc.search(word)


# lexicon for look up words stored in a csv file with word and label pairs
class LookupLexicon(Lexicon):
	def __init__(self, name, filepath):
		self.word_to_label = {}
		with open(filepath, 'r') as f:
			for line in f:
				word, label = line.strip().split(',')
				self.word_to_label[word] = label
		super().__init__(name)

	def label(self, word):
		if word in self.word_to_label:
			return [self.word_to_label[word]]
		else:
			return []


# liwc 22 lookup lexicon
class LIWC22Lexicon(LookupLexicon):
	def __init__(self):
		super().__init__('liwc22', global_config.liwc22.csv)


# Values lookup lexicon
class ValuesLexicon(LookupLexicon):
	def __init__(self):
		super().__init__('values', global_config.values.csv)
