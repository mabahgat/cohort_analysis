from pathlib import Path
from typing import List, Dict, Set
from liwc import Liwc

from config import global_config
from lexicons import Lexicon


class LexiconFormatError(ValueError):
	pass


class InvalidLabelError(ValueError):
	pass


class InvalidLexiconName(ValueError):
	pass


class LabelMapper:
	"""
	Maps and excludes labels.
	Label map is expected to map one child label to the another parent label.
	Parent labels are limited by the "Used Labels set"
	"""

	def __init__(self, use_labels: Set[str], label_map: Dict[str, str]):
		"""
		Creates a new instance
		:param use_labels: Set of labels
		:param label_map: Dictionary of labels mapped to others or None for top labels
		"""
		if use_labels is None:
			raise ValueError('List of used labels is None')
		self.__use_labels = use_labels
		self.__label_map = label_map
		self.__cache = {}

	def map(self, label: str) -> str:
		if label in self.__cache:
			return self.__cache[label]
		else:
			new_label = None
			if label in self.__use_labels:
				new_label = label
			elif self.__label_map is not None:
				new_label = self.__search(label)
			self.__cache[label] = new_label
			return new_label

	def __search(self, label: str):
		if label not in self.__label_map:
			raise InvalidLabelError('Unexpected label "{}"'.format(label))
		new_label = self.__label_map[label]
		if new_label is None or new_label in self.__use_labels:
			return new_label
		else:
			return self.__search(new_label)

	def map_list(self, labels: List[str]) -> List[str]:
		mapped_labels = [self.map(label) for label in labels]
		mapped_labels = [label for label in mapped_labels if label is not None]
		return list(dict.fromkeys(mapped_labels))   # remove duplicates

	def get_labels(self):
		return self.__use_labels

	def get_mapping(self):
		return self.__label_map


class LookUpLexicon(Lexicon):
	"""
	Loads a lexicon based on exact match.
	"""

	def __init__(self, name: str, csv_path: Path, sep: str = ','):
		self._path = Path(csv_path)
		self._sep = sep
		self._lookup = self.__load()
		super().__init__(name)

	def __load(self):
		if not self._path.exists():
			raise FileNotFoundError(f'Path invalid or does not exist "{self._path}"')

		lookup_dict = {}
		with open(self._path, mode='r', encoding='utf8') as lexicon_file:
			for line_index, line in enumerate(lexicon_file.readlines()):
				line = line.strip()
				parts = line.split(sep=self._sep)
				if len(parts) < 2:
					raise LexiconFormatError('Unexpected entry in lexicon at line {}: "{}"'.format(line_index, line))
				if len(parts) > 2:
					word = ','.join(parts[0:-1]).strip()
					label = parts[-1].strip()
				else:
					word, label = parts
				word = word.lower()
				label = label.lower()
				if word not in lookup_dict:
					lookup_dict[word] = []
				lookup_dict[word].append(label)
			return lookup_dict

	def label(self, word: str) -> List[str]:
		word = str(word).lower()
		if word not in self._lookup:
			return []
		else:
			return self._lookup[word]


class Liwc2015(Lexicon):

	DEFAULT_LABELS_MAP = {
		'function': None,
		'pronoun': 'function',
		'ppron': 'pronoun',
		'i': 'ppron',
		'we': 'ppron',
		'you': 'ppron',
		'shehe': 'ppron',
		'they': 'ppron',
		'ipron': 'pronoun',
		'article': 'function',
		'prep': 'function',
		'auxverb': 'function',
		'adverb': 'function',
		'conj': 'function',
		'negate': 'function',
		'verb': 'function',
		'adj': 'function',
		'compare': 'function',
		'interrog': 'function',
		'number': 'function',
		'quant': 'function',
		'affect': None,
		'posemo': 'affect',
		'negemo': 'affect',
		'anx': 'negemo',
		'anger': 'negemo',
		'sad': 'negemo',
		'social': None,
		'family': 'social',
		'friend': 'social',
		'female': 'social',
		'male': 'social',
		'cogproc': None,
		'insight': 'cogproc',
		'cause': 'cogproc',
		'discrep': 'cogproc',
		'tentat': 'cogproc',
		'certain': 'cogproc',
		'differ': 'cogproc',
		'percept': None,
		'see': 'percept',
		'hear': 'percept',
		'feel': 'percept',
		'bio': None,
		'body': 'bio',
		'health': 'bio',
		'sexual': 'bio',
		'ingest': 'bio',
		'drives': None,
		'affiliation': 'drives',
		'achiev': 'drives',
		'power': 'drives',
		'reward': 'drives',
		'risk': 'drives',
		'timeorient': None,
		'focuspast': 'timeorient',
		'focuspresent': 'timeorient',
		'focusfuture': 'timeorient',
		'relativ': None,
		'motion': 'relativ',
		'space': 'relativ',
		'time': 'relativ',
		'pconcern': None,
		'work': 'pconcern',
		'leisure': 'pconcern',
		'home': 'pconcern',
		'money': 'pconcern',
		'relig': 'pconcern',
		'death': 'pconcern',
		'informal': None,
		'swear': 'informal',
		'netspeak': 'informal',
		'assent': 'informal',
		'nonflu': 'informal',
		'filler': 'informal'
	}

	def __init__(self,
				 use_labels: Set[str] = None,
				 label_map: Dict[str, str] = DEFAULT_LABELS_MAP):
		self.liwc = Liwc(global_config.liwc2015.dic)

		if use_labels is not None:
			self.__label_mapper = LabelMapper(use_labels=use_labels, label_map=label_map)
		else:
			self.__label_mapper = None

		super().__init__('liwc2015')

	def label(self, word: str) -> List[str]:
		word = str(word).lower()
		labels = self.liwc.search(word)
		if self.__label_mapper is not None:
			return self.__label_mapper.map_list(labels)
		else:
			return labels

	def get_mapper(self):
		return self.__label_mapper


class LookUpLexiconWithMapping(LookUpLexicon):

	def __init__(self,
				 name: str,
				 csv_path: Path,
				 use_labels: Set[str],
				 label_map: Dict[str, str]):
		if use_labels is not None:
			self.__label_mapper = LabelMapper(use_labels=use_labels, label_map=label_map)
		else:
			self.__label_mapper = None
		super().__init__(name=name, csv_path=csv_path)

	def label(self, word: str) -> List[str]:
		word = str(word).lower()
		labels = super().label(word)
		if self.__label_mapper is not None:
			return self.__label_mapper.map_list(labels)
		else:
			return labels

	def used_labels(self):
		return self.__label_mapper

	def get_mapper(self) -> LabelMapper:
		return self.__label_mapper


class Values(LookUpLexiconWithMapping):

	DEFAULT_LABEL_MAP = {
		'autonomy': 'life',
		'creativity': 'cognition',
		'emotion': 'cognition',
		'moral': 'cognition',
		'cognition': 'life',
		'future': 'cognition',
		'thinking': 'cognition',
		'security': 'order',
		'inner-peace': 'order',
		'order': 'life',
		'justice': 'life',
		'advice': 'life',
		'career': 'life',
		'achievement': 'life',
		'wealth': 'life',
		'health': 'life',
		'learning': 'life',
		'nature': 'life',
		'animals': 'life',
		'purpose': 'work-ethic',
		'responsible': 'work-ethic',
		'hard-work': 'work-ethic',
		'work-ethic': None,
		'perseverance': 'work-ethic',
		'feeling-good': None,
		'forgiving': 'accepting-others',
		'accepting-others': None,
		'helping-others': 'society',
		'gratitude': None,
		'dedication': None,
		'self-confidence': None,
		'optimisim': None,
		'honesty': 'truth',
		'truth': None,
		'spirituality': 'religion',
		'religion': None,
		'significant-other': 'relationships',
		'marriage': 'significant-other',
		'friends': 'relationships',
		'relationships': 'social',
		'family': 'relationships',
		'parents': 'family',
		'siblings': 'family',
		'social': None,
		'children': 'family',
		'society': 'social',
		'art': 'life',
		'respect': 'self-confidence',
		'life': None
	}

	DEFAULT_LABEL_SET = {
		'life',
		'parents',
		'truth',
		'religion',
		'social',
		'feeling-good',
		'children',
		'animals',
		'learning',
		'order',
		'accepting-others'
	}

	def __init__(self,
				 use_labels: Set[str] = DEFAULT_LABEL_SET,
				 label_map: Dict[str, str] = DEFAULT_LABEL_MAP):
		super().__init__(name='values',
						 csv_path=global_config.values.csv,
						 use_labels=use_labels,
						 label_map=label_map)


class ExpandedLexicon(Lexicon):

	def __init__(self,
				 source_lexicon: Lexicon,
				 lexicon_expansion: LookUpLexicon):
		self.__source_lexicon = source_lexicon
		self.__lexicon_expansion = lexicon_expansion
		super().__init__(name=source_lexicon.get_name())

	def label(self, word: str) -> List[str]:
		word = str(word).lower()
		labels = self.__source_lexicon.label(word)
		if not labels:
			labels = self.__lexicon_expansion.label(word)
		return labels


class ValuesExpanded(ExpandedLexicon):

	def __init__(self):
		lex = Values()
		limited_labels = lex.get_mapper().get_labels() if lex.get_mapper() is not None else None
		label_mapping = lex.get_mapper().get_mapping() if lex.get_mapper() is not None else None
		expansion = LookUpLexiconWithMapping(name=f'{lex.get_name()}_expanded',
											 csv_path=global_config.values.expansion,
											 use_labels=limited_labels,
											 label_map=label_mapping)
		super().__init__(lex, expansion)


class Liwc2015Expanded(ExpandedLexicon):

	def __init__(self):
		lex = Liwc2015()
		limited_labels = lex.get_mapper().get_labels() if lex.get_mapper() is not None else None
		label_mapping = lex.get_mapper().get_mapping() if lex.get_mapper() is not None else None
		expansion = LookUpLexiconWithMapping(name=f'{lex.get_name()}_expanded',
											 csv_path=global_config.liwc2015.expansion,
											 use_labels=limited_labels,
											 label_map=label_mapping)
		super().__init__(lex, expansion)