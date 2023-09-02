from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Set, Union, Dict
from scipy.spatial.distance import cosine as cosine_distance

from gensim.models import Word2Vec
from nltk import word_tokenize
import numpy as np

from lexicons import Lexicon


# Class to create embeddings space from text files in a directory
class WordEmbeddings:
	logger = logging.getLogger(__name__)

	def __init__(self, corpus_path: str, model_path: str, vector_size: int = 100, window_size: int = 5,
				 min_freq_count: int = 1, workers_count: int = 4):
		self.content_path = corpus_path
		self.model_path = model_path
		self.model: Word2Vec = self.__load() if model_path and Path(model_path).exists() else None
		self.__vector_size = vector_size
		self.__window_size = window_size
		self.__min_freq_count = min_freq_count
		self.__workers_count = workers_count

	# Read all text files in the directory and generate one file with all the text
	@staticmethod
	def __read_all_files(dir_path: str):
		dir_path = Path(dir_path)
		text = ''
		for file_namae in os.listdir(dir_path):
			if file_namae.endswith('.txt'):
				text += WordEmbeddings.__read_text(str(dir_path / file_namae))
		return text

	@staticmethod
	def __read_text(file_path: str):
		with open(file_path, 'r', encoding='utf8') as in_file:
			return in_file.read()

	def build(self, force_overwrite: bool = False) -> WordEmbeddings:
		"""
		Build embeddings space
		:param force_overwrite: Rebuild the model even if it exists
		:return: Current instance
		"""
		if self.model is None or force_overwrite:
			corpus = self.__read_corpus()
			self.model = self.__build_word2vec_model(corpus)
			WordEmbeddings.logger.info('Created a new model')
		else:
			WordEmbeddings.logger.info(f'Build step skipped, model was loaded from {self.model_path}')
		return self

	def __read_corpus(self):
		if Path(self.content_path).is_dir():
			return WordEmbeddings.__read_all_files(self.content_path)
		else:
			return WordEmbeddings.__read_text(self.content_path)

	# build word2vec model from text
	def __build_word2vec_model(self, text):
		# tokenize text
		sentences = [word_tokenize(text)]
		# train model
		model = Word2Vec(sentences, vector_size=self.__vector_size, window=self.__window_size,
						 min_count=self.__min_freq_count, workers=self.__workers_count)
		# save model
		if self.model_path is not None:
			model.save(self.model_path)
		return model

	def save(self, model_path: str):
		self.model.save(model_path)
		self.model_path = model_path
		WordEmbeddings.logger.info(f'Model saved at {model_path}')

	# load word2vec model
	def __load(self):
		WordEmbeddings.logger.info(f'Loading embeddings from {self.model_path}')
		return Word2Vec.load(self.model_path)

	# get word2vec vector for word
	def get_vector_for(self, word):
		return self.model.wv[word]

	# get list of words in the model
	def voc(self):
		return list(self.model.wv.key_to_index.keys())


class RankDeltaRecord:

	def __init__(self, label_one: str,
				 label_two: str,
				 current_rank: int,
				 control_rank: int,
				 current_distance: float,
				 control_distance: float):
		self.label_one = label_one
		self.label_two = label_two
		self.current_rank = current_rank
		self.control_rank = control_rank
		self.current_distance = current_distance
		self.control_distance = control_distance
		self.rank_delta = self.current_rank - self.control_rank
		self.distance_delta = self.current_distance - self.control_distance

	def to_dict(self) -> Dict:
		return {
			'label_one': self.label_one,
			'label_two': self.label_two,
			'current_rank': self.current_rank,
			'control_rank': self.control_rank,
			'current_distance': self.current_distance,
			'control_distance': self.control_distance,
			'rank_delta': self.rank_delta,
			'distance_delta': self.distance_delta
		}

	def to_csv_str(self) -> str:
		return f'{self.label_one},{self.label_two},{self.current_rank},{self.control_rank},' \
			   f'{self.current_distance},{self.control_distance},{self.rank_delta},{self.distance_delta}'

	def __repr__(self):
		return str(self.to_dict())


# Class for embeddings space for lexicon labels based on word clusters for each category
class LabelEmbeddings:
	def __init__(self, lexicons: List[Lexicon], word_embeddings: WordEmbeddings, distance_method=cosine_distance):
		"""

		:param lexicons: List of lexicons to use to tag words
		:param word_embeddings:
		"""
		self.__lexicons = lexicons
		self.__source_word_embeddings = word_embeddings
		self.__lexicon_embeddings = {}
		self.__label_distances = None
		self.__distance_method = distance_method

	# for each label in lexicons, get words and their embeddings that have the same label and compute the centroid
	def build(self) -> LabelEmbeddings:
		word_embeddings_per_label = self.__collect_word_embeddings_per_label()
		self.__compute_centroids(word_embeddings_per_label)
		self.__compute_distances()
		return self

	# Collect word embeddings for each label
	def __collect_word_embeddings_per_label(self):
		word_embeddings_per_label = {}
		for word in self.__source_word_embeddings.voc():
			for lexicon in self.__lexicons:
				labels = lexicon.full_label(word)
				if labels:
					for label in labels:
						if label not in word_embeddings_per_label:
							word_embeddings_per_label[label] = []
						word_embeddings_per_label[label].append(self.__source_word_embeddings.get_vector_for(word))
		return word_embeddings_per_label

	# Compute centroids for each label out of the embeddings of the words that have the same label
	def __compute_centroids(self, word_embeddings_per_label):
		for label in word_embeddings_per_label:
			self.__lexicon_embeddings[label] = np.average(word_embeddings_per_label[label], axis=0)

	def __compute_distances(self) -> Dict[str, Dict[str, int]]:
		distances_dict = {}
		for label_one in self.__lexicon_embeddings:
			distances_dict[label_one] = {}
			for label_two in self.__lexicon_embeddings:
				if label_one == label_two:
					continue
				distances_dict[label_one][label_two] = self.__distance_method(self.__lexicon_embeddings[label_one],
																			  self.__lexicon_embeddings[label_two])
		self.__label_distances = distances_dict
		return distances_dict

	def get_distance(self, label_one, label_two) -> float:
		return self.__label_distances[label_one][label_two]

	def get_closest_label_to(self, label: str) -> (str, float):
		return sorted(self.__label_distances[label].items(), key=lambda kv: kv[1])[0]

	def get_labels(self) -> Set[str]:
		return set(self.__lexicon_embeddings.keys())

	# get lexicon embeddings space
	def get_embeddings(self):
		return self.__lexicon_embeddings

	def get_embeddings_for_label(self, label: str):
		return self.__lexicon_embeddings[label]

	# save lexicon embeddings
	def save_to_csv(self, file_path: Union[Path, str]):
		with open(file_path, 'w') as out_file:
			for label in self.__lexicon_embeddings:
				out_file.write(label + ',' + ','.join([str(x) for x in self.__lexicon_embeddings[label]]) + '\n')

	def save_distances_to_csv(self, file_path: Union[Path, str]):
		with open(file_path, 'w') as out_file:
			for label_one in self.__label_distances:
				for label_two in self.__label_distances[label_one]:
					dist = self.__label_distances[label_one][label_two]
					out_file.write(f'{label_one},{label_two},{dist}\n')

	@staticmethod
	def __compute_ranks(label_distances_dict: Dict[str, float]) -> Dict[str, int]:
		ordered = sorted(label_distances_dict.items(), key=lambda kv: kv[1])
		ordered = [label for label, _ in ordered]
		ordered = list(enumerate(ordered))
		return {label: rank for rank, label in ordered}

	def compute_rank_deltas(self, control: LabelEmbeddings, save_to_path: str = None) -> Dict[str, List[RankDeltaRecord]]:
		deltas = {}
		for label_one in self.__label_distances:
			current_ranks = LabelEmbeddings.__compute_ranks(self.__label_distances[label_one])
			control_ranks = LabelEmbeddings.__compute_ranks(control.__label_distances[label_one])
			rank_details = [RankDeltaRecord(label_one,
											label_two,
											current_ranks[label_two],
											control_ranks[label_two],
											self.__label_distances[label_one][label_two],
											control.__label_distances[label_one][label_two])
							for label_two in self.__label_distances if label_one != label_two]
			deltas[label_one] = rank_details

		if save_to_path is not None:
			with open(save_to_path, mode='w') as out_file:
				for deltas_for_label in deltas.values():
					for record in sorted(deltas_for_label, key=lambda r: r.rank_delta, reverse=True):
						print(record.to_csv_str(), file=out_file)
		return deltas
