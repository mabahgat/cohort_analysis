from nltk import word_tokenize
from typing import Dict
from tqdm import tqdm

from lexicons import Lexicon


def __build_voc(corpus_path: str) -> Dict[str, int]:
	voc_freq = {}
	with open(corpus_path, mode='r', encoding='utf8') as corpus_file:
		for line in tqdm(corpus_file, desc=f'building voc for {corpus_path}'):
			words = word_tokenize(line)
			for w in words:
				if w not in voc_freq:
					voc_freq[w] = 0
				voc_freq[w] += 1
	return voc_freq


def compute_lexicon_voc_coverage(corpus_path: str, lexicon: Lexicon) -> float:
	voc = set(__build_voc(corpus_path=corpus_path).keys())
	found_voc = set([w for w in voc if len(lexicon.label(w)) > 0])
	return float(len(found_voc)) / len(voc)
