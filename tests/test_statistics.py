from lexicons import LIWC2015Lexicon
from statistics import compute_lexicon_voc_coverage
from tests.utils import get_abs_file_path


def test_compute_voc_coverage():
	corpus_path = str(get_abs_file_path(__file__, 'resources/lorem_corpus.txt'))
	coverage = compute_lexicon_voc_coverage(corpus_path=corpus_path, lexicon=LIWC2015Lexicon())
	assert(coverage < 0.1)

	corpus_path_2 = str(get_abs_file_path(__file__, 'resources/content_corpus.txt'))
	coverage = compute_lexicon_voc_coverage(corpus_path=corpus_path_2, lexicon=LIWC2015Lexicon())
	assert(0.5 < coverage < 0.6)
