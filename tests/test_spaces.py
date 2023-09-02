from lexicons import LIWC2015Lexicon, ValuesLexicon
from spaces import WordEmbeddings, LabelEmbeddings
from tests.utils import get_abs_file_path
import pandas as pd


def get_sample_embeddings(source_file_name, tmp_path):
	model_path = str(tmp_path / 'tmp_model') if tmp_path else None
	corpus_path = str(get_abs_file_path(__file__, f'resources/{source_file_name}'))

	embeddings = WordEmbeddings(corpus_path=corpus_path, model_path=model_path).build()
	return embeddings


def get_sample_content_embeddings(tmp_path=None):
	return get_sample_embeddings('content_corpus.txt', tmp_path)


def get_sample_control_embeddings(tmp_path=None):
	return get_sample_embeddings('control_corpus.txt', tmp_path)


def test_word_embeddings_build_save_load(tmp_path):
	get_sample_content_embeddings(tmp_path)

	space_new = WordEmbeddings(corpus_path=None, model_path=str(tmp_path / 'tmp_model'))
	assert(len(space_new.voc()) > 0)

	assert(space_new.get_vector_for('someone') is not None)
	assert(space_new.get_vector_for('questions') is not None)


def test_label_embeddings():
	word_embeddings = get_sample_content_embeddings()
	liwc2015 = LIWC2015Lexicon()
	label_embeddings = LabelEmbeddings([liwc2015], word_embeddings=word_embeddings)
	label_embeddings.build()
	assert(len(label_embeddings.get_labels()) > 0)


def test_label_embeddings_save(tmp_path):
	out_file_path = tmp_path / 'sample_embeddings.csv'
	label_embeddings = LabelEmbeddings([LIWC2015Lexicon()], word_embeddings=get_sample_content_embeddings())
	label_embeddings.build()
	label_embeddings.save_to_csv(out_file_path)

	embeddings_df = pd.read_csv(str(out_file_path), header=None)
	assert(len(embeddings_df) == len(label_embeddings.get_labels()))


def test_label_embeddings_multiple(tmp_path):
	label_embeddings = LabelEmbeddings([LIWC2015Lexicon(), ValuesLexicon()], word_embeddings=get_sample_content_embeddings())
	label_embeddings.build()
	count = len(label_embeddings.get_labels())
	assert(count == 116)


def test_label_embeddings_distances():
	label_embeddings = LabelEmbeddings([LIWC2015Lexicon()], word_embeddings=get_sample_content_embeddings())
	label_embeddings.build()
	dist = label_embeddings.get_distance('liwc2015:drives', 'liwc2015:power')
	assert(0 < dist < 1)


def test_label_embeddings_closest_to():
	label_embeddings = LabelEmbeddings([LIWC2015Lexicon()], word_embeddings=get_sample_content_embeddings())
	label_embeddings.build()
	other_label, dist = label_embeddings.get_closest_label_to('liwc2015:i')
	assert(other_label == 'liwc2015:ppron')
	assert(0 < dist < 1)


def test_label_embeddings_save_distances(tmp_path):
	out_path = tmp_path / 'distances.csv'
	label_embeddings = LabelEmbeddings([LIWC2015Lexicon()], word_embeddings=get_sample_content_embeddings())
	label_embeddings.build()
	label_embeddings.save_distances_to_csv(out_path)

	distances_df = pd.read_csv(str(out_path), header=None)
	assert(len(distances_df) == 5256)


def test_compute_return_deltas(tmp_path):
	label_embeddings = LabelEmbeddings([LIWC2015Lexicon()], word_embeddings=get_sample_content_embeddings())
	label_embeddings.build()

	control_embeddings = LabelEmbeddings([LIWC2015Lexicon()], word_embeddings=get_sample_control_embeddings())
	control_embeddings.build()

	deltas = label_embeddings.compute_rank_deltas(control_embeddings)
	assert(len(deltas['liwc2015:compare']) == 72)


def test_compute_return_deltas_save(tmp_path):
	out_file_path = tmp_path / 'deltas.csv'
	label_embeddings = LabelEmbeddings([LIWC2015Lexicon()], word_embeddings=get_sample_content_embeddings())
	label_embeddings.build()

	control_embeddings = LabelEmbeddings([LIWC2015Lexicon()], word_embeddings=get_sample_control_embeddings())
	control_embeddings.build()

	deltas = label_embeddings.compute_rank_deltas(control_embeddings, str(out_file_path))

	assert(out_file_path.exists())
	embeddings_df = pd.read_csv(str(out_file_path), header=None)
	assert(len(embeddings_df) == 5256)
