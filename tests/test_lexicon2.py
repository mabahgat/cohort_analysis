from lexicons2 import ValuesExpanded, Liwc2015Expanded
from spaces import LabelEmbeddings
from tests.test_spaces import get_sample_content_embeddings


def test_values_expanded():
	lex = ValuesExpanded()
	assert(lex.full_label('brother') == ['values:social'])
	assert(lex.full_label('classcan') == ['values:life'])


def test_liwc2015_expanded():
	lex = Liwc2015Expanded()
	assert('liwc2015:family' in lex.full_label('brother'))
	assert(lex.full_label('classcan') == ['liwc2015:pconcern'])


def test_label_embeddings_distances():
	label_embeddings = LabelEmbeddings([Liwc2015Expanded()], word_embeddings=get_sample_content_embeddings())
	label_embeddings.build()
	dist = label_embeddings.get_distance('liwc2015:drives', 'liwc2015:power')
	assert(0 < dist < 1)
