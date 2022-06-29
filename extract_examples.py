import json
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from category_distances import load_liwc_annotations, load_values_annotations, load_anew_annotations, load_annotations


def load_word_lists():
    """
    Load category word lists; i.e.
    :return:
    """
    cat_load_list = [
        (load_liwc_annotations, "/Users/mbahgat/phd/datasets/reddit/_liwc_tags/voc_liwc_only_header.csv"),
        (load_anew_annotations, "/Users/mbahgat/phd/datasets/reddit/_anew/anew.csv"),
        (load_values_annotations, "/Users/mbahgat/phd/datasets/reddit/_values/values_lexicon.txt")
    ]
    return load_annotations(cat_load_list)


def load_corpus(subreddit_name):
    """

    :param subreddit_name:
    :return:
    """
    object_list = []

    json_corpus_path = '/Users/mbahgat/phd/datasets/reddit/{}/RS_2011-2019.{}.compact_jsonl'\
        .format(subreddit_name, subreddit_name)

    with open(json_corpus_path, 'r') as json_file:
        for line in json_file:
            object_list.append(json.loads(line))

    return object_list


def generate_word_lookup(json_obj):
    """
    Converts text to lists of words for quick lookup
    :param json_obj:
    :return:
    """
    words = []
    words.extend(word_tokenize(json_obj['title']))
    words.extend(word_tokenize(json_obj['selftext']))
    return set(words), words


CACHED_WORD_LOOKUP = 'cached_word_lookup'
TAGGED_TEXT = 'tagged_text'


def with_tag(word, label, tag_start, tag_end):
    """
    Generate a tagged string with label
    :param word:
    :param label:
    :param tag_start:
    :param tag_end:
    :return:
    """
    return tag_start + word + tag_end + '[' + label + ']'


def tag_words(sample, words_to_categories, tag_start='<<<', tag_end='>>>'):
    for i in range(len(sample[TAGGED_TEXT])):
        word = sample[TAGGED_TEXT][i].lower()
        if word in words_to_categories:
            sample[TAGGED_TEXT][i] = with_tag(word, words_to_categories[word], tag_start, tag_end)


def switch_key_value(org_map):
    """
    Switch key-value into a new map, accounting for duplicates in values
    :param org_map:
    :return:
    """
    switched_map = {}
    for label in org_map.keys():
        for word in org_map[label]:
            if word in switched_map:
                switched_map[word] = switched_map[word] + '-' + label
            else:
                switched_map[word] = label
    return switched_map


def find_examples_with_categories(corpus, category_word_lists, lookup_categories):
    found_examples = []
    words_to_categories = switch_key_value(category_word_lists)
    for sample in tqdm(corpus, desc='-'.join(lookup_categories)):
        if CACHED_WORD_LOOKUP not in sample:
            voc, tokenized_text = generate_word_lookup(sample)
            sample[CACHED_WORD_LOOKUP] = voc
            sample[TAGGED_TEXT] = tokenized_text
        found_count = 0
        for category in lookup_categories:
            for word in category_word_lists[category]:
                if word in sample[CACHED_WORD_LOOKUP]:
                    found_count += 1
                    break
        if found_count == len(lookup_categories):
            found_examples.append(sample)
            tag_words(sample, words_to_categories)
    return found_examples


def save_examples(path, examples):
    with open(path, 'w', encoding='utf8') as out_file:
        json.dump(examples, out_file, default=lambda o: 'skipped')


def get_and_save_example(corpus, category_word_lists, subreddit, category_list):
    examples = find_examples_with_categories(corpus, category_word_lists, category_list)
    save_examples('/Users/mbahgat/phd/datasets/reddit/{}/examples_{}_{}.json'.format(subreddit, subreddit,
                                                                                     '_'.join(category_list)),
                  examples)


if __name__ == '__main__':
    subreddit = 'BPD'

    category_word_lists = load_word_lists()
    corpus = load_corpus(subreddit)

    get_and_save_example(corpus, category_word_lists, subreddit, ['i', 'body'])
    get_and_save_example(corpus, category_word_lists, subreddit, ['i', 'feel'])

#    get_and_save_example(corpus, category_word_lists, subreddit, ['i', 'ANIMALS'])
#    get_and_save_example(corpus, category_word_lists, subreddit, ['i', 'death'])
#    get_and_save_example(corpus, category_word_lists, subreddit, ['i', 'relig'])
#     get_and_save_example(corpus, category_word_lists, subreddit, ['i', 'SECURITY'])
#     get_and_save_example(corpus, category_word_lists, subreddit, ['i', 'FRIENDS'])
#     get_and_save_example(corpus, category_word_lists, subreddit, ['i', 'PURPOSE'])
#     get_and_save_example(corpus, category_word_lists, subreddit, ['i', 'sad'])
#
#     get_and_save_example(corpus, category_word_lists, subreddit, ['they', 'death'])
#     get_and_save_example(corpus, category_word_lists, subreddit, ['they', 'RELATIONSHIPS'])
#     get_and_save_example(corpus, category_word_lists, subreddit, ['they', 'female'])
#     get_and_save_example(corpus, category_word_lists, subreddit, ['they', 'friend'])
#     get_and_save_example(corpus, category_word_lists, subreddit, ['they', 'LIFE'])
#     get_and_save_example(corpus, category_word_lists, subreddit, ['they', 'FAMILY'])
#     get_and_save_example(corpus, category_word_lists, subreddit, ['they', 'FRIENDS'])
#     get_and_save_example(corpus, category_word_lists, subreddit, ['they', 'male'])
#     get_and_save_example(corpus, category_word_lists, subreddit, ['they', 'relig'])
#     get_and_save_example(corpus, category_word_lists, subreddit, ['they', 'JUSTICE'])
#
#     get_and_save_example(corpus, category_word_lists, subreddit, ['home', 'RELATIONSHIPS'])
#     get_and_save_example(corpus, category_word_lists, subreddit, ['home', 'SIGNIFICANT-OTHER'])
#     get_and_save_example(corpus, category_word_lists, subreddit, ['home', 'health'])
#     get_and_save_example(corpus, category_word_lists, subreddit, ['home', 'PARENTS'])
#     get_and_save_example(corpus, category_word_lists, subreddit, ['home', 'AUTONOMY'])
