import csv
import sys
import numpy as np
from scipy.spatial.distance import euclidean as euclidean_distance, cosine as cosine_distance
import fasttext
from tqdm import tqdm


def get_marked_indexes(row):
    indexes = []
    for i in range(len(row)):
        if row[i] == 'X':
            indexes.append(i)
    return indexes


def load_liwc_annotations(path):
    """
    Load LIWC annotations
    :param path:
    :return:
    """
    with open(path) as annotations_file:
        annotations_csv = csv.reader(annotations_file)
        header = next(annotations_csv)
        index2class = header[1:]
        categories = {}
        for c in index2class:
            categories[c] = []
        for row in annotations_csv:
            word = row[0]
            category_indexes = get_marked_indexes(row)
            for i in category_indexes:
                categories[index2class[i - 1]].append(word)
        return categories


def load_values_annotations(path):
    """
    Loads words from values lexicon
    :param path:
    :return:
    """
    with open(path) as annotations_file:
        annotations_csv = csv.reader(annotations_file)
        values = {}
        for row in annotations_csv:
            word = row[0].strip()
            value = row[1].strip()
            if value not in values:
                values[value] = []
            values[value].append(word)
    return values


def number_to_level(num):
    """
    Convertts a ANEW lexicon value to high, neutral, low (string)
    :param num:
    :return:
    """
    if num < 4.5:
        return 'low'
    elif num > 5.5:
        return 'high'
    else:
        return 'neutral'


def load_anew_annotations(path):
    """
    Loads ANEW lexicons. Each dimension is abstracted into High, neutral, Low.
    :param path:
    :return:
    """
    categories = {
        'high_valence': [],
        'neutral_valence': [],
        'low_valence': [],
        'high_arousal': [],
        'neutral_arousal': [],
        'low_arousal': [],
        'high_dominance': [],
        'neutral_dominance': [],
        'low_dominance': []
    }

    with open(path) as annotation_file:
        annotations_csv = csv.reader(annotation_file)
        next(annotations_csv)   # skip header
        for row in annotations_csv:
            word = row[0]
            valence = float(row[1])
            arousal = float(row[2])
            dominance = float(row[3])

            categories[number_to_level(valence) + '_valence'].append(word)
            categories[number_to_level(arousal) + '_arousal'].append(word)
            categories[number_to_level(dominance) + '_dominance'].append(word)

    return categories


def load_liwc_category_hierarchy(path):
    """
    Load child parent relation
    :param path:
    :return:
    """
    child2parent = {}
    with open(path) as hierarchy_file:
        hierarchy_csv = csv.reader(hierarchy_file)
        for row in hierarchy_csv:
            child = row[0]
            parent = row[1]
            if child in child2parent:
                raise ValueError("duplicate child key {}".format(child))
            child2parent[child] = parent


def min_category_distance(cat1_vectors, cat2_vectors, distance_function=euclidean_distance, param1=None, param2=None):
    """
    Calculates the minimum distance for two points belonging to different categories
    :param param2:
    :param param1:
    :param distance_function:
    :param cat1_vectors:
    :param cat2_vectors:
    :return:
    """
    min_distance = 1000000000000
    for v1 in cat1_vectors:
        for v2 in cat2_vectors:
            d = distance_function(v1, v2)
            if d < min_distance:
                min_distance = d
    return d


centroid_cache = {}


def centroid(vectors, label=None):
    global centroid_cache
    if label and label in centroid_cache:
        return centroid_cache[label]

    dim_1 = len(vectors)
    dim_2 = len(vectors[0])
    centroid_vector = []
    for i in range(dim_2):
        centroid_vector.append(np.sum(vectors[:, i]) / dim_1)
    np_centroid_vector = np.array(centroid_vector)

    if label:
        centroid_cache[label] = np_centroid_vector
    return np_centroid_vector


def centroid_distance(cat1_vectors, cat2_vectors, distance_function=euclidean_distance, cat1_label=None, cat2_label=None):
    """
    Computes the distance between two categories by comparing the distance between two centroids
    :param cat2_label:
    :param cat1_label:
    :param distance_function:
    :param cat1_vectors:
    :param cat2_vectors:
    :return:
    """
    cat1_centroid = centroid(cat1_vectors, cat1_label)
    cat2_centroid = centroid(cat2_vectors, cat2_label)
    return distance_function(cat1_centroid, cat2_centroid)


def get_vectors(words, model):
    vectors = []
    for w in words:
        vectors.append(model.get_word_vector(w))
    return np.array(vectors)


def category_distance(category_1_name, category_2_name, distance_function, word_table, model, cat_label_prefix=None):
    """
    Returns distance between two categories computed by passed function
    :param cat_label_prefix:
    :param category_1_name:
    :param category_2_name:
    :param distance_function:
    :param word_table:
    :param model:
    :return:
    """
    cat1_label = None
    cat2_label = None
    if cat_label_prefix:
        cat1_label = cat_label_prefix + '_' + category_1_name
        cat2_label = cat_label_prefix + '_' + category_2_name
    cat1_vectors = get_vectors(word_table.get(category_1_name), model)
    cat2_vectors = get_vectors(word_table.get(category_2_name), model)
    return centroid_distance(cat1_vectors, cat2_vectors, distance_function, cat1_label, cat2_label)


def load_annotations(cat_load_list):
    annotations = {}
    for pair in cat_load_list:
        category_annotations = pair[0](pair[1])
        for category_key in category_annotations:
            if category_key not in annotations.keys():
                annotations[category_key] = category_annotations[category_key]
            else:
                annotations[category_key].extend(category_annotations[category_key])
    return annotations


category_to_word_list = None


def compute_and_print(cat_list, cat_load_list, subreddit, distance_func=euclidean_distance, out_files=[sys.stdout]):
    global category_to_word_list, we_model
    if not category_to_word_list:
        category_to_word_list = load_annotations(cat_load_list)

    we_model = fasttext.load_model("/Users/mbahgat/phd/datasets/reddit/{}/we_model/RS_2011-2019.{}.bin"
                                   .format(subreddit, subreddit))

    for i in tqdm(range(len(cat_list)), desc=subreddit):
        for j in range(len(cat_list)):
            if i == j:
                continue
            distance = category_distance(cat_list[i], cat_list[j], distance_func, category_to_word_list, we_model,
                                         subreddit)
            for file in out_files:
                print("{},{},{},{}".format(subreddit, cat_list[i], cat_list[j], distance), file=file)


def load_word_lists():
    cat_load_list = [
        (load_liwc_annotations, "/Users/mbahgat/phd/datasets/reddit/_liwc_tags/voc_liwc_only_header.csv"),
        (load_anew_annotations, "/Users/mbahgat/phd/datasets/reddit/_anew/anew.csv"),
        (load_values_annotations, "/Users/mbahgat/phd/datasets/reddit/_values/values_lexicon.txt")
    ]
    return load_annotations(cat_load_list)


def np_array_to_string(np_array):
    return np.array_str(np_array, max_line_width=1000000).replace('[', '').replace(']', '')


def run(distance_function, file_suffix):
    subreddits = ['BPD', 'IAmA', 'StopSelfHarm', 'SuicideWatch', 'bipolar2', 'depression', 'hardshipmates',
                  'mentalhealth', 'ptsd', 'rapecounseling', 'socialanxiety', 'worldnews']

    print('Running for Subreddits {}'.format(subreddits))

    liwc_categories = ['function', 'pronoun', 'ppron', 'i', 'we', 'you', 'shehe', 'they', 'ipron', 'article', 'prep',
                       'auxverb', 'adverb', 'conj', 'negate', 'verb', 'adj', 'compare', 'interrog', 'number', 'quant',
                       'affect', 'posemo', 'negemo', 'anx', 'anger', 'sad', 'social', 'family', 'friend', 'female',
                       'male', 'cogproc', 'insight', 'cause', 'discrep', 'tentat', 'certain', 'differ', 'percept',
                       'see', 'hear', 'feel', 'bio', 'body', 'health', 'sexual', 'ingest', 'drives', 'affiliation',
                       'achieve', 'power', 'reward', 'risk', 'focuspast', 'focuspresent', 'focusfuture', 'relativ',
                       'motion', 'space', 'time', 'work', 'leisure', 'home', 'money', 'relig', 'death', 'informal',
                       'swear', 'netspeak', 'assent', 'nonflu', 'filler']

    anew_categories = []
    #anew_categories = ['high_valence', 'neutral_valence', 'low_valence', 'high_arousal', 'neutral_arousal',
    #                   'low_arousal', 'high_dominance', 'neutral_dominance', 'low_dominance']

    values_categories = ['ACCEPTING-OTHERS', 'ACHIEVEMENT', 'ADVICE', 'ANIMALS', 'ART', 'AUTONOMY', 'CAREER',
                         'CHILDREN', 'COGNITION', 'CREATIVITY', 'DEDICATION', 'EMOTION', 'FAMILY', 'FEELING-GOOD',
                         'FORGIVING', 'FRIENDS', 'FUTURE', 'GRATITUDE', 'HARD-WORK', 'HEALTH', 'HELPING-OTHERS',
                         'HONESTY', 'INNER-PEACE', 'JUSTICE', 'LEARNING', 'LIFE', 'MARRIAGE', 'MORAL', 'NATURE',
                         'OPTIMISIM', 'ORDER', 'PARENTS', 'PERSEVERANCE', 'PURPOSE', 'RELATIONSHIPS', 'RELIGION',
                         'RESPECT', 'RESPONSIBLE', 'SECURITY', 'SELF-CONFIDENCE', 'SIBLINGS', 'SIGNIFICANT-OTHER',
                         'SOCIAL', 'SOCIETY', 'SPIRITUALITY', 'THINKING', 'TRUTH', 'WEALTH', 'WORK-ETHIC']

    categories = liwc_categories + anew_categories + values_categories

    cat_load_list = [
        (load_liwc_annotations, "/Users/mbahgat/phd/datasets/reddit/_liwc_tags/voc_liwc_only_header.csv"),
        (load_anew_annotations, "/Users/mbahgat/phd/datasets/reddit/_anew/anew.csv"),
        (load_values_annotations, "/Users/mbahgat/phd/datasets/reddit/_values/values_lexicon.txt")
    ]
    with open('/Users/mbahgat/phd/datasets/reddit/all_subreddits_distances.csv2_{}'.format(file_suffix), 'w') as out_file:
        for sr in subreddits:
            compute_and_print(cat_list=categories,
                              cat_load_list=cat_load_list,
                              distance_func=distance_function,
                              subreddit=sr,
                              out_files=[out_file])

    with open('/Users/mbahgat/phd/datasets/reddit/category_centroids.vec2_{}'.format(file_suffix), 'w') as centroids_file:
        for category in centroid_cache:
            subreddit_category_pair = category.split('_')
            print('{},{},{}'.format(subreddit_category_pair[0],
                                    subreddit_category_pair[1],
                                    np_array_to_string(centroid_cache[category])), file=centroids_file)


if __name__ == '__main__':
    run(distance_function=cosine_distance, file_suffix="cosine")
    centroid_cache = {}
    run(distance_function=euclidean_distance, file_suffix="euclidean")
