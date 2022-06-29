import csv
from tqdm import tqdm


def split_and_sort(distance_tuple_list):
    """
    Splits tuples and uses the first item (category 1) as the key and sorts the second item (Category 2)
    based on the third one (distance)
    :param distance_tuple_list:
    :return:
    """
    category_map = {}
    for d_tuple in distance_tuple_list:
        category_1 = d_tuple[0]
        if category_1 not in category_map:
            category_map[category_1] = []
        category_map[category_1].append((d_tuple[1], d_tuple[2]))

    for ref_category in category_map.keys():
        category_map[ref_category].sort(key=lambda pair: pair[1])
        category_map[ref_category] = [pair[0] for pair in category_map[ref_category]]
        category_map[ref_category] = array_to_rank_map(category_map[ref_category])

    return category_map


def load_subreddit_category_distances(path, subreddits):
    """
    Loads category distances for specific subreddits
    :param path:
    :param subreddits:
    :return:
    """
    subreddit_dict = {}
    with open(path, 'r') as distances_file:
        distances_csv = csv.reader(distances_file)
        for row in distances_csv:
            subreddit = row[0]
            if subreddit in subreddits:
                if subreddit not in subreddit_dict:
                    subreddit_dict[subreddit] = []
                subreddit_dict[subreddit].append((row[1], row[2], row[3]))

    for subreddit in subreddit_dict.keys():
        subreddit_dict[subreddit] = split_and_sort(subreddit_dict[subreddit])

    return subreddit_dict


def array_to_rank_map(arr):
    """
    Create content to Index map
    :param arr:
    :return:
    """
    word_to_index = {}
    rank = 0
    for item in arr:
        word_to_index[item] = rank
        rank += 1
    return word_to_index


def compute_rank_deltas_per_category(ref_category_list, other_category_list):
    """
    Computes ranks changes
    :param ref_category_list:
    :param other_category_list:
    :return:
    """
    rank_deltas = []
    for second_category in other_category_list.keys():
        my_rank = other_category_list[second_category]
        ref_rank = ref_category_list[second_category]
        delta = my_rank - ref_rank
        rank_deltas.append((second_category, delta))
    rank_deltas.sort(key=lambda pair: pair[1])
    return rank_deltas


def compute_rank_deltas_per_subreddit(ref_subreddit, other_subreddit):
    """
    Computes ranks changes for each category (cat 1) in the subreddit
    :param ref_subreddit:
    :param other_subreddit:
    :return:
    """
    rank_deltas_map = {}
    for category in other_subreddit.keys():
        rank_deltas_map[category] = compute_rank_deltas_per_category(ref_subreddit[category], other_subreddit[category])
    return rank_deltas_map


def save_category_deltas(path, subreddit_name, rank_deltas_map):
    with open(path, 'w') as out_file:
        for category in rank_deltas_map.keys():
            for rank_delta in rank_deltas_map[category]:
                print('{},{},{},{}'.format(subreddit_name, category, rank_delta[0], rank_delta[1]), file=out_file)


def compute_and_save_ranks(ref_subreddit, subreddit):
    subreddit_distances = load_subreddit_category_distances(
        '/Users/mbahgat/phd/datasets/reddit/all_subreddits_distances.csv',
        [ref_subreddit, subreddit])

    deltas = compute_rank_deltas_per_subreddit(subreddit_distances[ref_subreddit],
                                               subreddit_distances[subreddit])

    save_category_deltas('/Users/mbahgat/phd/datasets/reddit/{}/rank_deltas_{}.{}.csv'
                         .format(subreddit, ref_subreddit, subreddit),
                         subreddit,
                         deltas)


if __name__ == '__main__':
    ref_subreddit = '8plus2'
    subreddits = ['BPD', 'StopSelfHarm', 'SuicideWatch', 'bipolar2', 'depression', 'hardshipmates', 'mentalhealth',
                  'ptsd', 'rapecounseling', 'socialanxiety', 'worldnews', 'IAmA']

    for sr in tqdm(subreddits, desc='Subreddits'):
        compute_and_save_ranks(ref_subreddit, sr)
