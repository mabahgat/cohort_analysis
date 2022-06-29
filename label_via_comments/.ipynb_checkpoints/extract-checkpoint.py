import json
import tqdm
import csv
import math
import re
import random

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = '/Users/mbahgat/phd/datasets/reddit/_comment_analysis/'

MENTAL_SUBMISSIONS_PATH = ROOT + "RS_201908_mental.jsonl"
CONTROL_SUBMISSIONS_PATH = ROOT + "RS_201908_IAmA.jsonl"
MENTAL_COMMENTS_PATH = ROOT + "RC_201908_mental.jsonl"
CONTROL_COMMENTS_PATH = ROOT + "RC_201908_IAmA.jsonl"
COMMENT_COUNT_PATH = ROOT + "comments_count_per_submission.freq"

MENTAL_SUBMISSION_IDS_PATH = ROOT + "RS_201908_mental_ids.lst"
MENTAL_SUBMISSION_COUNTS_PATH = ROOT + "mental_comments_count_per_submission.freq"
CONTROL_SUBMISSION_COUNTS_PATH = ROOT + 'IAmA_comments_count_per_submission.freq'

MENTAL_SUBMISSIONS_WITH_COMMENTS_PATH = ROOT + 'RS_C_201908_mental.jsonl'
CONTROL_SUBMISSIONS_WITH_COMMENTS_PATH = ROOT + 'RS_C_201908_IAmA.jsonl'
MENTAL_WITH_FULL_INFO_PATH = ROOT + 'RS_FI_201908.json'
CONTROL_WITH_FULL_INFO_PATH = ROOT + 'RS_FI_201908_IAmA.json'

# LIWC prep directories
CONTROL_COMMENTS_TEXT_DIR_PATH = ROOT + 'iama_comments/'
CONTROL_COMMENTS_GROUPED_TEXT_DIR_PATH = ROOT + 'iama_comments_grouped/'
MENTAL_SUBMISSIONS_TEXT_DIR_PATH = ROOT + 'mental_submissions/'
CONTROL_SUBMISSIONS_TEXT_DIR_PATH = ROOT + 'iama_submissions/'
MENTAL_COMMENTS_ALLONE_DIR_PATH = ROOT + 'all_comments_per_submission/'
MENTAL_COMMENTS_SEPARATE_DIR_PATH = ROOT + 'spearated_comments_per_submission/'

# LIWC files
CONTROL_SEPARATE_PER_SUBMISSION_LIWC_PATH = ROOT + 'RC_201908_IAmA.liwc.csv'
CONTROL_GROUPED_PER_SUBMISSION_LIWC_PATH = ROOT + 'RC_201908_IAmA_grouped.liwc.csv'
CONTROL_SUBMISSIONS_LIWC_PATH = ROOT + 'RS_201908_IAmA.liwc.csv'
MENTAL_GROUPED_PER_SUBMISSION_LIWC_PATH = ROOT + 'RC_201908_mental_grouped.liwc.csv'
MENTAL_SEPARATE_PER_SUBMISSION_LIWC_PATH = ROOT + 'RC_201908_mental_separate.liwc.csv'
MENTAL_SUBMISSIONS_LIWC_PATH = ROOT + 'RS_201908_mental.liwc.csv'

LIWC_LABELS_PATH = ROOT + 'liwc_labels.csv'

# Reference files
CONTROL_COMMENTS_AVERAGE_SEPARATE_LIWC_PATH = ROOT + 'RC_201908_IAMA.avg.csv'
CONTROL_COMMENTS_AVERAGE_GROUPED_LIWC_PATH = ROOT + 'RC_201908_IAMA.avg_grouped.csv'
CONTROL_SUBMISSIONS_AVERAGE_LIWC_PATH = ROOT + 'RS_201908_IAmA.avg.csv'


def adjust_id(parent_id):
    """
    Parent IDs where given with an extra "t0_" as a prefix. Remove it
    :param parent_id:
    :return:
    """
    return parent_id.split('_')[1]


def load_comment_counts(counts_path, filter_path=None):
    """
    Loads comment count, with optionally filtering it for a specific list of IDs
    :param counts_path:
    :param filter_path:
    :return:
    """
    if filter_path:
        with open(filter_path, 'r') as filter_file:
            filter_ids = set([ids.strip() for ids in filter_file.readlines()])
    with open(counts_path, 'r') as counts_file:
        counts_dict = {}
        for count_id_pair in tqdm.tqdm(counts_file.readlines(), "Reading counts from {}: ".format(counts_path)):
            (count, pid) = count_id_pair.strip().split(' ');
            if not filter_path or pid in filter_ids:
                counts_dict[pid] = int(count)
        return counts_dict


def load_posts(content_file_path):
    """
    Loads from jsonl files with the whole json object on a single file
    :param content_file_path:
    :return:
    """
    with open(content_file_path, 'r', encoding='utf8') as content_file:
        return [json.loads(line) for line in tqdm.tqdm(content_file.readlines(), desc='Reading posts from {}'.format(content_file_path))]


def load_posts_from_json(content_file_path):
    """
    Loads posts from json (not jsonl) file
    :param content_file_path:
    :return:
    """
    with open(content_file_path, 'r', encoding='utf8') as content_file:
        return json.load(content_file)


def get_text(post_list):
    """
    extract text from the body field
    :param post_list:
    :return:
    """
    return [post['body'] for post in post_list]


def write_each_post_in_file(post_list, out_dir_path, get_text):
    """
    Write text files so it can be parsed for LIWC
    :param post_list:
    :param out_dir_path:
    :param get_text:
    :return:
    """
    for post in tqdm.tqdm(post_list, desc='Writing posts to file in path: {}'.format(out_dir_path)):
        write_to_file('{}{}.txt'.format(out_dir_path, post['id']), get_text(post))


def write_to_file(file_path, text):
    """
    Write text to file
    :param file_path:
    :param text:
    :return:
    """
    with open(file_path, 'w', encoding='utf8') as out_file:
        print(text, file=out_file)


def load_liwc(liwc_csv_path):
    """
    Reads LIWC file into a map between post ID and LWIC values
    :param liwc_csv_path:
    :return:
    """
    post_liwc_dict = {}
    with open(liwc_csv_path, 'r') as liwc_csv_file:
        csv_reader = csv.reader(liwc_csv_file)
        next(csv_reader) # skip header
        for row in csv_reader:
            ids = row[0].replace('.txt', '')
            liwc_values = [float(v) for v in row[1:]]
            if len(liwc_values) == 0:
                continue
            post_liwc_dict[ids] = liwc_values
        return post_liwc_dict


def average_liwc(post_liwc_dict):
    """
    Computes the average for LIWC values for all posts passed
    :param post_liwc_dict:
    :return:
    """
    sum = [0 for i in range(len(next(iter(post_liwc_dict.values()))))]
    count = 0
    for post_id in post_liwc_dict.keys():
        values = post_liwc_dict[post_id]
        sum = [sum_elm + values_elm for sum_elm, values_elm in zip(sum, values)]
        count += 1
    return [float(value) / count for value in sum]


def save_list(write_list, out_path):
    """
    Store a list in a file with eacah element in a line
    :param write_list:
    :param out_path:
    :return:
    """
    with open(out_path, 'w') as out_file:
        [print(item, file=out_file) for item in write_list]


def save_comment_counts(comment_counts, out_file_path):
    """
    Save comment counts in the same format as the original file
    :param comment_counts:
    :param out_file_path:
    :return:
    """
    with open(out_file_path, 'w') as out_file:
        [print('{} {}'.format(count, ids), file=out_file) for (ids, count) in comment_counts.items()]


def save_json(json_object, out_file_path):
    """
    Saves content as json (not jsonl)
    :param json_object:
    :param out_file_path:
    :return:
    """
    with open(out_file_path, 'w', encoding='utf8') as out_file:
        json.dump(json_object, out_file, indent=2)


def load_list(in_path, process_function=lambda x: x):
    """
    Reads a list from file, with one entry per line. Can process it
    :param in_path:
    :param process_function:
    :return:
    """
    with open(in_path, 'r') as in_file:
        lines = [line.strip() for line in in_file.readlines()]
        return [process_function(line) for line in lines]


def filter_on_freq(count_dict, threshold=None):
    """
    Returns post IDs with comments more th
    :param count_dict:
    :param threshold:
    :return:
    """
    if not threshold:
        return count_dict
    return [post_id for post_id, count in tqdm.tqdm(count_dict.items(), "Filter on frequency") if count >= threshold]


def filter_content(post_list, filter_ids):
    """
    Return a reduced list containing only posts with that id
    :param post_list:
    :param filter_ids:
    :return:
    """
    return [post for post in tqdm.tqdm(post_list, "Filtering content") if post['id'] in filter_ids]


def filter_on_keys(content_dict, include_set):
    """
    Filters a dict on value of keys
    :param content_dict:
    :param include_set:
    :return:
    """
    return {k: v for k, v in content_dict.items() if k in include_set}


def add_comments_to_submissions(submissions, comments):
    """
    Adds related comments to submissions
    :param submissions:
    :param comments:
    :return:
    """
    submissions_dict = {}
    for sub_post in submissions:
        submissions_dict[sub_post['id']] = sub_post
        sub_post['comments'] = []
    for com_post in tqdm.tqdm(comments, desc="Traversing comments"):
        parent_id = adjust_id(com_post['parent_id'])
        if parent_id in submissions_dict:
            submissions_dict[parent_id]['comments'].append(com_post)


def group_liwc_separate_comments(liwc_values_dict):
    """
    Group and average LIWC comments
    :param liwc_values_dict:
    :return:
    """
    submissions_dict = {}
    for value_id in liwc_values_dict.keys():
        sub_id, com_id = value_id.split('_')
        if value_id not in submissions_dict:
            submissions_dict[sub_id] = {}
        submissions_dict[sub_id][com_id] = liwc_values_dict[value_id]
    for sub_id in submissions_dict:
        submissions_dict[sub_id] = average_liwc(submissions_dict[sub_id])
    return submissions_dict


def group_comments_per_submission(submissions, out_dir=None):
    """
    Group all comments for each post
    :param submissions:
    :return:
    """
    comments = {}
    for sub_post in tqdm.tqdm(submissions, desc='Grouping and writing comments'):
        all_text = ''
        for com_post in sub_post['comments']:
            all_text += com_post['body']
            all_text += '\n'
        comments[sub_post['id']] = all_text
        if out_dir:
            write_to_file('{}{}.txt'.format(out_dir, sub_post['id']), all_text)
    return comments


def separate_comments_per_submission(submissions, out_dir=None):
    """
    Separate comments for each submission
    :param submission:
    :param out_dir:
    :return:
    """
    comments = {}
    for sub_post in tqdm.tqdm(submissions, desc='Separating and writing comments'):
        sub_comments = []
        for com_post in sub_post['comments']:
            sub_comments.append(com_post['body'])
            if out_dir:
                write_to_file('{}{}_{}.txt'.format(out_dir, sub_post['id'], com_post['id']), com_post['body'])
        comments[sub_post['id']] = sub_comments
    return comments


def group_per_parent(posts):
    """
    Group comments based on parents
    :param posts:
    :return:
    """
    parent_to_comments = {}
    for p in tqdm.tqdm(posts, desc='Comments'):
        if p['parent_id'] not in parent_to_comments:
            parent_to_comments[p['parent_id']] = []
        parent_to_comments[p['parent_id']].append(p)
    return parent_to_comments


def binarize(reference_list, values_list):
    """
    Change values from float to 0 vs 1 based on the reference
    :param reference_list:
    :param values_list:
    :return:
    """
    return [(0 if val < ref else 1) for (ref, val) in zip(reference_list, values_list)]


def add_labels(labels_list, values_list):
    """
    Converts a list to a dictionary of label-value pairs
    :param labels_list:
    :param values_list:
    :return:
    """
    return {label: value for label, value in zip(labels_list, values_list)}


def remove_posts(post_list, criteria):
    """
    Removes posts from the list based on a criteria
    :param post_list:
    :param criteria:
    :return:
    """
    return [p for p in post_list if not criteria(p)]


def reduce_post(post, liwc_include_set):
    # selftext, title, subreddit, LIWC(specific list) / body
    reduced_post = {
        'title': post['title'],
        'selftext': post['selftext'],
        'liwc' : {
            'submission': {
                'values': filter_on_keys(post['liwc']['submission']['values'], liwc_include_set),
                'classes': filter_on_keys(post['liwc']['submission']['classes'], liwc_include_set)
            },
            'comments': {
                'grouped': {
                    'values': filter_on_keys(post['liwc']['comments']['grouped']['values'], liwc_include_set),
                    'classes': filter_on_keys(post['liwc']['comments']['grouped']['classes'], liwc_include_set)
                },
                'separate': {
                    'values': filter_on_keys(post['liwc']['comments']['separate']['values'], liwc_include_set),
                    'classes': filter_on_keys(post['liwc']['comments']['separate']['classes'], liwc_include_set)
                }
            }
        },
        'comments': []
    }
    for comment_post in post['comments']:
        reduced_post['comments'].append(comment_post['body'])
    return reduced_post


def reduce_post_list(post_list, subreddit_set, liwc_set):
    """
    Reduce post list by including posts from specific subreddits, and reduce the content of posts
    :param post_list:
    :param subreddit_set:
    :param liwc_set:
    :return:
    """
    return [reduce_post(post, liwc_set) for post in post_list if post['subreddit'] in subreddit_set]


def save_analysis(subreddit_name, liwc_set):
    """
    Saves reduced form of the full json form
    :param subreddit_name:
    :param liwc_set:
    :return:
    """
    posts = load_posts_from_json(MENTAL_WITH_FULL_INFO_PATH)
    subreddits_set = set([subreddit_name])
    reduced_posts = reduce_post_list(posts, subreddits_set, liwc_set)
    save_json(reduced_posts, ROOT + 'RS_C_201908_{}.r.json'.format(subreddit_name))


def get_liwc_matrix(submission_posts, liwc_include_set, values_extract_fn):
    """
    Extract two matrices for liwc values for both submissions and comments
    :param submission_posts:
    :param liwc_include_set:
    :param values_extract_fn:
    :return:
    """
    if liwc_include_set:
        liwc_dict = {cat: [] for cat in liwc_include_set}
    else:
        liwc_dict = {cat: [] for cat, _ in values_extract_fn(submission_posts[0]).items()}

    for post in submission_posts:
        [liwc_dict[cat].append(value)
         for cat, value in values_extract_fn(post).items()
         if not liwc_include_set or cat in liwc_include_set]

    return liwc_dict


def printed_key(label1, label2):
    """
    Return a unique key regardless of order of labels
    :param label1:
    :param label2:
    :return:
    """
    return '{}'.format('-'.join(sorted([label1, label2])))


def plot_correlation_pdf(pdf_path, var1_dict, var2_dict, min_to_print=0, all_vs_all=True, remove_zeros=False):
    """
    Write to pdf files plots for correlation between lists or matrices
    :param pdf_path:
    :param var1_dict:
    :param var2_dict:
    :param min_to_print: minimum correlation to print in pdf
    :param all_vs_all: try all permutations
    :param remove_zeros: removes zero values if the happen at the same index in both arrays
    :return:
    """
    with PdfPages(pdf_path) as pdf_file:
        printed_set = set()
        for var1_label in tqdm.tqdm(var1_dict.keys(), desc="Plotting"):
            for var2_label in var2_dict.keys():
                itr_label = printed_key(var1_label, var2_label)
                if all_vs_all and itr_label not in printed_set or var1_label == var2_label:
                    if remove_zeros:
                        var1_list, var2_list = filter_zeros(var1_dict[var1_label], var2_dict[var2_label])
                    else:
                        var1_list, var2_list = var1_dict[var1_label], var2_dict[var2_label]
                    corr = spearmanr(var1_list, var2_list).correlation
                    if math.fabs(corr) < min_to_print:
                        continue
                    plt.figure()
                    plt.scatter(var1_dict[var1_label], var2_dict[var2_label])
                    plt.title('{} vs {} ({:.2f})'.format(var1_label, var2_label, corr))
                    pdf_file.savefig()
                    plt.close()
                    printed_set.add(itr_label)


def get_correlation(var1_dict, var2_dict, min_to_print=0, all_vs_all=True, remove_zeros=False):
    """
    Computes correlation between categories
    :param var1_dict:
    :param var2_dict:
    :param min_to_print:
    :param all_vs_all:
    :param remove_zeros:
    :return:
    """
    printed_dict = {}
    for var1_label in tqdm.tqdm(var1_dict.keys(), desc="Computing correlation"):
        for var2_label in var2_dict.keys():
            itr_label = printed_key(var1_label, var2_label)
            if all_vs_all and itr_label not in printed_dict.keys() or var1_label == var2_label:
                if remove_zeros:
                    var1_list, var2_list = filter_zeros(var1_dict[var1_label], var2_dict[var2_label])
                else:
                    var1_list, var2_list = var1_dict[var1_label], var2_dict[var2_label]
                corr = spearmanr(var1_list, var2_list).correlation
                if math.fabs(corr) < min_to_print:
                    continue
                printed_dict[itr_label] = corr
    return printed_dict


def write_correlation_comparison_csv(var1_corr_dict, var2_corr_dict, label1, label2, out_csv_path):
    """
    Writes correlation comparison between two variables to a csv. Both variables are expected to have the same keys.
    :param var1_corr_dict:
    :param var2_corr_dict:
    :param out_csv_path:
    :return:
    """
    with open(out_csv_path, 'w') as out_csv_file:
        print('label,same,{},{}'.format(label1, label2), file=out_csv_file)
        for label in var1_corr_dict.keys():
            corr1 = var1_corr_dict[label]
            corr2 = var2_corr_dict[label]
            is_same_category = "Yes" if re.match('(\\w+)-\\1', label) else "No"
            print('{},{},{},{}'.format(label, is_same_category, corr1, corr2), file=out_csv_file)


def filter_zeros(var1_list, var2_list):
    """
    Filter zero values that happen at the same index
    :param var1_list:
    :param var2_list:
    :return:
    """
    var1_new_list = []
    var2_new_list = []
    for v1, v2 in zip(var1_list, var2_list):
        if not (v1 == 0 and v2 == 0):
            var1_new_list.append(v1)
            var2_new_list.append(v2)
    return var1_new_list, var2_new_list


# Main executions #

def minimize_counts_for_mental():
    """
    Creates a smaller list of counts (for optimized loading)
    :return:
    """
    comment_counts = load_comment_counts(COMMENT_COUNT_PATH, MENTAL_SUBMISSION_IDS_PATH)
    save_comment_counts(comment_counts, MENTAL_SUBMISSION_COUNTS_PATH)


def prepare_for_liwc_control_set():
    """
    Writes from the control set to LIWC
    :return:
    """
    post_list = load_posts(CONTROL_COMMENTS_PATH)
    write_each_post_in_file(post_list, CONTROL_COMMENTS_TEXT_DIR_PATH, lambda post: post['body'])


def prepare_for_liwc_mental_submissions():
    """
    Writes from the mental set to LIWC
    :return:
    """
    post_list = load_posts(MENTAL_SUBMISSIONS_WITH_COMMENTS_PATH)
    write_each_post_in_file(post_list, MENTAL_SUBMISSIONS_TEXT_DIR_PATH,
                            lambda post: '{}\n{}'.format(post['title'], post['selftext']))


def prepare_for_liwc_control_submissions():
    """
    Writes from the mental set to LIWC
    :return:
    """
    post_list = load_posts(CONTROL_SUBMISSIONS_PATH)
    write_each_post_in_file(post_list, CONTROL_SUBMISSIONS_TEXT_DIR_PATH,
                            lambda post: '{}\n{}'.format(post['title'], post['selftext']))


def compute_average_for_control_set():
    """
    Computes the average values of LIWC for the control set
    :return:
    """
    control_liwc_dict = load_liwc(CONTROL_SEPARATE_PER_SUBMISSION_LIWC_PATH)
    average_control_liwc = average_liwc(control_liwc_dict)
    save_list(average_control_liwc, CONTROL_COMMENTS_AVERAGE_SEPARATE_LIWC_PATH)


def compute_average_for_grouped_control_set():
    """
    Computes the average values of LIWC for the control set
    :return:
    """
    control_liwc_dict = load_liwc(CONTROL_GROUPED_PER_SUBMISSION_LIWC_PATH)
    average_control_liwc = average_liwc(control_liwc_dict)
    save_list(average_control_liwc, CONTROL_COMMENTS_AVERAGE_GROUPED_LIWC_PATH)


def compute_average_for_submissions_control_set():
    """
    Computes the average values of LIWC for the control set
    :return:
    """
    control_liwc_dict = load_liwc(CONTROL_SUBMISSIONS_LIWC_PATH)
    average_control_liwc = average_liwc(control_liwc_dict)
    save_list(average_control_liwc, CONTROL_SUBMISSIONS_AVERAGE_LIWC_PATH)


def assign_comments_to_submissions_mental():
    """
    Add to submissions objects all related comments
    :return:
    """
    all_mental_submissions = load_posts(MENTAL_SUBMISSIONS_PATH)
    all_mental_comments = load_posts(MENTAL_COMMENTS_PATH)
    comment_counts = load_comment_counts(MENTAL_SUBMISSION_COUNTS_PATH)
    submission_ids_with_lots_of_comments = filter_on_freq(comment_counts, 50)
    submissions = filter_content(all_mental_submissions, submission_ids_with_lots_of_comments)
    add_comments_to_submissions(submissions, all_mental_comments)
    with open(MENTAL_SUBMISSIONS_WITH_COMMENTS_PATH, 'w') as submissions_file:
        json.dump(submissions, submissions_file)


def assign_comments_to_submissions_control():
    """
    Add to submissions objects all related comments
    :return:
    """
    all_mental_submissions = load_posts(CONTROL_SUBMISSIONS_PATH)
    all_mental_comments = load_posts(CONTROL_COMMENTS_PATH)
    comment_counts = load_comment_counts(CONTROL_SUBMISSION_COUNTS_PATH)
    submission_ids_with_lots_of_comments = filter_on_freq(comment_counts, 50)
    submissions = filter_content(all_mental_submissions, submission_ids_with_lots_of_comments)
    add_comments_to_submissions(submissions, all_mental_comments)
    with open(CONTROL_SUBMISSIONS_WITH_COMMENTS_PATH, 'w') as submissions_file:
        json.dump(submissions, submissions_file)


def liwc_for_all_comments_per_submissions():
    """
    Write for LIWC consumption, all comments per submission into a single file
    :return:
    """
    submissions = load_posts(MENTAL_SUBMISSIONS_WITH_COMMENTS_PATH)
    group_comments_per_submission(submissions=submissions, out_dir=MENTAL_COMMENTS_ALLONE_DIR_PATH)


def liwc_for_separate_comments_per_submission():
    """
    Write for LIWC consumption, each comment in a separate file
    :return:
    """
    submissions = load_posts(MENTAL_SUBMISSIONS_WITH_COMMENTS_PATH)
    separate_comments_per_submission(submissions, MENTAL_COMMENTS_SEPARATE_DIR_PATH)


def group_control_set_comments_per_parent():
    """
    Group comments based on their parents and then prep them for LIWC
    :return:
    """
    comments = load_posts(CONTROL_COMMENTS_PATH)
    parent_to_comments_dict = group_per_parent(comments)
    for parent_id in tqdm.tqdm(parent_to_comments_dict.keys(), desc='Comment groups'):
        all_post_text = ''
        for com_post in parent_to_comments_dict[parent_id]:
            all_post_text += com_post['body']
            all_post_text += '\n'
        write_to_file('{}{}.txt'.format(CONTROL_COMMENTS_GROUPED_TEXT_DIR_PATH, adjust_id(parent_id)), all_post_text)


def build_full_info_file_mental():
    """
    Builds json file with all info included
    :return:
    """
    submissions = load_posts(MENTAL_SUBMISSIONS_WITH_COMMENTS_PATH)

    labels_list = load_list(LIWC_LABELS_PATH)

    submission_avg_liwc = load_list(CONTROL_SUBMISSIONS_AVERAGE_LIWC_PATH, lambda x: float(x))
    comments_grouped_avg_liwc = load_list(CONTROL_COMMENTS_AVERAGE_GROUPED_LIWC_PATH, lambda x: float(x))
    comments_separate_avg_liwc = load_list(CONTROL_COMMENTS_AVERAGE_SEPARATE_LIWC_PATH, lambda x: float(x))

    submissions_liwc_dict = load_liwc(MENTAL_SUBMISSIONS_LIWC_PATH)
    comments_separate_liwc_dict = group_liwc_separate_comments(load_liwc(MENTAL_SEPARATE_PER_SUBMISSION_LIWC_PATH))
    comments_grouped_liwc_dict = load_liwc(MENTAL_GROUPED_PER_SUBMISSION_LIWC_PATH)

    for post in tqdm.tqdm(submissions, 'Submissions'):
        post['liwc'] = {
            'submission': {
                'values': add_labels(labels_list, submissions_liwc_dict[post['id']]),
                'classes': add_labels(labels_list, binarize(submission_avg_liwc, submissions_liwc_dict[post['id']]))
            },
            'comments': {
                'grouped': {
                    'values': add_labels(labels_list, comments_grouped_liwc_dict[post['id']]),
                    'classes': add_labels(labels_list, binarize(comments_grouped_avg_liwc,
                                                                comments_grouped_liwc_dict[post['id']]))
                },
                'separate': {
                    'values': add_labels(labels_list, comments_separate_liwc_dict[post['id']]),
                    'classes': add_labels(labels_list, binarize(comments_separate_avg_liwc,
                                                                comments_separate_liwc_dict[post['id']]))
                }
            }
        }

    save_json(submissions, MENTAL_WITH_FULL_INFO_PATH)


def build_full_info_file_control():
    """
    Builds json file with all info included
    :return:
    """
    submissions = load_posts(CONTROL_SUBMISSIONS_WITH_COMMENTS_PATH)

    labels_list = load_list(LIWC_LABELS_PATH)

    submissions_liwc_dict = load_liwc(CONTROL_SUBMISSIONS_LIWC_PATH)
    comments_grouped_liwc_dict = load_liwc(CONTROL_GROUPED_PER_SUBMISSION_LIWC_PATH)

    for post in tqdm.tqdm(submissions, 'Submissions'):
        post['liwc'] = {
            'submission': {
                'values': add_labels(labels_list, submissions_liwc_dict[post['id']])
            },
            'comments': {
                'grouped': {
                    'values': add_labels(labels_list, comments_grouped_liwc_dict[post['id']])
                }
            }
        }

    save_json(submissions, CONTROL_WITH_FULL_INFO_PATH)


def reduce_info():
    liwc_set = set(['i', 'you', 'shehe', 'they', 'affect', 'posemo', 'negemo', 'family', 'friend', 'feel', 'body',
                    'achieve', 'power', 'work', 'home', 'money', 'death', 'swear'])
    save_analysis('depression', liwc_set)
    save_analysis('SuicideWatch', liwc_set)
    save_analysis('socialanxiety', liwc_set)


def correlation_mental():
    """
    Creates scatter plot for correlation between different LIWC categories for mental health subreddits
    :return:
    """
    liwc_set = set(['i', 'you', 'shehe', 'they', 'affect', 'posemo', 'negemo', 'family', 'friend', 'feel', 'body',
                    'achieve', 'power', 'work', 'home', 'money', 'death', 'swear'])
    liwc_set = None

    submission_posts = load_posts_from_json(MENTAL_WITH_FULL_INFO_PATH)
    submissions_liwc_dict = get_liwc_matrix(submission_posts, liwc_set, lambda p: p['liwc']['submission']['values'])
    comments_liwc_dict = get_liwc_matrix(submission_posts, liwc_set, lambda p: p['liwc']['comments']['grouped']['values'])

    plot_correlation_pdf(ROOT + 'RS_201908.liwc_correlation_no-zeros_all_min0.3.pdf',
                         submissions_liwc_dict,
                         comments_liwc_dict,
                         min_to_print=0.3)


def correlation_control():
    """
    Creates scatter plot for correlation between different LIWC categories for control subreddit
    :return:
    """
    liwc_set = set(['i', 'you', 'shehe', 'they', 'affect', 'posemo', 'negemo', 'family', 'friend', 'feel', 'body',
                    'achieve', 'power', 'work', 'home', 'money', 'death', 'swear'])

    submission_posts = load_posts_from_json(CONTROL_WITH_FULL_INFO_PATH)
    submissions_liwc_dict = get_liwc_matrix(submission_posts, liwc_set, lambda p: p['liwc']['submission']['values'])
    comments_liwc_dict = get_liwc_matrix(submission_posts, liwc_set, lambda p: p['liwc']['comments']['grouped']['values'])
    plot_correlation_pdf(ROOT + 'RS_201908.IAmA_liwc_correlation.pdf',
                         submissions_liwc_dict,
                         comments_liwc_dict,
                         all_vs_all=False)


def compare_correlation():
    liwc_set = None

    # Mental case
    mental_submission_posts = load_posts_from_json(MENTAL_WITH_FULL_INFO_PATH)
    mental_submissions_liwc_dict = get_liwc_matrix(mental_submission_posts, liwc_set, lambda p: p['liwc']['submission']['values'])
    mental_comments_liwc_dict = get_liwc_matrix(mental_submission_posts, liwc_set, lambda p: p['liwc']['comments']['grouped']['values'])
    mental_correlation_dict = get_correlation(mental_submissions_liwc_dict, mental_comments_liwc_dict)

    # Control set case
    control_submission_posts = load_posts_from_json(CONTROL_WITH_FULL_INFO_PATH)
    control_submissions_liwc_dict = get_liwc_matrix(control_submission_posts, liwc_set, lambda p: p['liwc']['submission']['values'])
    control_comments_liwc_dict = get_liwc_matrix(control_submission_posts, liwc_set, lambda p: p['liwc']['comments']['grouped']['values'])
    control_correlation_dict = get_correlation(control_submissions_liwc_dict, control_comments_liwc_dict)

    #write_correlation_comparison_csv(mental_correlation_dict, control_correlation_dict, 'Mental', 'Control', ROOT +
    #                                 'RS_201908.mental_IAmA_correlation_compare.csv')

    mental_correlation_list = []
    control_correlation_list = []
    for label in mental_correlation_dict:
        if not math.isnan(mental_correlation_dict[label]) and not math.isnan(control_correlation_dict[label]):
            mental_correlation_list.append(mental_correlation_dict[label])
            control_correlation_list.append(control_correlation_dict[label])
    corr = spearmanr(mental_correlation_list, control_correlation_list).correlation

    mental_correlation_same_cat_list = []
    control_correlation_same_cat_list = []
    for label in mental_correlation_dict:
        if re.match('(\\w+)-\\1', label) and not math.isnan(mental_correlation_dict[label]) \
                and not math.isnan(control_correlation_dict[label]):
            mental_correlation_same_cat_list.append(mental_correlation_dict[label])
            control_correlation_same_cat_list.append(control_correlation_dict[label])
    corr_same_cat = spearmanr(mental_correlation_same_cat_list, control_correlation_same_cat_list).correlation

    print('overall correlation: {}'.format(corr))
    print('Same category correlation: {}'.format(corr_same_cat))


def get_random_correlation():
    liwc_set = set(['i', 'you', 'shehe', 'they', 'affect', 'posemo', 'negemo', 'family', 'friend', 'feel', 'body',
                    'achieve', 'power', 'work', 'home', 'money', 'death', 'swear'])

    mental_submission_posts = load_posts_from_json(MENTAL_WITH_FULL_INFO_PATH)
    mental_submissions_liwc_dict = get_liwc_matrix(mental_submission_posts, liwc_set, lambda p: p['liwc']['submission']['values'])
    mental_comments_liwc_dict = get_liwc_matrix(mental_submission_posts, liwc_set, lambda p: p['liwc']['comments']['grouped']['values'])
    var1_dict = mental_submissions_liwc_dict
    var2_dict = mental_comments_liwc_dict

    all_vs_all = False
    remove_zeros = True
    min_to_print = -0.0
    shuffle = True
    random.seed(0)

    printed_dict = {}
    for var1_label in tqdm.tqdm(var1_dict.keys(), desc="Computing correlation"):
        for var2_label in var2_dict.keys():
            itr_label = printed_key(var1_label, var2_label)
            if all_vs_all and itr_label not in printed_dict.keys() or var1_label == var2_label:
                if remove_zeros:
                    var1_list, var2_list = filter_zeros(var1_dict[var1_label], var2_dict[var2_label])
                else:
                    var1_list, var2_list = var1_dict[var1_label], var2_dict[var2_label]
                if shuffle:
                    random.shuffle(var1_list)
                    random.shuffle(var2_list)
                corr = spearmanr(var1_list, var2_list).correlation
                if math.fabs(corr) < min_to_print:
                    continue
                printed_dict[itr_label] = corr

    values = [printed_dict[key] for key in printed_dict.keys()]
    print('Average correlation {}'.format(sum(values) / len(values)))


if __name__ == '__main__':
    get_random_correlation()
