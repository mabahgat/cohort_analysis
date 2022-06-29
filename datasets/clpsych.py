import json
import argparse
import re
import sys
import time
import requests
import tqdm

POST_IDS_ROOT = '/Users/mbahgat/phd/datasets/umd_reddit_suicidewatch_dataset_v2/crowd/comments/'
TRAIN_POST_IDS_PATH = POST_IDS_ROOT + 'train_posts_ids.lst'
TEST_POST_IDS_PATH = POST_IDS_ROOT + 'test_posts_ids.lst'


def load_list(in_path, process_function=lambda x: x):
    """
    Reads a list from file, with one entry per line. Can process it
    :param in_path:
    :param process_function:
    :return:
    """
    with open(in_path, 'r') as in_file:
        return [process_function(line.strip()) for line in in_file.readlines()]


def load_list(path):
    """
    Loads a list from a file
    :param path:
    :return:
    """
    with open(path, 'r') as list_file:
        return [line.strip() for line in list_file.readlines()]


def parse_via_json(json_string, field_string):
    """
    Retrieves a specific field by parsing the record into JSON first
    :param json_string:
    :param field_string:
    :return:
    """
    return json.loads(json_string)[field_string]


def parse_via_regex(json_string, field_string):
    """
    Retrieves a specific field by regex matching
    :param json_string:
    :param field_string:
    :return:
    """
    return re.search('"{}":"([^"]+)"'.format(field_string), json_string)[1]


def extract_from_dump(posts_file_path, ids_set, field_extract_fn, id_parse_fn):
    """
    Extract comments related to passed IDs
    :param posts_file_path:
    :param ids_set:
    :param field_extract_fn:
    :param id_parse_fn:
    :return:
    """
    with open(posts_file_path, 'r', encoding='utf8') as posts_file:
        relevant_comments_list = []
        for line in posts_file.readlines():
            if id_parse_fn(field_extract_fn(line, 'parent_id')) in ids_set:
                relevant_comments_list.append(line)
        return relevant_comments_list


def get_comment_api(comment_id):
    """
    Retrieves comment object from push shift api
    :param comment_id:
    :return:
    """
    response = requests.get("https://api.pushshift.io/reddit/submission/", {'id', comment_id})
    if response.status_code != 200:
        print('bbad response {} for comment id {}'.format(response.status_code, comment_id))
    return response.json()


def extract_from_api(ids_set, subreddit_name):
    """
    Retrieves comments from Reddit API # https://api.pushshift.io/reddit/submission/comment_ids/{}
    :param ids_set:
    :return:
    """
    relevant_comments_dict = {}
    for post_id in tqdm.tqdm(ids_set, desc='Retrieving Comments'):
        #
        response = requests.get("http://www.reddit.com/r/{}/comments/{}.json".format(subreddit_name, post_id),
                                headers={'User-agent': 'bot 0.1'})
        if response.status_code != 200:
            print('bad response {} for {}'.format(response.status_code, post_id), file=sys.stderr)
            continue
        response_data = response.json()
        relevant_comments_dict[post_id] = [child_obj['data'] for child_obj in response_data[1]['data']['children']]
        time.sleep(0.5)
    return relevant_comments_dict


# Main-s #

def extract_comments_from_corpus():
    parser = argparse.ArgumentParser(description='Extract comments from Reddit corpus jsonl files')
    parser.add_argument('-l', '--list', help='list of ids to extract', required=True)
    parser.add_argument('--api', help='use pushshift api to retrieve data', action='store_true')
    parser.add_argument('-c', '--corpus', help='corpus file to look into')
    parser.add_argument('-o', '--output', help='output file')
    args = parser.parse_args()

    post_ids = load_list(args.list)

    # Retrieve comments
    if args.api:
        relevant_comments = extract_from_api(post_ids, 'SuicideWatch')
    else:
        relevant_comments = extract_from_dump(args.corpus, post_ids, parse_via_json, lambda id: id.split('_')[1])

    # Write output
    if args.output:
        out_file = open(args.output, 'w', encoding='utf8')
    else:
        out_file = sys.stdout
    json.dump(relevant_comments, out_file)


if __name__ == '__main__':
    start_time = time.time()
    extract_comments_from_corpus()
    print("--- %s seconds ---" % (time.time() - start_time))