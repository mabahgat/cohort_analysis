import csv
import re

in_csv_path = '/Users/mbahgat/phd/datasets/reddit/all_subreddits_distances_3cat.csv'
out_csv_path = '/Users/mbahgat/phd/datasets/reddit/all_subreddits_distances_3cat_values-added.csv'

if __name__ == '__main__':
    with open(in_csv_path, 'r') as in_file, open(out_csv_path, 'w') as out_file:
        in_csv = csv.reader(in_file)
        for row in in_csv:
            subreddit = row[0]
            cat1 = row[1]
            cat2 = row[2]
            distance = row[3]
            if re.match("[A-Z\\-]", cat1):
                cat1 = 'values_' + cat1
            if re.match("[A-Z\\-]", cat2):
                cat2 = 'values_' + cat2
            print("{},{},{},{}".format(subreddit, cat1, cat2, distance), file=out_file)
