import tensorflow as tf
from tensorboard.plugins import projector
import fasttext
import numpy as np
import csv
import os


def tf_visualize(tb_path, voc_file_name, vectors, tensor_label='embeddings'):
    """
    Creates tensor model to be viewed in Tensor board
    :param tb_path:
    :param voc_file_name:
    :param vectors:
    :param tensor_label:
    :return:
    """
    sess = tf.InteractiveSession()
    tf.Variable(vectors, trainable=False, name=tensor_label)
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(tb_path, sess.graph)

    projector_config = projector.ProjectorConfig()
    embed = projector_config.embeddings.add()
    embed.tensor_name = tensor_label
    embed.metadata_path = voc_file_name

    projector.visualize_embeddings(writer, projector_config)
    saver.save(sess, tb_path + 'model.ckpt')


def write_voc(root_path, voc_file_name, words):
    """
    Write vocabulary file
    :param root_path:
    :param voc_file_name:
    :param words:
    :return:
    """
    if not root_path.endswith('/'):
        root_path += '/'
    with open(root_path + voc_file_name, 'w') as voc_file:
        for w in words:
            print(w, file=voc_file)


def load_embeddings_from_model(model_path):
    """
    Loads words and embeddings from fast text model
    :param model_path:
    :return:
    """
    model = fasttext.load_model(model_path)
    return model.voc(), np.array([model.get_word_vector(word) for word in model.voc()])


def embeddings_str_to_nums(arr_string):
    """
    Converts float array written in text to numpy array
    :param arr_string:
    :return:
    """
    return np.array([float(num) for num in arr_string.split()])


def load_from_file(path, subreddit_label=None):
    """
    Loads words and embeddings text-like file
    :param path:
    :return:
    """
    words = []
    embeddings = []
    with open(path, 'r') as in_file:
        embeddings_csv = csv.reader(in_file)
        for row in embeddings_csv:
            subreddit = row[0]
            word = row[1]
            embeddings_str = row[2]
            if not subreddit_label or subreddit == subreddit_label:
                words.append(word)
                embeddings.append(embeddings_str_to_nums(embeddings_str))
    return words, np.array(embeddings)


def tb_subreddit_path(tb_path, subreddit):
    new_path = tb_path + subreddit
    os.mkdir(new_path)
    return new_path


if __name__ == '__main__':
    subreddit = 'BPD'
    embeddings_path = '/Users/mbahgat/phd/datasets/reddit/category_centroids.vec2_cosine'

    tb_path = '/Users/mbahgat/phd/ws/reddit_analysis/tb/'
    voc_file = '{}.voc'.format(subreddit)

    tb_path = tb_subreddit_path(tb_path, subreddit)

    words, vectors = load_from_file(embeddings_path, subreddit)
    write_voc(tb_path, voc_file, words)
    tf_visualize(tb_path, voc_file, vectors, tensor_label=subreddit)
