import os.path
import sys

# Local imports
current_file = os.path.realpath(__file__)
root_dir = os.path.dirname(os.path.dirname(current_file))
sys.path.append(root_dir)
import utils.gutenberg

# Options
text_name = 'frankenstein'
sequence_length = 10

# Download and tokenize text data, if needed
data_url = utils.gutenberg.get_url(text_name)
data_dir = os.path.join(root_dir, 'data', text_name)
corpus = utils.gutenberg.GutenbergCorpus(data_dir, data_url)

# Sample access functions
def get_sample(index):
    return corpus[index:index+sequence_length]
def num_samples():
    return len(corpus) - sequence_length + 1
def sample_dims():
    return (sequence_length,)
