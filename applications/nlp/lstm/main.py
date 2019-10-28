import os.path
import sys

current_file = os.path.realpath(__file__)
root_dir = os.path.dirname(os.path.dirname(current_file))
sys.path.append(root_dir)
import utils.gutenberg

num_tokens = 1000
#data_url = 'http://www.gutenberg.org/files/100/100-0.txt' # Shakespeare
data_url = 'https://www.gutenberg.org/files/84/84-0.txt' # Frankenstein
data_dir = os.path.join(root_dir, 'data', 'frankenstein')
dataset = utils.gutenberg.GutenbergDataset(data_dir, num_tokens, data_url)
