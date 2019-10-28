import os.path
import sys

current_file = os.path.realpath(__file__)
root_dir = os.path.dirname(os.path.dirname(current_file))
sys.path.append(root_dir)
import utils.gutenberg

data_url = 'https://www.gutenberg.org/files/84/84-0.txt' # Frankenstein
# data_url = 'http://www.gutenberg.org/files/100/100-0.txt' # Shakespeare
data_dir = os.path.join(root_dir, 'data', 'frankenstein')
dataset = utils.gutenberg.GutenbergDataset(data_dir, data_url)
