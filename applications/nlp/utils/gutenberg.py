import collections
import os
import os.path
import re
import urllib.request

import numpy as np
import spacy
import spacy.lang.en

def strip_boilerplate(raw_file, stripped_file):
    """Remove header and footer from Project Gutenberg text file

    See:

    https://www.gutenberg.org/wiki/Gutenberg:Project_Gutenberg_Header_How-To

    Args:
        raw_file (str): Text file downloaded from Project Gutenberg
        stripped_file (str): Path where the stripped file will be saved

    """
    with open(raw_file, 'r') as in_file, \
         open(stripped_file, 'w') as out_file:
        started = False
        begin_regex = re.compile('^\*\*\* START OF THIS PROJECT GUTENBERG EBOOK .* \*\*\*$')
        end_regex = re.compile('^\*\*\* END OF THIS PROJECT GUTENBERG EBOOK .* \*\*\*$')
        for line in in_file:
            if started:
                if end_regex.match(line):
                    break
                else:
                    out_file.write(line)
            elif begin_regex.match(line):
                started = True


def tokenize(vocab_size,
             text_data_file,
             token_data_file,
             remove_whitespace=True):
    """Convert text file to sequence of token indices

    Tokenization is performed with SpaCy and tokens are ordered based
    on how often they occur in the text. If a token is not one of the
    `vocab_size`-1 most common, then it is assigned index
    `vocab_size`-1.

    The token data is saved in an .npz file with two arrays: `data` is
    a sequence of token indices and `vocab` is a list of token
    strings.

    Args:
        vocab_size (int): Number of tokens in vocabulary
        text_data_file (str): Text file
        token_data_file (str): Path where the tokenized data will be
            saved as an .npz file
        remove_whitespace (bool): Whether to discard tokens that are
            purely made of whitespace

    Returns:
        numpy.array of int: Sequence of token indices
        list of str: Tokens

    """

    # Tokenize text file with SpaCy
    nlp = spacy.lang.en.English()
    text_data = []
    token_counts = collections.Counter()
    with open(text_data_file) as f:
        for line in f:
            for _token in nlp(line):
                token = str(_token)
                if remove_whitespace and token.isspace():
                    continue
                text_data.append(token.lower())
                token_counts[token] += 1

    # Find the most common tokens
    token_strings = []
    token_indices = {}
    for index, (token, _) in enumerate(token_counts.most_common(vocab_size-1)):
        token_strings.append(token)
        token_indices[token] = index
    token_strings.append('<UNKNOWN>')

    # Convert text to token indices
    token_data = np.zeros(len(text_data), dtype=int)
    for i, token in enumerate(text_data):
        token_data[i] = token_indices.get(token, vocab_size-1)

    # Save to file and return tokenized data
    np.savez_compressed(token_data_file, data=token_data, vocab=token_strings)
    return token_data, token_strings


class GutenbergDataset():
    """Tokenized text dataset from Project Gutenberg

    Args:
        data_dir (str): Directory for downloading data and
            intermediate
        vocab_size (int): Number of tokens in vocabulary
        data_url (str): URL to Project Gutenberg text file

    """
    def __init__(self,
                 data_dir,
                 vocab_size,
                 data_url):

        # Create data directory if needed
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)
        data_dir = os.path.realpath(data_dir)

        # Load tokenized data
        # Note: If needed, download the text data from Project
        # Gutenberg and tokenize it.
        token_data_file = os.path.join(data_dir, 'token_data.npz')
        if os.path.isfile(token_data_file):
            data = np.load(token_data_file)
            token_data = data['data']
            vocab = data['vocab']
        else:
            text_data_file = os.path.join(data_dir, 'text_data.txt')
            if not os.path.isfile(text_data_file):
                raw_file = os.path.join(data_dir, 'raw.txt')
                if not os.path.isfile(raw_file):
                    urllib.request.urlretrieve(data_url,
                                               filename=raw_file)
                strip_boilerplate(raw_file, text_data_file)
            token_data, vocab = tokenize(vocab_size,
                                         text_data_file,
                                         token_data_file)

        # Class members
        self.token_data = token_data
        self.vocab = vocab
