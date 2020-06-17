"""Helper functions for text data from Project Gutenberg."""
import array
import os
import os.path
import re
import urllib.request
import numpy as np


def get_url(name):
    """URL to Project Gutenberg text file."""
    urls = {
        'frankenstein': 'https://www.gutenberg.org/files/84/84-0.txt',
        'shakespeare': 'https://www.gutenberg.org/files/100/100-0.txt',
    }
    return urls[name.lower()]


def strip_boilerplate(raw_file, stripped_file):
    """Remove header and footer from Project Gutenberg text file.

    See:

    https://www.gutenberg.org/wiki/Gutenberg:Project_Gutenberg_Header_How-To

    Args:
        raw_file (str): Text file downloaded from Project Gutenberg.
        stripped_file (str): Path where the stripped file will be
            saved.

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


def tokenize(text_file,
             encoded_file=None,
             vocab_file=None,
             ignore_whitespace=True):
    """Convert text file to sequence of token IDs.

    Tokenization is performed with BERT tokenizer.

    Args:
        text_file (str): Text file to be encoded.
        encoded_file (str, optional): If provided, path where the
           encoded data will be saved as an .npz file. The sequence of
           token IDs is saved as 'encoded_data' and the vocabulary
           size is saved as 'vocab_size'.
        vocab_file (str, optional): If provided, path where the
            vocabulary will be saved as a text file.
        ignore_whitespace (bool, optional): Whether to ignore text
            lines that are purely made of whitespace (default: True).

    Returns:
        array of int: Sequence of token IDs.
        int: Number of tokens in vocabulary.

    """

    # Get BERT tokenizer from Transformers
    import transformers
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = tokenizer.vocab_size
    if vocab_file:
        tokenizer.save_vocabulary(vocab_file)

    # Apply tokenizer to text file
    encoded_data = array.array('l')
    with open(text_file) as f:
        for line in f:
            if ignore_whitespace and line.isspace():
                continue
            encoded_data.extend(tokenizer.encode(line))
    if encoded_file:
        np.savez_compressed(encoded_file,
                            encoded_data=encoded_data,
                            vocab_size=vocab_size)
    return encoded_data, vocab_size


class GutenbergCorpus():
    """Tokenized text from Project Gutenberg.

    Args:
        data_dir (str): Directory for downloading data and
            intermediate.
        data_url (str): URL to Project Gutenberg text file.

    Attributes:
        token_data (array of int): Sequence of token IDs.
        vocab_size (int): Number of tokens in vocabulary.

    """
    def __init__(self, data_dir, data_url):

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
            token_data = data['encoded_data']
            vocab_size = int(data['vocab_size'])
        else:
            text_data_file = os.path.join(data_dir, 'text_data.txt')
            if not os.path.isfile(text_data_file):
                raw_file = os.path.join(data_dir, 'raw.txt')
                if not os.path.isfile(raw_file):
                    urllib.request.urlretrieve(data_url,
                                               filename=raw_file)
                strip_boilerplate(raw_file, text_data_file)
            vocab_file = os.path.join(data_dir, 'vocab.txt')
            token_data, vocab_size = tokenize(text_data_file,
                                              token_data_file,
                                              vocab_file)

        # Class members
        self.token_data = token_data
        self.vocab_size = vocab_size

    def __iter__(self):
        """Iterator through token IDs."""
        return self.token_data.__iter__()
    def __getitem__(self, key):
        """Get token ID."""
        return self.token_data.__getitem__(key)
    def __len__(self):
        """Get total number of tokens in corpus."""
        return self.token_data.__len__()
