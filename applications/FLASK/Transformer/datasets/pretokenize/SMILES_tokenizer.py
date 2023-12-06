import regex as re
import json
import numpy as np
import os


PATTERN = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"


class MolTokenizer():
    def __init__(self, vocab_file: str = '',
                 do_lower_case=False,
                 unk_token='<pad>',
                 sep_token='<eos>',
                 pad_token='<pad>',
                 cls_token='<bos>',
                 mask_token='<mask>',
                 **kwargs):
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.sep_token = sep_token
        self.cls_token = cls_token
        self.mask_token = mask_token 
        self.regex_tokenizer = re.compile(PATTERN)
        self.wordpiece_tokenizer = None
        self.basic_tokenizer = None
        self.vocab_file = vocab_file

    def load_vocab_file(self):
        if os.path.exists(self.vocab_file):
            with open(self.vocab_file, 'r') as f:
              self.vocab = json.load(f)
        else:
            raise NameError("Vocab file not douns")

    def _tokenize(self, text):
        split_tokens = self.regex_tokenizer.findall(text)
        return split_tokens

    def tokenize(self, text):
        split_tokens = self._tokenize(text)
        output = np.zeros(len(split_tokens))
        for i, each in enumerate(split_tokens):
            output[i] = self.vocab[each]
        return output

    def encode(self, token):
        return self.vocab[token]

    def convert_tokens_to_string(self, tokens):
        out_string = "".join(tokens).strip()
        return out_string  
    
    def generate_vocab_file(self, dataset_name):
        vocab_dict = {}
        vocab_dict[self.pad_token] = 0
        vocab_dict[self.sep_token] = 1
        vocab_dict[self.cls_token] = 2
        vocab_dict[self.mask_token] = 3

        counter = 4
        with open(dataset_name, 'r') as f:
            for smiles in f:
                for token in self._tokenize(smiles.strip()):
                    if not token in vocab_dict.keys():
                        vocab_dict[token] = counter
                        counter += 1

        with open(self.vocab_file, 'w') as f:
          f.write(json.dumps(vocab_dict))
