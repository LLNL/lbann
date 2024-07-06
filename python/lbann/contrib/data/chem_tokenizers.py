"""
Contains a variety of molecular tokenizers for use with dataset readers.
"""
from enum import Enum, auto
import re
from typing import Dict, Type

try:
    from transformers import BertTokenizer
except (ModuleNotFoundError, ImportError):
    print('This file requires Huggingface Transformers to run.')
    raise

try:
    import selfies
except (ModuleNotFoundError, ImportError):
    selfies = None

try:
    import atomInSmiles
except (ModuleNotFoundError, ImportError):
    atomInSmiles = None


class ChemTokenType(Enum):
    SMILES = auto()
    SELFIES = auto()
    AIS = auto()


class ChemTokenizer(BertTokenizer):

    @staticmethod
    def from_smiles(text: str) -> str:
        raise NotImplementedError

    @staticmethod
    def from_selfies(text: str) -> str:
        raise NotImplementedError

    @staticmethod
    def from_ais(text: str) -> str:
        raise NotImplementedError


class SMILESTokenizer(ChemTokenizer):
    """
    SMILES tokenizer
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, do_lower_case=False, **kwargs)

        # Pattern adopted from atomsInSmiles and the Regression Transformer repositories
        SMILES_TOKENIZER_PATTERN = r"(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
        self.regexp = re.compile(SMILES_TOKENIZER_PATTERN)

    def _tokenize(self, text: str, **kwargs):
        text = text[text.find('|') + 1:]
        return list(self.regexp.findall(text))

    @staticmethod
    def from_smiles(text: str):
        return text

    @staticmethod
    def from_selfies(text: str):
        return selfies.decoder(text)

    @staticmethod
    def from_ais(text: str):
        return atomInSmiles.decode(text)


class SELFIESTokenizer(ChemTokenizer):
    """
    SELFIES tokenizer.

    Self-Referencing Embedded Strings (SELFIES): A 100% robust molecular string
    representation. Mario Krenn, Florian Haese, AkshatKumar Nigam,
    Pascal Friederich, Alan Aspuru-Guzik.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, do_lower_case=False, **kwargs)
        if selfies is None:
            raise ImportError(
                'The SELFIES package is required for this tokenizer, '
                'install with ``pip install selfies``')

    def _tokenize(self, text: str, **kwargs):
        text = text[text.find('|') + 1:]
        return list(selfies.split_selfies(text))

    @staticmethod
    def from_smiles(text: str):
        return selfies.encoder(text)

    @staticmethod
    def from_selfies(text: str):
        return text

    @staticmethod
    def from_ais(text: str):
        return selfies.encoder(atomInSmiles.decode(text))


class AISTokenizer(ChemTokenizer):
    """
    Atom-in-SMILES tokenizer.

    Ucak UV, Ashyrmamatov I, Lee J (2023) Improving the quality of chemical
    language model outcomes with atom-in-SMILES tokenization. J Cheminformatics
    15:55. https://doi.org/10.1186/s13321-023-00725-9
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, do_lower_case=False, **kwargs)
        if atomInSmiles is None:
            raise ImportError(
                'The Atom-in-SMILES package is required for this tokenizer, '
                'install with ``pip install atomInSmiles``')

    def _tokenize(self, text: str, **kwargs):
        text = text[text.find('|') + 1:]
        return text.split(' ')

    @staticmethod
    def from_smiles(text: str):
        return atomInSmiles.encode(text)

    @staticmethod
    def from_selfies(text: str):
        return atomInSmiles.encode(selfies.decoder(text))

    @staticmethod
    def from_ais(text: str):
        return text


TOKENIZERS: Dict[ChemTokenType, Type[ChemTokenizer]] = {
    ChemTokenType.SMILES: SMILESTokenizer,
    ChemTokenType.SELFIES: SELFIESTokenizer,
    ChemTokenType.AIS: AISTokenizer,
}
