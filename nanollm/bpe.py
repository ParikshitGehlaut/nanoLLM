from typing import Any
import regex as re
import pickle

def get_pair(ids, counts):
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, index):
    """
    This function will iterate over ids and every time, it sees a instance of pair,
    it will replace pair with index 
    """
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
            new_ids.append(index)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1

    return new_ids

class RegexTokenizer:
    def __init__(self, vocab_size, verbose=False):
        self.vocab_size = vocab_size
        self.vocabulary = {i : bytes([i]) for i in range(256)}
        self.merges = {}
        self.verbose = verbose
        self.pattern = re.compile(
        r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")

    def train(self, text):
        # Encode the text
        # iterate over text, (self.vocab_size - 256) times
        # count all of the pairs in a dict
        # choose the pair with highest freq
        # merge that pair as a new token
        # add that token to the vocab like {256 : byte_string}
        # add to self.merges
        assert self.vocab_size >= 256, "vocab size must be at least 256"
        number_merges = self.vocab_size - 256

        text_chunks = re.findall(self.pattern, text)
        encoded_chunks = [list(text_chunk.encode('utf-8')) for text_chunk in text_chunks]

        for i in range(number_merges):
            pairs = {}
            for encoded_chunk in encoded_chunks:
                get_pair(encoded_chunk, pairs)
            pair = max(pairs, key=pairs.get)
            index = 256 + i
            encoded_chunks = [merge(encoded_chunk, pair, index) for encoded_chunk in encoded_chunks]
            self.merges[pair] = index
            self.vocabulary[index] = self.vocabulary[pair[0]] + self.vocabulary[pair[1]]

    def encode(self, text):
        text_chunks = re.findall(self.pattern, text)
        encoded_text = []

        for text_chunk in text_chunks:
            encoded_chunk = self._encode_chunk(text_chunk)
            encoded_text.extend(encoded_chunk)

        return encoded_text
        
    def _encode_chunk(self, text):
        """
        self.merges is important here
        we get text, then we convert that text to byte strings and then to integers
        """
        ids = list[Any](text.encode('utf-8'))

        # Step 2: apply merges in training order
        for pair, index in self.merges.items():
            ids = merge(ids, pair, index)

        return ids

    def decode(self, ids):
        """
        decode get ids
        1. convert the ids to their byte strings
        2. covert the byte strings to strings via the vocabulary
        """
        # step 1 : look up each id in vocabulary -> get its byte string
        # step 2 : join all bytes strings together
        token_bytes = b"".join(self.vocabulary[i] for i in ids)
        
        # step 3 : decode the final bytes as 'utf-8'
        return token_bytes.decode('utf-8')
    
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({
                "merges": self.merges,
                "vocabulary": self.vocabulary,
                "pattern" : self.pattern.pattern,
            }, f)

    @classmethod
    def load(cls, path):
        tokenizer = cls.__new__(cls) 

        with open(path, "rb") as f:
            data = pickle.load(f)
            tokenizer.merges = data["merges"]
            tokenizer.vocabulary = data["vocabulary"]
            tokenizer.pattern = re.compile(data["pattern"])

        return tokenizer

tokenizer = RegexTokenizer(258, False)
text = """low low"""
tokenizer.train(text)

# encoded_txt = tokenizer.encode("My name is Parikshit.")
# decoded_txt = tokenizer.decode(encoded_txt)

# print(encoded_txt)
# print(decoded_txt)

# print(tokenizer.encode('are hello'))
# print(list[int]('are hello'.encode('utf-8')))