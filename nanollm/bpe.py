from typing import Any

def get_pair(ids):
    counts = {}
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

class BasicTokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocabulary = {i : bytes([i]) for i in range(256)}
        self.merges = {}

    def train(self, text):
        # Encode the text
        # iterate over text, (self.vocab_size - 256) times
        # count all of the pairs in a dict
        # choose the pair with highest freq
        # merge that pair as a new token
        # add that token to the vocab like {256 : byte_string}
        # add to self.merges
        assert self.vocab_size > 256, "No merges"
        number_merges = self.vocab_size - 256

        byte_strings = text.encode('utf-8')
        ids = list[Any](byte_strings)

        for i in range(number_merges):
            pairs = get_pair(ids)
            pair = max(pairs, key=pairs.get)
            index = 256 + i
            ids = merge(ids, pair, index)

        print(ids)

        # print(sorted([(v, k) for k, v in pairs.items()], reverse=True)[:10])

    def encode():
        pass

    def decode():
        pass


tokenizer = BasicTokenizer(266)
text = """The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to "support
Unicode" in our software (whatever that means-like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports,"""
tokenizer.train(text)