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
    def __init__(self, vocab_size, verbose=False):
        self.vocab_size = vocab_size
        self.vocabulary = {i : bytes([i]) for i in range(256)}
        self.merges = {}
        self.verbose = verbose

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

        byte_strings = text.encode('utf-8')
        ids = list[Any](byte_strings)

        length_initial = len(ids)

        for i in range(number_merges):
            pairs = get_pair(ids)
            pair = max(pairs, key=pairs.get)
            index = 256 + i
            ids = merge(ids, pair, index)
            self.merges[pair] = index
            self.vocabulary[index] = self.vocabulary[pair[0]] + self.vocabulary[pair[1]]

        if self.verbose:
            length_final = len(ids) 
            compression = length_initial / length_final

            print(length_final, length_initial)
            print(compression)
            print(ids)

        # print(sorted([(v, k) for k, v in pairs.items()], reverse=True)[:10])

    def encode(self, text):
        """
        self.merges is important here
        we get text, then we convert that text to byte strings and then to integers
        """
        # Step 1: raw bytes → list of ints
        ids = list(text.encode('utf-8'))
    
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


tokenizer = BasicTokenizer(266, False)
text = """The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to "support
Unicode" in our software (whatever that means-like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports,"""
tokenizer.train(text)

encoded_txt = tokenizer.encode("My name is Parikshit.")
decoded_txt = tokenizer.decode(encoded_txt)

print(encoded_txt)
print(decoded_txt)

print(tokenizer.encode('are hello'))
print(list[int]('are hello'.encode('utf-8')))