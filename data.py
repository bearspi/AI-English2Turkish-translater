import torch, os 
from io import open

class Dictionary(object):
    def __init__(self) -> None:
        self.word2idx = {}
        self.idx2word = []
    
    
    def add_word(self, word) -> int:
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            return self.word2idx[word]
    
    def __len__(self) -> int:
        return len(self.idx2word)
    
class Corpus(object):
    def __init__(self, path) -> None:
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))
    
    def tokenize(self, path):
        
        assert os.path.exists(path)
        
        with open(file=path, mode="r", encoding="utf8") as f:
            for line in f:
                words = line.split()
                for word in words:
                    self.dictionary.add_word(word)
        
        with open(file=path, mode="r", encoding="utf8") as f:
            ids = []
            for line in f:
                ids_line = []
                words = line.split()
                for word in words:
                    ids_line.append(self.dictionary.word2idx[word])
                ids.append(torch.tensor(ids_line).type(torch.int64))
            ids = torch.concat(ids)
            return ids
