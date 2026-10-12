from dataclasses import dataclass
import numpy.random as npr


@dataclass
class Sentence:
    words: list[str]

    def cat(self) -> str:
        return "".join(self.words)


@dataclass
class Document:
    sentences: list[Sentence]


@dataclass
class Corpus:
    documents: list[Document]

    def sample(self, rng=npr.default_rng(0)) -> Document:
        return self.documents[rng.integers(low=0, high=len(self.documents))]


class TrieList:
    word_end = -1

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = {}
        self.word_end = -1
        self.par_list = []
        self.par_dict = dict()
        self.relationship = dict()
        self.root['skipgram'] = dict()

    def insert(self, index, word, note=True):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        curNode = self.root
        for c in word:
            if not c in curNode:
                curNode[c] = {}
            curNode = curNode[c]

        if self.word_end in curNode:
            raise Exception(f'already has "{word}"')
        else:
            curNode[self.word_end] = note
        
        if index == -1:
            self.par_list.append(word)
            self.par_dict[word] = len(self.par_list) - 1
        else:
            self.par_list.insert(index, word)
            self.par_dict[word] = index
       
    def append(self, word, note=True):
        self.insert(-1, word, note)

    def _del(self, word): # delete from the trie and dict
        curNode = self.root
        def f(word, curNode):
            if len(word) == 0:
                del curNode[-1]
            else:  
                f(word[1:], curNode[word[0]])
                if len(curNode[word[0]]) == 0:
                    del curNode[word[0]]

        f(word, curNode)
        del self.par_dict[word]
    
    def pop(self, index=-1):
        word = self.par_list.pop(index)
        self._del(word)
        return word

    def exchange(self, i, j):
        self.par_list[j], self.par_list[i] = self.par_list[i], self.par_list[j]

    def move(self, i: int, step: int):
        """
        Shift the type at index ``i`` by ``step`` positions in the ranked lexicon.

        Negative ``step`` promotes toward the head (lower ordinal); positive
        ``step`` demotes toward the tail. If demotion would move the type past
        the end of the list, the type is removed entirely (active forgetting).

        Historical note: an earlier demotion policy never deleted. When a step
        would have gone past the end, it clamped the type onto the last slot
        instead, so the lexicon could shrink only via the separate passive
        probation / life mechanism. The delete-on-overflow behaviour below
        matches the LiB paper's active-forgetting rule.
        """
        w = self.par_list[i]

        if step < 0 and i != 0:  # Promotion
            self.par_list[i+step+1: i+1] = self.par_list[i+step: i]
            self.par_list[i+step] = w
            self.par_dict[w] = i + step
        elif step > 0:  # Demotion
            if i+step < len(self.par_list):  # Demotion to within the list
                self.par_list[i:i+step] = self.par_list[i+1: i+step+1]
                self.par_list[i+step] = w
                self.par_dict[w] = i + step
            else:  # Drop past tail
                self.par_list[i:] = self.par_list[i+1:]
                self._del(w)

    def group_move(self, words, update_rate):
        """
        Apply a sequence of reward/punish events to the ranked lexicon.

        ``words`` is an iterable of ``(type, e)`` pairs from training: ``e == -1``
        means reward (promote), ``e == 1`` means punish (demote). Each event is
        applied in order with a step of size roughly ``|e| * (index * update_rate)``.

        Historical note: a discarded alternative first counted how many times
        each type was rewarded or punished in the batch, then moved it once with
        a step that grew geometrically in that count, ``(1 ± update_rate)**n``.
        The live rule treats every event separately with a linear step, which is
        closer to the paper's per-update re-ranking formula.
        """
        self.update_ind()  # make "index_with_prior" works

        offset = 0
        for w, e in words:
            if w in self.par_dict:
                raw_ind = self.index_with_prior(w, offset)

                if raw_ind is not None and raw_ind > 0:
                    if e == 1: 
                        offset += 1

                    step = e * (int((raw_ind) * update_rate) + 1)                        
                    self.move(raw_ind, step)

        self.update_ind()

    def group_remove(self, words):
        if len(words) > 0:
            words_ = []
            for w in words: 
                if w in self.par_dict:
                    self._del(w)
                    words_.append(w)

            w_set = set(words_)
            self.par_list = [i for i in self.par_list if i not in w_set]

            self.update_ind()

    def index(self, word):
        return self.par_list.index(word)

    def index_with_prior(self, word, offset=0, nothing=False):
        try:
            prior_ind = self.par_dict[word]
            if offset == 0:
                return self.par_list.index(word, prior_ind)
            else:
                start = prior_ind - offset 
                if start < 0:
                    start = 0
                return self.par_list.index(word, start)
        except:
            if nothing==False:
                raise
            return nothing

    def update_ind(self):
        self.par_dict = dict((word,ind) for ind,word in enumerate(self.par_list))

    def remove(self, word):
        ind = self.index_with_prior(word)

        self._del(word)
        self.par_list[ind: ind + 1] = []

    def be_a_part(self, string):
        if not self.search(string):
            return False

        curNode = self.root
        for c in string:
            curNode = curNode[c]
        return (len(curNode) > 1)

    def startwith(self, string):
        if not self.search(string):
            return []
        
        chunks = []
        curNode = self.root
        for c in string:
            curNode = curNode[c]
            
        def f(pre, curNode):
            if len(curNode) == 1:
                chunks.append(pre)
            for i in curNode:
                if i!=-1:
                    f(pre+i, curNode[i])
        f(string, curNode)
        return chunks

    def has_a_larger(self, string):
        if not self.search(string):
            return None
 
        curNode = self.root
        for c in string:
            curNode = curNode[c]
            
        def f(pre, curNode):
            subs = [i for i in curNode if i!=-1]
            if len(subs) == 0:
                return pre
            elif len(subs) == 1:
                for i in subs:
                    return f(pre+i, curNode[i])
            else:
                return None

        if curNode == {-1: True}:
            return None
        else:
            return f(string, curNode)

    def search(self, word):
        return word in self.par_dict

    def skipgram_match(self, chunks: list[str]):
        current_node = self.root['skipgram']
        gram = []

        for c in chunks:
            if c in current_node:
                gram.append(c)
                if len(gram) > 1:
                    break
                else:
                    current_node = current_node[c]
            elif len(gram) > 0:
                gram.append(c)
        else:
            gram = []

        if len(gram) > 2:
            return [gram[0],gram[-1]], gram[1:-1], chunks[chunks.index(gram[0])+1:]
        else:
            return None, None, None

    def match(self, string: str):
        curNode = self.root
        words = ['','']
        for c in string:
            if not c in curNode:
                if -1 in curNode:
                    return words[-1]
                else:
                    return words[-2]
                
            if -1 in curNode:
                words.append(words[-1]) # we know there was a chunk, now we make a new hypothesis
                
            words[-1] += c
            curNode = curNode[c]
        if -1 in curNode:
            return words[-1]
        else:
            return words[-2]

    def match_two(self, string: str):
        curNode = self.root
        words = ['','']
        for c in string:
            if not c in curNode:
                if -1 in curNode:
                    if len(words) > 1:
                        return words[-2:]
                    else:
                        return ['',words[-1]]
                else:
                    if len(words) > 2:
                        return words[-3:-1]
                    else:
                        return ['','']

            if -1 in curNode:
                words.append(words[-1]) # we know there was a chunk, now we make a new hypothesis

            words[-1] += c

            curNode = curNode[c]
        if -1 in curNode:
            return words[-2:]
        elif len(words) > 2:
            return words[-3:-1]
        else:
            return ['','']
        
    def match_2nd(self, string):
        curNode = self.root
        words = ['','']
        for c in string:
            if -1 in curNode:
                words.append(words[-1]) # we know there was a chunk, now we make a new hypothesis
            if not c in curNode:
                if -1 in curNode:
                    return words[-2]
                elif len(words) > 3:
                    return words[-3]
                else:
                    return ''
            words[-1] += c
            curNode = curNode[c]
        if -1 in curNode:
            return words[-2]
        elif len(words) > 3:
            return words[-3]
        else:
            return ''

    def __iter__(self):
        return iter(self.par_list)
            
    def __contains__(self, word):
        return self.search(word)
    
    def __repr__(self):
        return str(self.par_list)
    
    def __len__(self):
        return len(self.par_list)
    
    def __set__(self):
        return set(self.par_list)
    
    def __getitem__(self, key):
        return self.par_list[key]
    
    def __setitem__(self, key, value):
        self._del(self.par_list[key])

        curNode = self.root
        for c in value:
            if not c in curNode:
                curNode[c] = {}
            curNode = curNode[c]
          
        curNode[self.word_end] = True

        self.par_list[key] = value

    def debug(self, mode):
        from .visualisation import Visualiser
        from .segmentation import TwoGreedySegmenter
        return Visualiser(TwoGreedySegmenter(self), self).debug_lexicon(self, mode=mode)
