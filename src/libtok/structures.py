from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Optional, Sequence, Union

import numpy.random as npr
from numpy.random import Generator as RNG


class Bound(Enum):
    """Sentence boundary markers used by the skip-gram path."""
    BOS = "bos"
    EOS = "eos"


@dataclass(frozen=True)
class SkipGram:
    """Skip-gram template: left and right anchors with a fill in between.

    Frozen so instances are hashable and equal when ``left``/``right`` match.
    """
    left: Union[str, Bound]
    right: Union[str, Bound]


VocabEntry = Union[str, SkipGram]
Chunk = Union[str, Bound]


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

    def sample(self, rng: RNG = npr.default_rng(0)) -> Document:
        return self.documents[rng.integers(low=0, high=len(self.documents))]


class TrieList:
    word_end = -1

    def __init__(self):
        self.root = {}
        self.word_end = -1
        self.types_sorted = []
        self.types_indexed = dict()

        self.skipgram_root = {}
        self.skipgram_frames: dict[str, set[SkipGram]] = dict()  # fill string → skip-gram frames that produced it

    def _trie_and_path(self, word: VocabEntry) -> tuple[dict, Sequence]:
        if isinstance(word, SkipGram):
            return self.skipgram_root, (word.left, word.right)
        return self.root, word

    def insert(self, index: int, word: VocabEntry, note: Any = True):
        """
        Inserts a word into the trie.
        """
        cur_node, path = self._trie_and_path(word)
        for c in path:
            if not c in cur_node:
                cur_node[c] = {}
            cur_node = cur_node[c]

        if self.word_end in cur_node:
            raise Exception(f'already has "{word}"')
        else:
            cur_node[self.word_end] = note
        
        if index == -1:
            self.types_sorted.append(word)
            self.types_indexed[word] = len(self.types_sorted) - 1
        else:
            self.types_sorted.insert(index, word)
            self.types_indexed[word] = index
       
    def append(self, word: VocabEntry, note: Any = True):
        self.insert(-1, word, note)

    def _unlink(self, word: VocabEntry):
        """Drop ``relationship`` entries that refer to a removed type."""
        if isinstance(word, str):
            self.skipgram_frames.pop(word, None)
        elif isinstance(word, SkipGram):
            empty = []
            for fill, frames in self.skipgram_frames.items():
                frames.discard(word)
                if not frames:
                    empty.append(fill)
            for fill in empty:
                del self.skipgram_frames[fill]

    def _del(self, word: VocabEntry): # delete from the trie and dict
        cur_node, path = self._trie_and_path(word)
        def f(remaining, cur):
            if len(remaining) == 0:
                del cur[-1]
            else:  
                f(remaining[1:], cur[remaining[0]])
                if len(cur[remaining[0]]) == 0:
                    del cur[remaining[0]]

        f(path, cur_node)
        del self.types_indexed[word]
        self._unlink(word)
    
    def pop(self, index: int = -1) -> VocabEntry:
        word = self.types_sorted.pop(index)
        self._del(word)
        return word

    def exchange(self, i: int, j: int):
        self.types_sorted[j], self.types_sorted[i] = self.types_sorted[i], self.types_sorted[j]

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
        w = self.types_sorted[i]

        if step < 0 and i != 0:  # Promotion
            self.types_sorted[i + step + 1: i + 1] = self.types_sorted[i + step: i]
            self.types_sorted[i + step] = w
            self.types_indexed[w] = i + step
        elif step > 0:  # Demotion
            if i+step < len(self.types_sorted):  # Demotion to within the list
                self.types_sorted[i:i + step] = self.types_sorted[i + 1: i + step + 1]
                self.types_sorted[i + step] = w
                self.types_indexed[w] = i + step
            else:  # Drop past tail
                self.types_sorted[i:] = self.types_sorted[i + 1:]
                self._del(w)

    def group_move(self, words: Iterable[tuple[VocabEntry, int]], update_rate: float):
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
            if w in self.types_indexed:
                raw_ind = self.index_with_prior(w, offset)

                if raw_ind is not None and raw_ind > 0:
                    if e == 1: 
                        offset += 1

                    step = e * (int((raw_ind) * update_rate) + 1)                        
                    self.move(raw_ind, step)

        self.update_ind()

    def group_remove(self, words: Sequence[VocabEntry]):
        if len(words) > 0:
            words_ = []
            for w in words: 
                if w in self.types_indexed:
                    self._del(w)
                    words_.append(w)

            w_set = set(words_)
            self.types_sorted = [i for i in self.types_sorted if i not in w_set]

            self.update_ind()

    def index(self, word: VocabEntry) -> int:
        return self.types_sorted.index(word)

    def index_with_prior(self, word: VocabEntry, offset: int=0, nothing: Optional[int]=None) -> Optional[int]:
        try:
            prior_ind = self.types_indexed[word]
            return self.types_sorted.index(word, max(0, prior_ind - offset))
        except:
            if nothing is None:
                raise KeyError(word)
            return nothing

    def update_ind(self):
        self.types_indexed = dict((word, ind) for ind,word in enumerate(self.types_sorted))

    def remove(self, word: VocabEntry):
        ind = self.index_with_prior(word)

        self._del(word)
        self.types_sorted[ind: ind + 1] = []

    def be_a_part(self, string: str) -> bool:
        if not self.search(string):
            return False

        cur_node = self.root
        for c in string:
            cur_node = cur_node[c]
        return len(cur_node) > 1

    def startwith(self, string: str) -> list[str]:
        if not self.search(string):
            return []
        
        chunks = []
        cur_node = self.root
        for c in string:
            cur_node = cur_node[c]
            
        def f(pre, cur):
            if len(cur) == 1:
                chunks.append(pre)
            for i in cur:
                if i!=-1:
                    f(pre + i, cur[i])
        f(string, cur_node)
        return chunks

    def has_a_larger(self, string: str) -> Optional[str]:
        if not self.search(string):
            return None
 
        cur_node = self.root
        for c in string:
            cur_node = cur_node[c]
            
        def f(pre, cur):
            subs = [i for i in cur if i != -1]
            if len(subs) == 0:
                return pre
            elif len(subs) == 1:
                for i in subs:
                    return f(pre + i, cur[i])
            else:
                return None

        if cur_node == {-1: True}:
            return None
        else:
            return f(string, cur_node)

    def search(self, word: VocabEntry) -> bool:
        return word in self.types_indexed

    def skipgram_match(self, chunks: list[Chunk]) -> tuple[Optional[list[Chunk]], Optional[list[Chunk]], Optional[list[Chunk]]]:
        current_node = self.skipgram_root
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

    def match(self, string: str) -> str:
        cur_node = self.root
        words = ['','']
        for c in string:
            if not c in cur_node:
                if -1 in cur_node:
                    return words[-1]
                else:
                    return words[-2]
                
            if -1 in cur_node:
                words.append(words[-1]) # we know there was a chunk, now we make a new hypothesis
                
            words[-1] += c
            cur_node = cur_node[c]
        if -1 in cur_node:
            return words[-1]
        else:
            return words[-2]

    def match_two(self, string: str) -> tuple[str,str]:
        cur_node = self.root
        words = ['','']
        for c in string:
            if not c in cur_node:
                if -1 in cur_node:
                    if len(words) > 1:
                        return words[-2:]
                    else:
                        return ('', words[-1])
                else:
                    if len(words) > 2:
                        return words[-3:-1]
                    else:
                        return ('', '')

            if -1 in cur_node:
                words.append(words[-1]) # we know there was a chunk, now we make a new hypothesis

            words[-1] += c

            cur_node = cur_node[c]
        if -1 in cur_node:
            return words[-2:]
        elif len(words) > 2:
            return words[-3:-1]
        else:
            return ('', '')

    def match_2nd(self, string: str) -> str:
        cur_node = self.root
        words = ['','']
        for c in string:
            if -1 in cur_node:
                words.append(words[-1]) # we know there was a chunk, now we make a new hypothesis
            if not c in cur_node:
                if -1 in cur_node:
                    return words[-2]
                elif len(words) > 3:
                    return words[-3]
                else:
                    return ''
            words[-1] += c
            cur_node = cur_node[c]
        if -1 in cur_node:
            return words[-2]
        elif len(words) > 3:
            return words[-3]
        else:
            return ''

    def __iter__(self):
        return iter(self.types_sorted)
            
    def __contains__(self, word: VocabEntry) -> bool:
        return self.search(word)
    
    def __repr__(self) -> str:
        return str(self.types_sorted)
    
    def __len__(self) -> int:
        return len(self.types_sorted)
    
    def __set__(self) -> set:
        return set(self.types_sorted)
    
    def __getitem__(self, key: Union[int, slice]) -> Any:
        return self.types_sorted[key]
    
    def __setitem__(self, key: int, value: VocabEntry):
        self._del(self.types_sorted[key])

        cur_node, path = self._trie_and_path(value)
        for c in path:
            if not c in cur_node:
                cur_node[c] = {}
            cur_node = cur_node[c]
          
        cur_node[self.word_end] = True

        self.types_sorted[key] = value

    def debug(self, mode: str) -> list:
        from .visualisation import LiBVisualiser
        from .segmentation import TwoGreedySegmenter
        return LiBVisualiser(TwoGreedySegmenter(self), self).debug_lexicon(self, mode=mode)
