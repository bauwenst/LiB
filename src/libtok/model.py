from itertools import accumulate
from typing import Counter as CounterType, Iterable, Optional, Sequence

from .structures import Corpus, Document, TrieList, VocabEntry
from .vocabularisation import LiBVocabulariser
from .visualisation import LiBVisualiser


class LessIsBetter:
    """Facade composing Vocabulariser, TwoGreedySegmenter, and Visualiser."""

    def __init__(
        self,
        life: int = 10,
        max_len: int = 12,

        memory_in: float = 0.25,
        memory_out: float = 0.0001,
        update_rate: float = 0.2,
        mini_gap: int = 2,
        use_skip: bool = False
    ):
        """Arguments explained in Vocabulariser.__init__()"""
        self.vocabulariser = LiBVocabulariser(
            life=life,
            max_len=max_len,
            memory_in=memory_in,
            memory_out=memory_out,
            update_rate=update_rate,
            mini_gap=mini_gap,
            use_skip=use_skip,
        )
        self.segmenter = self.vocabulariser.segmenter
        self.visualiser = LiBVisualiser(self.segmenter, self.vocabulariser.memory)

    @property
    def memory(self) -> TrieList:
        return self.vocabulariser.memory

    @property
    def memory_log(self) -> list:
        return self.vocabulariser.memory_log

    @property
    def to_dropout(self) -> dict:
        return self.vocabulariser.to_dropout

    def initialise(self, corpus_samples: Iterable[Iterable[str]]):
        return self.vocabulariser.initialise(corpus_samples)

    def eval_memorizer(self, doc_covered: Sequence[int]):
        return self.vocabulariser.eval_memorizer(doc_covered)

    def dropout(
        self,
        to_test: dict,
        doc: str,
        doc_covered: Sequence[int],
        doc_loc: int,
        reward_list: list,
        strict: bool = False,
    ):
        return self.vocabulariser.dropout(to_test, doc, doc_covered, doc_loc, reward_list, strict=strict)

    def batch_update_memory(self, reward_list: list, used_chunks: CounterType):
        return self.vocabulariser.batch_update_memory(reward_list, used_chunks)

    def memorize(
        self,
        to_memorize: set,
        to_get: Optional[VocabEntry],
        reward_list: list,
    ):
        return self.vocabulariser.memorize(to_memorize, to_get, reward_list)

    def reading(self, article: Document):
        """See ``Vocabulariser.reading``."""
        return self.vocabulariser.reading(article)

    def get_f1(
        self,
        references: Sequence[Sequence[str]],
        candidates: Sequence[Sequence[str]],
    ) -> tuple[float, float, float]:
        return self.vocabulariser.get_micro_f1(references, candidates)

    def run(self, epochs: int, corpus_train: Corpus, corpus_test: Corpus):
        return self.vocabulariser.run(epochs, corpus_train, corpus_test)

    def training_step(
        self, epoch_id: int, corpus_train: Corpus, corpus_test: Corpus
    ):
        return self.vocabulariser.training_step(epoch_id, corpus_train, corpus_test)

    def find_subs(self, large_chunk: str, level: int = 2) -> tuple[str, ...]:
        """See ``TwoGreedySegmenter.find_subs``."""
        return self.segmenter.find_subs(large_chunk, level=level)

    def show_reading(
        self,
        article: Document,
        max_len: int = 10,
        decompose: bool = False,
        display: bool = False,
        return_chunks: bool = False,
        comb_sents: bool = True,
    ):
        """
        Inference-time segmentation of ``article`` (legacy name for ``segment``).

        Delegates to ``TwoGreedySegmenter.segment`` (two-greedy decode over the
        current lexicon). Optionally prints the chunks via the visualiser.

        Parameters
        ----------
        article :
            Document to segment.
        max_len :
            Match window in characters (default ``10``).
        decompose :
            If True, decompose chunks with ``find_subs``.
        display :
            If True, print the segmentation (space-separated chunks).
        return_chunks :
            If True, return chunk strings; if False, return boundary offsets.
        comb_sents :
            If True, flatten across sentences before return / display.

        Returns
        -------
        list or set
            Same as ``TwoGreedySegmenter.segment``.
        """
        chunks = self.segmenter.segment(
            article,
            max_len=max_len,
            decompose=decompose,
            return_chunks=True,
            comb_sents=comb_sents,
        )
        if display:
            self.visualiser.display_segmentation(chunks, comb_sents=comb_sents)
        if return_chunks:
            return chunks
        else:
            return set(accumulate([len(c) for c in chunks]))

    def show_result(self, article: Document, decompose: bool = False):
        return self.visualiser.show_result(article, decompose=decompose)

    def demo(
        self,
        article: Document,
        decompose: bool = False,
        section: tuple[int, Optional[int]] = (0, None),
    ):
        return self.visualiser.demo(article, decompose=decompose, section=section)

    def show_result_with_idx(self, article: Document, decompose: bool = False):
        return self.visualiser.show_result_with_idx(article, decompose=decompose)
