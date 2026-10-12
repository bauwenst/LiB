from dataclasses import dataclass
from typing import Counter as CounterType, Iterable, Optional, Sequence

from collections import Counter
from itertools import accumulate, groupby
import random
import numpy as np
import time

from .segmentation import TwoGreedySegmenter
from .structures import Bound, Corpus, Document, SkipGram, TrieList, VocabEntry


class LiBVocabulariser:
    """Online LiB lexicon trainer: builds a ranked vocabulary by reading documents."""

    def __init__(
        self,
        life: int = 10,
        max_len: int = 12,

        memory_in: float = 0.25,
        memory_out: float = 0.0001,
        update_rate: float = 0.2,
        mini_gap: int = 2,
        use_skip: bool = False,

        eval_interval: int = 100,
    ):
        """
        Parameters
        ----------
        life :
            Probation length ``τ₀`` for passive forgetting. After each training
            epoch, types in the tail of the lexicon are put on probation; if a
            type is not judged useful again within ``life`` epochs, it is removed.
            Higher for larger vocabulary.
        max_len :
            Maximum character length of a chunk considered during matching and
            when proposing concatenations of adjacent chunks to memorise.
            Higher for larger vocabulary.
        memory_in :
            Sampling probability ``α`` for proposing a new chunk (adjacent-pair
            or short unknown span) for memorisation. A type is appended only
            after it has been sampled at least twice (see ``memorize``).
            Higher for larger vocabulary.
        memory_out :
            Forgetting ratio ``ω``. Each epoch, the last ``ω · |L|`` lexicon
            entries (at least the final slice of the ranked list) are placed on
            probation via ``to_dropout``.
            Lower for larger vocabulary.
        update_rate :
            Re-ranking rate ``Δ``. After Good/Bad evaluation, a type's ordinal
            moves toward the head (reward) or tail (punish) by a step proportional
            to ``Δ``; demotion past ``|L|`` deletes the type (active forgetting).
            Lower for larger vocabulary (smaller demotion steps, less active deletion).
        mini_gap :
            Maximum length of an unknown span (or skip-gram fill) that may be
            proposed for memorisation. Longer gaps are ignored.
            Higher for larger vocabulary.
        use_skip :
            If True, enable the experimental skip-gram path: sample skip-gram
            slots and optionally memorise filled spans linked via
            ``memory.relationship`` (filled span → set of ``SkipGram`` frames).
            Enabling tends to grow the vocabulary (extra skip-gram types).
        """
        self._life = life
        self._max_len = max_len
        self._memory_in = memory_in
        self._memory_out = memory_out
        self._update_rate = update_rate
        self._mini_gap = mini_gap
        self._use_skip = use_skip

        self._eval_interval = eval_interval

        self.to_dropout = dict()
        self.memory = TrieList()
        self.memory_log = []
        self.segmenter = TwoGreedySegmenter(self.memory)

    def initialise(self, corpus_samples: Iterable[Iterable[str]]):
        self.to_dropout.clear()
        for i in set([i for j in corpus_samples for i in j]):
            self.memory.append(i)
            self.to_dropout[i] = self._life

    def eval_memorizer(self, doc_covered: Sequence[int]):
        return (1-sum([i==0 for i in doc_covered])/len(doc_covered),
                [(key, len(list(group))) for key, group in groupby(doc_covered)])

    def dropout(
        self,
        to_test: dict,
        doc: str,
        doc_covered: Sequence[int],
        doc_loc: int,
        reward_list: list,
        strict: bool = False,
    ):
        for chunk_to_test in list(to_test.keys()):
            if strict and not self.memory.search(chunk_to_test):
                del to_test[chunk_to_test]
                continue

            onset, end, N_unknowns, N_chunks, chunks_list_0, chunks_list = to_test[chunk_to_test]
            while end < doc_loc:
                chunk_next = self.memory.match(doc[end:end+self._max_len])
                chunks_list.append(chunk_next)
                chunk_size_next = len(chunk_next)
                if chunk_size_next > 0:
                    end += chunk_size_next
                    N_chunks += 1
                else:
                    end += 1
                    N_unknowns += 1
                    N_chunks += 1

            if end == doc_loc:
                del to_test[chunk_to_test]

                doc_section = doc_covered[onset: end]

                N_unknowns_0 = sum(len(c) == 0 for c in chunks_list_0)
                N_chunks_0 = sum(len(c) > 0 for c in chunks_list_0)

                redundant = N_unknowns_0 == N_unknowns and \
                              ((N_chunks_0 > N_chunks) or
                               (N_chunks_0 == N_chunks and
                                    sum(self.memory.index_with_prior(i) for i in chunks_list_0 if i in self.memory) > sum(self.memory.index_with_prior(i) for i in chunks_list if i in self.memory)))

                if N_unknowns_0 > N_unknowns or redundant:
                    smaller_chunk = self.memory.match(chunk_to_test[:-1])
                    reward_list.append((chunk_to_test, 1))
                    reward_list.append((smaller_chunk, -1))
                else:
                    reward_list.append((chunk_to_test, -1))

                    if chunk_to_test in self.to_dropout:
                        del self.to_dropout[chunk_to_test]
            else:
                to_test[chunk_to_test] = onset, end, N_unknowns, N_chunks, chunks_list_0, chunks_list

    def batch_update_memory(
        self,
        reward_list: list,
        used_chunks: CounterType,
    ):
        to_del = []
        for chunk_to_test in list(self.to_dropout.keys()):
            current_life = self.to_dropout[chunk_to_test]
            if current_life <= 1:
                del self.to_dropout[chunk_to_test]
                to_del.append(chunk_to_test)
            else:
                self.to_dropout[chunk_to_test] = current_life - 1

        for c, label in reward_list:
            if c in self.memory.skipgram_frames:
                for frame in self.memory.skipgram_frames[c]:
                    reward_list.append((frame, label))

        self.memory.group_remove(to_del)

        reward_list = [(w,e) for w,e in reward_list if w not in set(to_del)]
        self.memory.group_move(reward_list, self._update_rate)

        forget_samples = range(int((1 - self._memory_out) * len(self.memory)), len(self.memory))

        for ind in forget_samples:
            chunk_t = self.memory[ind]
            if chunk_t not in self.to_dropout:
                self.to_dropout[chunk_t] = self._life

    def memorize(
        self,
        to_memorize: set,
        to_get: Optional[VocabEntry],
        reward_list: list,
    ):
        if to_get and not self.memory.search(to_get):
            if to_get in to_memorize:
                to_memorize.remove(to_get)
                self.memory.append(to_get)
            else:
                to_memorize.add(to_get)

    def reading(self, article: Document):
        """
        One training pass over a document: segment, memorise, evaluate, update.

        Walks each sentence left to right with larger-first ``match_two``. The
        primary (longest) match is always committed as the training segmentation;
        when a second-longest alternative exists, ``dropout`` later compares the
        two continuations and emits Good/Bad rewards for lexicon re-ranking.
        Adjacent chunks (and short unknown spans) may be sampled into the
        lexicon; then ``batch_update_memory`` applies passive forgetting,
        re-ranking, and active deletion.

        Parameters
        ----------
        article :
            Document whose sentences are concatenated per sentence and read in
            order.

        Returns
        -------
        covered_rate : float
            Fraction of character positions covered by at least one known chunk.
        avg_chunk_len : float
            Characters per effective chunk group (working-memory load proxy).
        mem_usage : float
            Fraction of lexicon types that appeared in this document's
            segmentation.
        in_count : int
            Net number of types appended to the lexicon during this pass
            (before forgetting removals in ``batch_update_memory``).
        """
        used_chunks = Counter()

        doc_covered_all = []
        reward_list = []
        punish_list= []
        to_memorize = set()
        l_memory_size = len(self.memory)
        new_memorized = Counter()

        for sentence in article.sentences:
            concatenated_sentence = sentence.cat()

            chunks_in_sent = []
            singles = []
            to_test = dict()
            last_chunk = ['', 0] # 0 unknown, 1 known

            sent_covered = [0] * len(concatenated_sentence)
            i = 0
            while i < len(concatenated_sentence):
                self.dropout(to_test, concatenated_sentence, sent_covered, i, reward_list)

                chunk_2nd, chunk = self.memory.match_two(concatenated_sentence[i:i+self._max_len])

                for chunk_to_test in list(to_test.keys()):
                    to_test[chunk_to_test][4].append(chunk)

                if len(chunk) > 0:
                    if len(last_chunk) + len(chunk) <= self._max_len and random.random() < self._memory_in:
                        to_get = None
                        if last_chunk[1] == 0:
                            if 0 < len(last_chunk[0]) <= self._mini_gap:
                                to_get = last_chunk[0]
                        elif len(last_chunk[0] + chunk) <= self._max_len:
                            to_get = last_chunk[0] + chunk
                        self.memorize(to_memorize, to_get, reward_list)

                    if len(chunk) > 1 and len(chunk_2nd) > 0:
                        to_test[chunk] = [i, i + len(chunk_2nd), 0, 1, [chunk], [chunk_2nd]] # chunk_to_test, start position, current position, number of unknowns, number of chunks
                    elif chunk in self.to_dropout:
                        del self.to_dropout[chunk]

                    chunk_s = len(chunk)
                    sent_covered[i: i + chunk_s] = [sent_covered[i-1] + 1] * chunk_s
                    chunks_in_sent.append(chunk)
                    i += chunk_s
                    last_chunk = [chunk, 1]
                    used_chunks[chunk] += 1
                else:
                    if last_chunk[1] == 1:
                        last_chunk = ['', 0]
                    last_chunk[0] += concatenated_sentence[i]
                    chunks_in_sent.append(concatenated_sentence[i])
                    i += 1

            if self._use_skip:
                chunks_in_sent = [Bound.BOS] + chunks_in_sent + [Bound.EOS]
                for a, b in zip(chunks_in_sent[:-2], chunks_in_sent[2:]):
                    if random.random() < (self._memory_in * self._memory_in):
                        self.memorize(to_memorize, SkipGram(a, b), reward_list)

                while True:
                    skip_gram, skip, chunks_in_sent = self.memory.skipgram_match(chunks_in_sent)
                    if skip is not None and len(skip) > 1:
                        skip = ''.join(skip)
                        if len(skip) <= self._mini_gap and random.random() < self._memory_in:
                            frame = SkipGram(skip_gram[0], skip_gram[1])
                            self.memory.skipgram_frames.setdefault(skip, set()).add(frame)
                            if not self.memory.search(skip):
                                self.memory.append(skip)
                    if chunks_in_sent == None:
                        break

            doc_covered_all += sent_covered

        in_count = len(self.memory) - l_memory_size

        self.batch_update_memory(reward_list, used_chunks)

        covered_rate, chunk_groups = self.eval_memorizer(doc_covered_all)
        chunk_in_use = sum([g_len if key==0 else 1 for key, g_len in chunk_groups])

        mem_usage = len(set(used_chunks.keys()) & set(self.memory)) / len(self.memory)

        return covered_rate, len(doc_covered_all)/chunk_in_use, mem_usage, in_count

    def run(self, epochs: int, corpus_train: Corpus, corpus_test: Corpus):
        for epoch_id in range(epochs):
            self.training_step(epoch_id, corpus_train, corpus_test)

    def training_step(self, epoch_id: int, corpus_train: Corpus, corpus_test: Corpus):
        """
        Samples one document from the train set, does a training step, then potentially evaluates.
        """
        document = corpus_train.sample()
        covered_rate, avg_chunk_len, mem_usage, in_count = self.reading(document)

        if epoch_id % self._eval_interval == 0:  # Every 100 epochs, you run validation-set statistics.
            self.eval_step(corpus_test, latest_epoch=epoch_id, latest_covered_rate=covered_rate, latest_mem_usage=mem_usage)

    def eval_step(self, corpus_test: Corpus,
                  latest_epoch: int, latest_covered_rate, latest_mem_usage):
            @dataclass
            class ValidationStats:
                mem_length: list[int]
                covered: list[float]
                mem_usage: list[float]

                pr1: list[float]
                re1: list[float]
                pr2: list[float]
                re2: list[float]

            stats = ValidationStats([], [], [], [], [], [], [])
            stats.mem_length.append(len(self.memory))
            stats.covered.append(latest_covered_rate)
            stats.mem_usage.append(latest_mem_usage)

            for document in corpus_test.documents:
                chunk_pos_0 = set(accumulate([len(pretoken) for sentence in document.sentences for pretoken in sentence.words]))

                chunk_pos = self.segmenter.segment(document, decompose=True)
                precision_1 = len(chunk_pos_0&chunk_pos)/len(chunk_pos)
                recall_1 = len(chunk_pos_0&chunk_pos)/len(chunk_pos_0)

                reference_chunks = [sentence.words for sentence in document.sentences]
                chunks = self.segmenter.segment(document, return_chunks=True, comb_sents=False, decompose=True)
                precision_2, recall_2, f1_2 = self.get_micro_f1(reference_chunks, chunks)

                stats.pr1.append(precision_1)
                stats.re1.append(recall_1)
                stats.pr2.append(precision_2)
                stats.re2.append(recall_2)

            precision_1, recall_1, precision_2, recall_2 = float(np.mean(stats.pr1)), float(np.mean(stats.re1)), float(np.mean(stats.pr2)), float(np.mean(stats.re2))
            self.memory_log.append([time.time(), 2*precision_2*recall_2/(precision_2+recall_2),len(self.memory)])

            print(f'{latest_epoch}\t  MemLength: {int(np.mean(stats.mem_length))}')
            print(f'[B] Precision: {precision_1*100:.2f}% \t Recall: {recall_1*100:.2f}% \t F1: {2*precision_1*recall_1/(precision_1+recall_1)*100:.2f}%')
            print(f'[L] Precision: {precision_2*100:.2f}% \t Recall: {recall_2*100:.2f}% \t F1: {2*precision_2*recall_2/(precision_2+recall_2)*100:.2f}%')
            print()

    def get_micro_f1(self, references: Sequence[Sequence[str]], candidates: Sequence[Sequence[str]]) -> tuple[float, float, float]:
        n_references = n_candidates = n_correct = 0
        for reference_tokens, candidate_tokens in zip(references, candidates):
            n_references += len(reference_tokens)
            n_candidates += len(candidate_tokens)
            gold_chunk_set = set(reference_tokens)
            n_correct += sum(token in gold_chunk_set
                             for token in candidate_tokens)
        pre = n_correct / n_candidates
        rec = n_correct / n_references
        f1 = 2 * pre * rec / (pre + rec)
        return pre, rec, f1
