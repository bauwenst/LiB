from dataclasses import dataclass
from typing import Counter as CounterType, Iterable, Optional, Sequence, Union

from collections import Counter
from itertools import accumulate, groupby
import random
import numpy as np
import time
import unicodedata

from .segmentation import TwoGreedySegmenter
from .structures import Bound, Chunk, Corpus, Document, SkipGram, TrieList, VocabEntry


@dataclass
class PathCompetition:
    onset: int
    end_so_far: int
    n_unknowns_on_2nd: int
    n_chunks_on_2nd: int
    chunks_on_1st: list[str]
    chunks_on_2nd: list[str]


@dataclass
class TrainingEpochStats:
    covered_rate: float          # Fraction of character positions covered by at least one known chunk.
    average_chunk_length: float  # Characters per effective chunk group (working-memory load proxy).
    memory_usage: float          # Fraction of lexicon types that appeared in this document's segmentation.
    in_count: int                # Net number of types appended to the lexicon during this pass (before forgetting removals in ``batch_update_memory``).


class LiBVocabulariser:
    """
    Online LiB lexicon trainer: builds a ranked vocabulary by reading documents.

    Additions from upstream:
        - Customisable segmentation refinement depth.
        - Customisable minimum occurrences.
        - Filter that forbids types from entering the lexicon that are mixes of letters/numbers/punctuation.
        - Corpus sampling controls (stochasticity, epochs)

    Also fixes multiple bugs that were present in upstream, including:
        - Whether an UNK string was lexicalised used to be prevented by adding to it a different string's length: https://github.com/ray306/LiB/blob/388248f7306492dcddd67425ea70185c23753986/model.py#L139
        - In that same check, the actual string's length was being miscomputed as the length of the tuple containing it: https://github.com/ray306/LiB/blob/388248f7306492dcddd67425ea70185c23753986/model.py#L139
        - Path competitions were not resolved if they existed at the end of a sentence.
        - In the segmenter, refinement used to stop earlier than necessary due to a check resulting in a false negative.
        - Soon: path competition wins were not used to segment a document, but rather the greedy segmentation was always used.
    """

    def __init__(
        self,
        probation_time: int = 10,
        max_len: int = 12,

        memory_in: float = 0.25,
        memory_out: float = 0.0001,
        minimum_occurrence: int = 2,
        update_rate: float = 0.2,
        mini_gap: int = 2,
        do_skipgrams: bool = False,
        segmentation_refinement: int = 2,

        n_epochs: int = 1,
        eval_interval: int = 100,
        do_sample_corpus: bool = False
    ):
        """
        Parameters
        ----------
        probation_time :
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
            or short unknown span) for memorisation. If it is sampled
            minimum_occurrence times, it is appended to the lexicon.
            Higher for larger vocabulary.
        memory_out :
            Forgetting ratio ``ω``. Each epoch, the last ``ω · |L|`` lexicon
            entries (at least the final slice of the ranked list) are placed on probation.
            Lower for larger vocabulary.
        minimum_occurrence :
            Amount of times a type needs to be sampled before adding it to the vocabulary.
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
        do_skipgrams :
            If True, enable the experimental skip-gram path: sample skip-gram
            slots and optionally memorise filled spans linked via
            ``memory.relationship`` (filled span → set of ``SkipGram`` frames).
            Enabling tends to grow the vocabulary (extra skip-gram types).
        """
        self._forget_time = probation_time
        self._max_len = max_len
        self._memory_in = memory_in
        self._memory_out = memory_out
        self._minimum_occurrence = minimum_occurrence
        self._update_rate = update_rate
        self._mini_gap = mini_gap
        self._use_skip = do_skipgrams

        self._n_epochs = n_epochs
        self._eval_interval = eval_interval
        self._do_sample_corpus = do_sample_corpus

        self.time_before_forget = dict()
        self.memory = TrieList()
        self.memory_log = []
        self.segmenter = TwoGreedySegmenter(self.memory, refinement_level=segmentation_refinement)

    def initialise(self, spaceless_examples: Iterable[str]):
        self.time_before_forget.clear()
        for atom in set(atom for example in spaceless_examples for atom in example):
            self.memory.append(atom)
            self.time_before_forget[atom] = self._forget_time

    def memorize(self, lexicon_candidates: Counter[VocabEntry], proposal: VocabEntry):
        if self.memory.search(proposal):
            return

        if proposal in lexicon_candidates:
            lexicon_candidates[proposal] -= 1
            if lexicon_candidates[proposal] == 0:
                lexicon_candidates.pop(proposal)
                self.memory.append(proposal)
        elif self._acceptableEntry(proposal):
            lexicon_candidates[proposal] = self._minimum_occurrence

    def _acceptableEntry(self, typ: VocabEntry) -> bool:
        if isinstance(typ, str):
            return self._acceptableString(typ)
        else:
            raise NotImplementedError()

    def _acceptableString(self, s: str) -> bool:  # This is where you would put a preprocessor. For now we just use character classes.
        s = unicodedata.normalize('NFC', s)  # Makes accents absorbed into letters.
        if not s:
            return False

        seen = 0
        for c in s:
            if c.isalpha():
                flag = 1
            elif c.isdigit():
                flag = 2
            # elif c.isspace():  # Probably you want to allow spaces to combine with everything.
            #     flag = 4
            else:
                flag = 8  # punct/symbol/other

            if seen and seen != flag:
                return False
            seen = flag

        return True

    def batch_update_memory(self, reward_list: list):
        types_to_remove = []
        for chunk_to_test in list(self.time_before_forget):
            current_life = self.time_before_forget[chunk_to_test]
            if current_life <= 1:
                del self.time_before_forget[chunk_to_test]
                types_to_remove.append(chunk_to_test)
            else:
                self.time_before_forget[chunk_to_test] = current_life - 1

        for c, label in reward_list:
            if c in self.memory.skipgram_frames:
                for frame in self.memory.skipgram_frames[c]:
                    reward_list.append((frame, label))

        self.memory.group_remove(types_to_remove)

        reward_list = [(w,e) for w,e in reward_list if w not in set(types_to_remove)]
        self.memory.group_move(reward_list, self._update_rate)

        for ind in range(int((1 - self._memory_out) * len(self.memory)), len(self.memory)):
            chunk_t = self.memory[ind]
            if chunk_t not in self.time_before_forget:
                self.time_before_forget[chunk_t] = self._forget_time

    def train_step(self, examples: Union[Document,Iterable[str]]) -> TrainingEpochStats:
        """
        One training pass over a document: segment, memorise, evaluate, update.

        Walks each sentence left to right with larger-first ``match_two``. The
        primary (longest) match is always committed as the training segmentation;
        when a second-longest alternative exists, ``resolve_competitions`` later compares the
        two continuations and emits Good/Bad rewards for lexicon re-ranking.
        Adjacent chunks (and short unknown spans) may be sampled into the
        lexicon; then ``batch_update_memory`` applies passive forgetting,
        re-ranking, and active deletion.

        Parameters
        ----------
        examples :
            Spaceless sentences.
        """
        if not self.memory:
            raise RuntimeError("You first need to initialise the vocabulariser.")

        token_counts: CounterType[VocabEntry] = Counter()
        M_chars_covered_doc: list[int] = []  # flat per-char coverage flags across the whole document (0 = uncovered)
        reward_list: list[tuple[VocabEntry, int]] = []  # (type, label) with label -1 = Good / promote, +1 = Bad / demote
        lexicon_candidates: Counter[VocabEntry] = Counter()  # proposal -> remaining sightings needed before append (two-strike gate)
        old_memory_size = len(self.memory)

        for concatenated_sentence in ((sentence.cat() for sentence in examples.sentences) if isinstance(examples, Document) else examples):
            tokens: list[Chunk] = []  # committed chunk strings for this sentence (incl. single chars when OOV)
            alternatives: dict[str, PathCompetition] = dict()
            last_chunk     = ''
            last_chunk_OOV = True

            M_chars_covered_here = [0] * len(concatenated_sentence)  # per-char coverage in this sentence; nonzero once a known chunk covers the position
            i = 0
            while i < len(concatenated_sentence):
                self.resolve_competitions(alternatives, concatenated_sentence, i, reward_list)

                second_longest_prefix, longest_prefix = self.memory.match_two(concatenated_sentence[i:i+self._max_len])

                # extend each open primary-path hypothesis with the chunk we are about to commit
                for alternative_paths in alternatives.values():
                    alternative_paths.chunks_on_1st.append(longest_prefix)

                # Did we match a known type at i?
                if longest_prefix:
                    if random.random() < self._memory_in:  # If yes, try to add new type to vocab.
                        if last_chunk_OOV:
                            if 0 < len(last_chunk) <= self._mini_gap:  # Is the OOV gap short enough to memorise?
                                self.memorize(lexicon_candidates, last_chunk)
                        else:
                            proposed_new_type = last_chunk + longest_prefix
                            if len(proposed_new_type) <= self._max_len:  # Is known+known concatenation short enough to memorise?
                                self.memorize(lexicon_candidates, proposed_new_type)

                    # Open a Good/Bad contest (longest vs second-longest), else maybe rescue from probation
                    if second_longest_prefix:
                        alternatives[longest_prefix] = PathCompetition(
                            onset=i,
                            end_so_far=i + len(second_longest_prefix),
                            n_unknowns_on_2nd=0,
                            n_chunks_on_2nd=1,
                            chunks_on_1st=[longest_prefix],
                            chunks_on_2nd=[second_longest_prefix]
                        )
                    elif longest_prefix in self.time_before_forget:
                        del self.time_before_forget[longest_prefix]

                    # always commit the longest match (training; segmenter may choose otherwise) TODO: This is probably bad.
                    n_chars = len(longest_prefix)
                    M_chars_covered_here[i:i+n_chars] = [M_chars_covered_here[i-1] + 1] * n_chars
                    i += n_chars
                    last_chunk     = longest_prefix
                    last_chunk_OOV = False
                    tokens.append(longest_prefix)
                    token_counts[longest_prefix] += 1
                else:  # No lexicon match: consume one character as OOV
                    if not last_chunk_OOV:
                        last_chunk = ''
                        last_chunk_OOV = True
                    last_chunk += concatenated_sentence[i]
                    tokens.append(concatenated_sentence[i])
                    i += 1
            # End of the sentence is reached.
            self.resolve_competitions(alternatives, concatenated_sentence, len(concatenated_sentence), reward_list)  # Fix that wasn't done originally.

            # optional skip-gram path (experimental; off by default)
            if self._use_skip:
                tokens = [Bound.BOS] + tokens + [Bound.EOS]
                for a, b in zip(tokens[:-2], tokens[2:]):
                    # decide: sample a SkipGram(left, right) frame?
                    if random.random() < (self._memory_in * self._memory_in):
                        self.memorize(lexicon_candidates, SkipGram(a, b))

                while True:
                    skip_gram, skip, tokens = self.memory.skipgram_match(tokens)
                    # decide: found a multi-chunk fill under an existing skip-gram frame?
                    if skip is not None and len(skip) > 1:
                        skip = ''.join(skip)
                        # decide: is the fill short enough and sampled for memorisation?
                        if len(skip) <= self._mini_gap and random.random() < self._memory_in:
                            frame = SkipGram(skip_gram[0], skip_gram[1])
                            self.memory.skipgram_frames.setdefault(skip, set()).add(frame)
                            # decide: fill not yet in lexicon?
                            if not self.memory.search(skip):
                                self.memory.append(skip)
                    if tokens == None:
                        break

            M_chars_covered_doc += M_chars_covered_here

        in_count = len(self.memory) - old_memory_size
        self.batch_update_memory(reward_list)

        chunk_groups = [(key, len(list(group))) for key, group in groupby(M_chars_covered_doc)]  # runs of equal coverage ids; each known-chunk run counts as 1 group, each OOV char as its own
        chunk_in_use = sum([g_len if key==0 else 1 for key, g_len in chunk_groups])
        return TrainingEpochStats(
            covered_rate=1 - sum(i==0 for i in M_chars_covered_doc) / len(M_chars_covered_doc),
            average_chunk_length=len(M_chars_covered_doc) / chunk_in_use,
            memory_usage=len(set(token_counts) & set(self.memory)) / len(self.memory),
            in_count=in_count
        )

    def resolve_competitions(self, to_test: dict[str,PathCompetition], text: str, doc_index: int, reward_list: list, strict: bool=False):
        for chunk_to_test in list(to_test):
            if strict and not self.memory.search(chunk_to_test):
                del to_test[chunk_to_test]
                continue

            paths = to_test[chunk_to_test]
            while paths.end_so_far < doc_index:
                chunk_next = self.memory.match(text[paths.end_so_far:paths.end_so_far + self._max_len])
                chunk_size_next = len(chunk_next)
                if chunk_size_next > 0:
                    paths.end_so_far += chunk_size_next
                else:
                    paths.end_so_far += 1
                    paths.n_unknowns_on_2nd += 1

                paths.n_chunks_on_2nd += 1
                paths.chunks_on_2nd.append(chunk_next)

            if paths.end_so_far == doc_index:
                N_unknowns_1 = sum(len(c) == 0 for c in paths.chunks_on_1st)
                N_chunks_1   = sum(len(c)  > 0 for c in paths.chunks_on_1st)
                N_unknowns_2, N_chunks_2 = paths.n_unknowns_on_2nd, paths.n_chunks_on_2nd

                alternative_wins = False
                if N_unknowns_1 > N_unknowns_2:
                    alternative_wins = True
                elif N_unknowns_1 == N_unknowns_2:
                    if N_chunks_1 > N_chunks_2:
                        alternative_wins = True
                    elif N_chunks_1 == N_chunks_2:
                        alternative_wins = sum(self.memory.index_with_prior(i) for i in paths.chunks_on_1st if i in self.memory) \
                                         > sum(self.memory.index_with_prior(i) for i in paths.chunks_on_2nd if i in self.memory)

                if alternative_wins:
                    smaller_chunk = self.memory.match(chunk_to_test[:-1])
                    reward_list.append((chunk_to_test, +1))  # + is bad, - is good.
                    reward_list.append((smaller_chunk, -1))
                else:
                    reward_list.append((chunk_to_test, -1))

                    if chunk_to_test in self.time_before_forget:
                        del self.time_before_forget[chunk_to_test]

                del to_test[chunk_to_test]

    ####################################################################################################################

    def run(self, corpus_train: Corpus, corpus_test: Corpus):
        n_documents = len(corpus_train.documents)
        for epoch_id in range(self._n_epochs):
            for i in range(n_documents):
                document = corpus_train.documents[i] if not self._do_sample_corpus else corpus_train.sample()

                results = self.train_step(document)
                if i % self._eval_interval == 0:  # Every 100 steps, you run validation-set statistics.
                    self.eval_epoch(corpus_test, latest_epoch=epoch_id + i/n_documents, latest_train_results=results)

    def eval_epoch(self, corpus_test: Corpus,
                   latest_epoch: float, latest_train_results: TrainingEpochStats):
            @dataclass
            class ValidationStats:
                mem_length: list[int]
                covered: list[float]
                mem_usage: list[float]

                pr1: list[float]
                re1: list[float]
                pr2: list[float]
                re2: list[float]

            # TODO: Makes very little sense to have lists for these but never actually append more than one measurement.
            stats = ValidationStats([], [], [], [], [], [], [])
            stats.mem_length.append(len(self.memory))
            stats.covered.append(latest_train_results.covered_rate)
            stats.mem_usage.append(latest_train_results.memory_usage)

            for document in corpus_test.documents:
                chunk_pos_0 = set(accumulate([len(pretoken) for sentence in document.sentences for pretoken in sentence.words]))

                chunk_pos = self.segmenter.segment(document)
                precision_1 = len(chunk_pos_0&chunk_pos)/len(chunk_pos)
                recall_1 = len(chunk_pos_0&chunk_pos)/len(chunk_pos_0)

                reference_chunks = [sentence.words for sentence in document.sentences]
                chunks = self.segmenter.segment(document, return_chunks=True, flatten=False)
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
