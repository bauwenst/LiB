from dataclasses import dataclass

from collections import Counter
from itertools import accumulate, groupby
import random
import numpy as np
import time

from .structures import TrieList, Corpus, Document


class LessIsBetter:

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
        self._life = life
        self._max_len = max_len
        self._memory_in = memory_in
        self._memory_out = memory_out
        self._update_rate = update_rate
        self._mini_gap = mini_gap
        self._use_skip = use_skip

        self.to_dropout = dict()
        self.memory = TrieList()
        self.memory_log = []

    def initialise(self, corpus_samples):
        self.to_dropout.clear()
        for i in set([i for j in corpus_samples for i in j]):
            self.memory.append(i)
            self.to_dropout[i] = self._life

    def eval_memorizer(self, doc_covered):
        return (1-sum([i==0 for i in doc_covered])/len(doc_covered),
                [(key, len(list(group))) for key, group in groupby(doc_covered)])

    def dropout(self, to_test, doc, doc_covered, doc_loc, reward_list, strict=False):
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

    def batch_update_memory(self, reward_list, used_chunks):
        to_del = []
        for chunk_to_test in list(self.to_dropout.keys()):
            current_life = self.to_dropout[chunk_to_test]
            if current_life <= 1:
                del self.to_dropout[chunk_to_test]
                to_del.append(chunk_to_test)
            else:
                self.to_dropout[chunk_to_test] = current_life - 1

        for c, label in reward_list:
            if c in self.memory.relationship:
    #             print(lable, c,memory.relationship[c])
                reward_list.append((self.memory.relationship[c], label))

        self.memory.group_remove(to_del)

        reward_list = [(w,e) for w,e in reward_list if w not in set(to_del)]
        self.memory.group_move(reward_list, self._update_rate)

        forget_samples = range(int((1 - self._memory_out) * len(self.memory)), len(self.memory))

        for ind in forget_samples:
            chunk_t = self.memory[ind]
            if chunk_t not in self.to_dropout:
                self.to_dropout[chunk_t] = self._life

    def memorize(self, to_memorize, to_get, reward_list):
        if to_get and not self.memory.search(to_get):
            if to_get in to_memorize:
                to_memorize.remove(to_get)
                self.memory.append(to_get)
            else:
                to_memorize.add(to_get)

    def reading(self, article: Document):
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
                chunks_in_sent = ['[bos]'] + chunks_in_sent + ['[eos]']
                for a,b in zip(chunks_in_sent[:-2], chunks_in_sent[2:]):
                    if random.random() < (self._memory_in * self._memory_in):
                        self.memorize(to_memorize, ('skipgram', a,b), reward_list)

                while True:
                    skip_gram, skip, chunks_in_sent = self.memory.skipgram_match(chunks_in_sent)
                    if skip is not None and len(skip) > 1:
                        skip = ''.join(skip)
    #                     if (len(skip) < 8):print(skip)
                        if (len(skip) <= self._mini_gap) and (random.random() < self._memory_in):
        #                     memorize(memory, to_memorize, skip, reward_list)
                            if not self.memory.search(skip):
                                self.memory.relationship[skip] = ('skipgram', *skip_gram)
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

    def get_f1(self, references, candidates):
        gold_chunk = est_chunk = correct_chunk = 0
        for gold_sentence, estimated_sentence in zip(references, candidates):
            gold_chunk += len(gold_sentence)
            est_chunk += len(estimated_sentence)
            gold_chunk_set = set(gold_sentence)
            correct_chunk += sum(im in gold_chunk_set for im in estimated_sentence)
        pre = correct_chunk / est_chunk
        rec = correct_chunk / gold_chunk
        f1 = 2 * pre * rec / (pre + rec)
        return pre, rec, f1

    def run(self, epochs: int, corpus_train: Corpus, corpus_test: Corpus):
        for epoch_id in range(epochs):
            self.training_step(epoch_id, corpus_train, corpus_test)

    def training_step(self, epoch_id: int, corpus_train: Corpus, corpus_test: Corpus):
        document = corpus_train.sample()
        covered_rate, avg_chunk_len, mem_usage, in_count = self.reading(document)
    #     memory_log.append([time.time(), memory.par_list.copy()])

        # Every 100 epochs, you run validation-set statistics.
        if epoch_id % 100 == 0:
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
            stats.covered.append(covered_rate)
            stats.mem_usage.append(mem_usage)

            for document in corpus_test.documents:
                chunk_pos_0 = set(accumulate([len(pretoken) for sentence in document.sentences for pretoken in sentence.words]))

                chunk_pos = self.show_reading(document, decompose=True)
                precision_1 = len(chunk_pos_0&chunk_pos)/len(chunk_pos)
                recall_1 = len(chunk_pos_0&chunk_pos)/len(chunk_pos_0)

                chunks_0 = [sentence.words for sentence in document.sentences]

                chunks = self.show_reading(document, return_chunks=True, comb_sents=False, decompose=True)
                precision_2, recall_2, f1_2 = self.get_f1(chunks_0, chunks)

                stats.pr1.append(precision_1)
                stats.re1.append(recall_1)
                stats.pr2.append(precision_2)
                stats.re2.append(recall_2)

            precision_1, recall_1, precision_2, recall_2 = float(np.mean(stats.pr1)), float(np.mean(stats.re1)), float(np.mean(stats.pr2)), float(np.mean(stats.re2))
            self.memory_log.append([time.time(), 2*precision_2*recall_2/(precision_2+recall_2),len(self.memory)])

            print(f'{epoch_id}\t  MemLength: {int(np.mean(stats.mem_length))}')
    #          B: {math.log(MemLength)/avg_chunk_len_1:.3f}
    #         print()
    #         print(f'Precision: {precision_0*100:.2f}% \t Recall: {recall_0*100:.2f}% \t F1: {2*precision_0*recall_0/(precision_0+recall_0)*100:.2f}%')
    #         print(f'Chunk_len: {avg_chunk_len_1:.1f} \t Word_len: {avg_chunk_len_2:.1f} \t',end='')
            print(f'[B] Precision: {precision_1*100:.2f}% \t Recall: {recall_1*100:.2f}% \t F1: {2*precision_1*recall_1/(precision_1+recall_1)*100:.2f}%')
            print(f'[L] Precision: {precision_2*100:.2f}% \t Recall: {recall_2*100:.2f}% \t F1: {2*precision_2*recall_2/(precision_2+recall_2)*100:.2f}%')
            print()

    def find_subs(self, large_chunk, level=2):
        chunk_1 = large_chunk

        subs = []
        while True:
            chunk_1 = self.memory.match(chunk_1[:-1])
            chunk_2 = large_chunk[len(chunk_1):]

            if chunk_1!='' and chunk_2 in self.memory:
                subs.append((chunk_1, chunk_2, (self.memory.index_with_prior(chunk_1), self.memory.index_with_prior(chunk_2))))
            if len(chunk_1) <= 1:
                break

        if len(subs) > 0:
            sub = sorted(subs, key=lambda x:x[2])[0]
            if max(sub[2]) < self.memory.index_with_prior(large_chunk, nothing=len(self.memory)):
                if level == 1:
                    return sub[:2]
                else:
                    return self.find_subs(sub[0], level-1) + self.find_subs(sub[1], level-1)
            else:
                return (large_chunk,)
        else:
            return (large_chunk,)

    def show_reading(self, article: Document, max_len=10, decompose=False, display=False, return_chunks=False, comb_sents=True):
        chunks = []

        for sentence in article.sentences:
            concatenated_sentence = sentence.cat()

            sent_chunks = []
            i = 0
            while i < len(concatenated_sentence):
                chunk_2nd, chunk = self.memory.match_two(concatenated_sentence[i:i+max_len])
                if len(chunk) > 0:

                    if len(chunk) > 1 and len(chunk_2nd) > 0:
                        onset, end_0, end = i, i + len(chunk), i + len(chunk_2nd)
                        N_unknowns_0, N_chunks_0, N_unknowns, N_chunks =  0, 1, 0, 1
                        chunks_t_0 = [chunk]
                        chunks_t = [chunk_2nd]

                        while end_0 != end:
                            if end_0 > end:
                                next_chunk = self.memory.match(concatenated_sentence[end: end+max_len])
                                chunk_size_next = len(next_chunk)
                                if chunk_size_next > 0:
                                    end += chunk_size_next
                                    N_chunks += 1
                                    chunks_t.append(next_chunk)
                                else:
                                    end += 1
                                    N_unknowns += 1
                                    N_chunks += 1
                                    chunks_t.append(concatenated_sentence[end-1])

                            elif end_0 < end:
                                next_chunk = self.memory.match(concatenated_sentence[end_0: end_0+max_len])
                                chunk_size_next = len(next_chunk)
                                if chunk_size_next > 0:
                                    end_0 += chunk_size_next
                                    N_chunks_0 += 1
                                    chunks_t_0.append(next_chunk)

                                else:
                                    end_0 += 1
                                    N_unknowns_0 += 1
                                    N_chunks_0 += 1
                                    chunks_t_0.append(concatenated_sentence[end_0-1])

                        redundant = N_unknowns_0 == N_unknowns and \
                                  ((N_chunks_0 > N_chunks) or
                                   (N_chunks_0 == N_chunks and
                                        sum(self.memory.index_with_prior(i, nothing=len(self.memory)) for i in chunks_t_0) >
                                        sum(self.memory.index_with_prior(i, nothing=len(self.memory)) for i in chunks_t)))

                        if N_unknowns_0 > N_unknowns or redundant:
                            sent_chunks += chunks_t
                            i += sum(len(c) for c in chunks_t)
                        else:
                            sent_chunks += chunks_t_0
                            i += sum(len(c) for c in chunks_t_0)

                    else:
                        sent_chunks.append(chunk)
                        i += len(chunk)
                else:
                    i += 1
                    sent_chunks.append(concatenated_sentence[i-1])

            chunks.append(sent_chunks)

        if decompose:
            chunks = [[sub_c for c in sent for sub_c in self.find_subs(c)] for sent in chunks]

        if comb_sents:
            chunks = [c for sent in chunks for c in sent]

        if display:
            if comb_sents:
                print(' '.join(chunks))
            else:
                for concatenated_sentence in chunks: print(' '.join(concatenated_sentence))

        if return_chunks:
            return chunks
        else:
            return set(accumulate([len(c) for c in chunks]))

    def show_reading(self, article: Document, max_len=10, decompose=False, display=False, return_chunks=False, comb_sents=True):  # FIXME [TB]: Another duplicate method implement.
        chunks = []

        for sentence in article.sentences:
            concatenated_sentence = sentence.cat()

            sent_chunks = []
            i = 0
            while i < len(concatenated_sentence):
                chunk_2nd, chunk = self.memory.match_two(concatenated_sentence[i: i+max_len])
                if len(chunk) > 0:

                    sent_chunks.append(chunk)
                    i += len(chunk)
                else:
                    i += 1
                    sent_chunks.append(concatenated_sentence[i-1])

            chunks.append(sent_chunks)

        if decompose:
            chunks = [[sub_c for c in sent for sub_c in self.find_subs(c)] for sent in chunks]

        if comb_sents:
            chunks = [c for sent in chunks for c in sent]

        if display:
            if comb_sents:
                print(' '.join(chunks))
            else:
                for concatenated_sentence in chunks: print(' '.join(concatenated_sentence))

        if return_chunks:
            return chunks
        else:
            return set(accumulate([len(c) for c in chunks]))

    def show_result(self, article: Document, decompose=False):
        chunk_pos = self.show_reading(article, decompose=decompose)
        chunk_pos_0 = set(accumulate([len(pretoken) for sentence in article.sentences for pretoken in sentence.words]))

        doc = ''.join([sentence.cat() for sentence in article.sentences])
        chunk_pos_0, chunk_pos = [0] + sorted(chunk_pos_0),[0] + sorted(chunk_pos)

        i, j = 0, 0
        li_0, li_1 = [], []
        while i < len(chunk_pos_0) and j < len(chunk_pos):
            if chunk_pos_0[i] < chunk_pos[j]:
                li_0.append(doc[chunk_pos_0[i-1]: chunk_pos_0[i]])
                i += 1
            elif chunk_pos_0[i] > chunk_pos[j]:
                li_1.append(doc[chunk_pos[j-1]: chunk_pos[j]])
                j += 1
            else:
                li_0.append(doc[chunk_pos_0[i-1]: chunk_pos_0[i]])
                li_1.append(doc[chunk_pos[j-1]: chunk_pos[j]])

                li_0.append('\t')
                li_1.append('\t')
                if len(li_0) > 12 and len(li_1) > 12:
                    print(' '.join(li_0))
                    print(' '.join(li_1))
                    print()
                    li_0, li_1 = [], []
                i += 1
                j += 1

    def demo(self, article: Document, decompose=False, section=(0,-1)):
        start_sentence_idx, end_sentence_idx = section
        start_sentence_idx_bis, end_sentence_idx_bis = -1, 0
        count = 0
        for chunk_i in range(999):
            if count == len(''.join([sentence.cat() for sentence in article.sentences[:start_sentence_idx]])):  # TODO: This comparison makes no sense. Count is in units of pretokens. The join is counting characters.
                start_sentence_idx_bis = chunk_i
            elif count == len(''.join([sentence.cat() for sentence in article.sentences[:end_sentence_idx]])):
                end_sentence_idx_bis = chunk_i
                break

            count += len(article.sentences[chunk_i].words)

        self.show_result(
            article_raw[start_sentence_idx_bis:end_sentence_idx_bis], article[start_sentence_idx:end_sentence_idx],
            decompose=decompose
        )  # TODO This one is difficult to refactor because supposedly you want to get different sets of sentences from the pretokenised vs. concatenated form of the corpus?

    def show_result_with_idx(self, article: Document, decompose=False):
        chunk_pos = self.show_reading(article, decompose=decompose)
        chunk_pos_0 = set(accumulate([len(pretoken) for sentence in article.sentences for pretoken in sentence.words]))

        doc = ''.join(sentence.cat() for sentence in article.sentences)
        chunk_pos_0, chunk_pos = [0] + sorted(chunk_pos_0), [0] + sorted(chunk_pos)

        i, j = 1, 1
        li_0, li_1 = [], []
        while i < len(chunk_pos_0) and j < len(chunk_pos):
            if len(li_0) > 15 and len(li_1) > 15:
                print(' '.join(li_0))
                print(' '.join(li_1))
                print()
                li_0, li_1 = [], []

            chunk_i = doc[chunk_pos_0[i-1]: chunk_pos_0[i]]
            chunk_j = doc[chunk_pos[j-1]: chunk_pos[j]]
            try:
                chunk_i_idx = self.memory.index_with_prior(chunk_i)
            except:
                chunk_i_idx = '___'
            try:
                chunk_j_idx = self.memory.index_with_prior(chunk_j)
            except:
                chunk_j_idx = '___'
            if chunk_pos_0[i] < chunk_pos[j]:
                li_0.append(chunk_i)
                li_0.append(f'_{chunk_i_idx}')
                i += 1
                li_0.append(' ')
            elif chunk_pos_0[i] > chunk_pos[j]:
                li_1.append(chunk_j)
                li_1.append(f'_{chunk_j_idx}')
                j += 1
                li_1.append(' ')
            else:
                li_0.append(chunk_i)
                li_0.append(f'_{chunk_i_idx}')
                li_1.append(chunk_j)
                li_1.append(f'_{chunk_j_idx}')

                li_0.append(' ')
                li_1.append(' ')

                i += 1
                j += 1
