from collections import Counter
from itertools import accumulate,groupby
import random
import numpy as np
import time

from .structures import TrieList
# memory = TrieList()


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
        self.memory_log = []

    def initialise(self, memory, corpus_samples):
        self.to_dropout.clear()
        for i in set([i for j in corpus_samples for i in j]):
            memory.append(i)
            self.to_dropout[i] = self._life

    def eval_memorizer(self, doc_covered):
        return (1-sum([i==0 for i in doc_covered])/len(doc_covered),
                [(key, len(list(group))) for key, group in groupby(doc_covered)])

    def dropout(self, memory, to_test, doc, doc_covered, doc_loc, reward_list, strict=False):
        for chunk_to_test in list(to_test.keys()):
            if strict and not memory.search(chunk_to_test):
                del to_test[chunk_to_test]
                continue

            onset, end, N_unknowns, N_chunks, chunks_list_0, chunks_list = to_test[chunk_to_test]
            while end < doc_loc:
                chunk_next = memory.match(doc[end:end+self._max_len])
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
                              ((N_chunks_0 > N_chunks) or \
                               (N_chunks_0 == N_chunks and \
                                    sum(memory.index_with_prior(i) for i in chunks_list_0 if i in memory) > sum(memory.index_with_prior(i) for i in chunks_list if i in memory)))

                if N_unknowns_0 > N_unknowns or redundant:
                    smaller_chunk = memory.match(chunk_to_test[:-1])
                    reward_list.append((chunk_to_test, 1))
                    reward_list.append((smaller_chunk, -1))
                else:
                    reward_list.append((chunk_to_test, -1))

                    if chunk_to_test in self.to_dropout:
                        del self.to_dropout[chunk_to_test]
            else:
                to_test[chunk_to_test] = onset, end, N_unknowns, N_chunks, chunks_list_0, chunks_list

    def batch_update_memory(self, memory, reward_list, used_chunks):
        to_del = []
        for chunk_to_test in list(self.to_dropout.keys()):
            current_life = self.to_dropout[chunk_to_test]
            if current_life <= 1:
                del self.to_dropout[chunk_to_test]
                to_del.append(chunk_to_test)
            else:
                self.to_dropout[chunk_to_test] = current_life - 1

        for c, label in reward_list:
            if c in memory.relationship:
    #             print(lable, c,memory.relationship[c])
                reward_list.append((memory.relationship[c], label))

        memory.group_remove(to_del)

        reward_list = [(w,e) for w,e in reward_list if w not in set(to_del)]
        memory.group_move(reward_list, self._update_rate)

        forget_samples = range(int((1 - self._memory_out) * len(memory)), len(memory))

        for ind in forget_samples:
            chunk_t = memory[ind]
            if chunk_t not in self.to_dropout:
                self.to_dropout[chunk_t] = self._life

    def memorize(self, memory, to_memorize, to_get, reward_list):
        if to_get and not memory.search(to_get):
            if to_get in to_memorize:
                to_memorize.remove(to_get)
                memory.append(to_get)
            else:
                to_memorize.add(to_get)

    def reading(self, memory, article):
        used_chunks = Counter()

        doc_covered_all = []
        reward_list = []
        punish_list= []
        to_memorize = set()
        l_memory_size = len(memory)
        new_memorized = Counter()

        for sent in article:
            chunks_in_sent = []
            singles = []
            to_test = dict()
            last_chunk = ['', 0] # 0 unknown, 1 known

            sent_covered = [0] * len(sent)
            i = 0
            while i < len(sent):
                self.dropout(memory, to_test, sent, sent_covered, i, reward_list)

                chunk_2nd, chunk = memory.match_two(sent[i: i+self._max_len])

                for chunk_to_test in list(to_test.keys()):
                    to_test[chunk_to_test][4].append(chunk)

                if len(chunk) > 0:
                    if (len(last_chunk) + len(chunk) <= self._max_len) and (random.random() < self._memory_in):
                        to_get = None
                        if last_chunk[1] == 0:
                            if 0 < len(last_chunk[0]) <= self._mini_gap:
                                to_get = last_chunk[0]
                        elif len(last_chunk[0] + chunk) <= self._max_len:
                            to_get = last_chunk[0] + chunk
                        self.memorize(memory, to_memorize, to_get, reward_list)

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
                    last_chunk[0] += sent[i]
                    chunks_in_sent.append(sent[i])
                    i += 1

            if self._use_skip:
                chunks_in_sent = ['[bos]'] + chunks_in_sent + ['[eos]']
                for a,b in zip(chunks_in_sent[:-2], chunks_in_sent[2:]):
                    if random.random() < (self._memory_in * self._memory_in):
                        self.memorize(memory, to_memorize, ('skipgram', a,b), reward_list)

                while True:
                    skip_gram, skip, chunks_in_sent = memory.skipgram_match(chunks_in_sent)
                    if skip is not None and len(skip) > 1:
                        skip = ''.join(skip)
    #                     if (len(skip) < 8):print(skip)
                        if (len(skip) <= self._mini_gap) and (random.random() < self._memory_in):
        #                     memorize(memory, to_memorize, skip, reward_list)
                            if not memory.search(skip):
                                memory.relationship[skip] = ('skipgram', *skip_gram)
                                memory.append(skip)
                    if chunks_in_sent == None:
                        break

            doc_covered_all += sent_covered

        in_count = len(memory) - l_memory_size

        self.batch_update_memory(memory, reward_list, used_chunks)

        covered_rate, chunk_groups = self.eval_memorizer(doc_covered_all)
        chunk_in_use = sum([g_len if key==0 else 1 for key, g_len in chunk_groups])

        mem_usage = len(set(used_chunks.keys()) & set(memory))/len(memory)

        return covered_rate, len(doc_covered_all)/chunk_in_use, mem_usage, in_count

    def record_scores(self, scores_hist, scores):
        for k,v in scores.items():
            if k not in scores_hist:
                scores_hist[k] = []

            scores_hist[k].append(v)

    def output_scores(self, scores_hist, items):
        return [np.mean(scores_hist[k]) for k in items]

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

    def run(self, epoch_id, memory, corpus_train, corpus_test):
        article_whole, article_raw_whole = random.choice(corpus_train)
        article, article_raw = article_whole, article_raw_whole

        covered_rate, avg_chunk_len, mem_usage, in_count = self.reading(memory, article)

    #     memory_log.append([time.time(), memory.par_list.copy()])

        if epoch_id % 100 == 0:
            scores_hist = dict()
            self.record_scores(scores_hist,
                      {'MemLength':len(memory),
                       'Covered':covered_rate,
                       'MemUsage':mem_usage})

            for article, article_raw in corpus_test:
                chunk_pos_0 = set(accumulate([len(c) for sent in article_raw for c in sent]))

                chunk_pos = self.show_reading(memory, article, decompose=True)
                precision_1 = len(chunk_pos_0&chunk_pos)/len(chunk_pos)
                recall_1 = len(chunk_pos_0&chunk_pos)/len(chunk_pos_0)

                chunks_0 = [[c for c in sent] for sent in article_raw]

                chunks = self.show_reading(memory, article, return_chunks=True, comb_sents=False, decompose=True)
                precision_2, recall_2, f1_2 = self.get_f1(chunks_0, chunks)

                self.record_scores(scores_hist,
                              {'precision_1':precision_1,
                               'recall_1':recall_1,
                               'precision_2':precision_2,
                               'recall_2':recall_2,})

            mem_length,covered,mem_usage,precision_1,recall_1,precision_2,recall_2 = self.output_scores(
                scores_hist,
                ['MemLength','Covered','MemUsage','precision_1','recall_1','precision_2','recall_2']
            )
            self.memory_log.append([time.time(), 2*precision_2*recall_2/(precision_2+recall_2),len(memory)])

            print(f'{epoch_id}\t  MemLength: {int(mem_length)}')
    #          B: {math.log(MemLength)/avg_chunk_len_1:.3f}
    #         print()
    #         print(f'Precision: {precision_0*100:.2f}% \t Recall: {recall_0*100:.2f}% \t F1: {2*precision_0*recall_0/(precision_0+recall_0)*100:.2f}%')

    #         print(f'Chunk_len: {avg_chunk_len_1:.1f} \t Word_len: {avg_chunk_len_2:.1f} \t',end='')
            print(f'[B] Precision: {precision_1*100:.2f}% \t Recall: {recall_1*100:.2f}% \t F1: {2*precision_1*recall_1/(precision_1+recall_1)*100:.2f}%')
            print(f'[L] Precision: {precision_2*100:.2f}% \t Recall: {recall_2*100:.2f}% \t F1: {2*precision_2*recall_2/(precision_2+recall_2)*100:.2f}%')
            print()

    def find_subs(self, memory, large_chunk, level=2):
        chunk_1 = large_chunk

        subs = []
        while True:
            chunk_1 = memory.match(chunk_1[:-1])
            chunk_2 = large_chunk[len(chunk_1):]

            if chunk_1!='' and chunk_2 in memory:
                subs.append((chunk_1, chunk_2, (memory.index_with_prior(chunk_1), memory.index_with_prior(chunk_2))))
            if len(chunk_1) <= 1:
                break

        if len(subs) > 0:
            sub = sorted(subs, key=lambda x:x[2])[0]
            if max(sub[2]) < memory.index_with_prior(large_chunk, nothing=len(memory)):
                if level == 1:
                    return sub[:2]
                else:
                    return self.find_subs(memory, sub[0], level-1) + self.find_subs(memory, sub[1], level-1)
            else:
                return (large_chunk,)
        else:
            return (large_chunk,)

    def show_reading(self, memory, article, max_len=10, decompose=False, display=False, return_chunks=False, comb_sents=True):
        chunks = []

        for sent in article:
            sent_chunks = []
            i = 0
            while i < len(sent):
                chunk_2nd, chunk = memory.match_two(sent[i: i+max_len])
                if len(chunk) > 0:

                    if len(chunk) > 1 and len(chunk_2nd) > 0:
                        onset, end_0, end = i, i + len(chunk), i + len(chunk_2nd)
                        N_unknowns_0, N_chunks_0, N_unknowns, N_chunks =  0, 1, 0, 1
                        chunks_t_0 = [chunk]
                        chunks_t = [chunk_2nd]

                        while end_0 != end:
                            if end_0 > end:
                                next_chunk = memory.match(sent[end: end+max_len])
                                chunk_size_next = len(next_chunk)
                                if chunk_size_next > 0:
                                    end += chunk_size_next
                                    N_chunks += 1
                                    chunks_t.append(next_chunk)
                                else:
                                    end += 1
                                    N_unknowns += 1
                                    N_chunks += 1
                                    chunks_t.append(sent[end-1])

                            elif end_0 < end:
                                next_chunk = memory.match(sent[end_0: end_0+max_len])
                                chunk_size_next = len(next_chunk)
                                if chunk_size_next > 0:
                                    end_0 += chunk_size_next
                                    N_chunks_0 += 1
                                    chunks_t_0.append(next_chunk)

                                else:
                                    end_0 += 1
                                    N_unknowns_0 += 1
                                    N_chunks_0 += 1
                                    chunks_t_0.append(sent[end_0-1])

                        redundant = N_unknowns_0 == N_unknowns and \
                                  ((N_chunks_0 > N_chunks) or \
                                   (N_chunks_0 == N_chunks and \
                                        sum(memory.index_with_prior(i, nothing=len(memory)) for i in chunks_t_0) > \
                                        sum(memory.index_with_prior(i, nothing=len(memory)) for i in chunks_t)))

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
                    sent_chunks.append(sent[i-1])

            chunks.append(sent_chunks)

        if decompose:
            chunks = [[sub_c for c in sent for sub_c in self.find_subs(memory,c)] for sent in chunks]

        if comb_sents:
            chunks = [c for sent in chunks for c in sent]

        if display:
            if comb_sents:
                print(' '.join(chunks))
            else:
                for sent in chunks: print(' '.join(sent))

        if return_chunks:
            return chunks
        else:
            return set(accumulate([len(c) for c in chunks]))

    def show_reading(self, memory, article, max_len=10, decompose=False, display=False, return_chunks=False, comb_sents=True):
        chunks = []

        for sent in article:
            sent_chunks = []
            i = 0
            while i < len(sent):
                chunk_2nd, chunk = memory.match_two(sent[i: i+max_len])
                if len(chunk) > 0:

                    sent_chunks.append(chunk)
                    i += len(chunk)
                else:
                    i += 1
                    sent_chunks.append(sent[i-1])

            chunks.append(sent_chunks)

        if decompose:
            chunks = [[sub_c for c in sent for sub_c in self.find_subs(memory,c)] for sent in chunks]

        if comb_sents:
            chunks = [c for sent in chunks for c in sent]

        if display:
            if comb_sents:
                print(' '.join(chunks))
            else:
                for sent in chunks: print(' '.join(sent))

        if return_chunks:
            return chunks
        else:
            return set(accumulate([len(c) for c in chunks]))

    def show_result(self, memory, article_raw, article, decompose=False):
        chunk_pos = self.show_reading(memory, article, decompose=decompose)
        chunk_pos_0 = set(accumulate([len(c) for sent in article_raw for c in sent]))


        doc = ''.join(article)
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

    def demo(self, article_raw, article, memory, decompose=False, section=(0,-1)):
        onset, end = section
        count = 0
        for chunk_i in range(999):
            if count == len(''.join(article[:onset])):
                chunk_i_0 = chunk_i
            elif count == len(''.join(article[:end])):
                break

            count += len(article_raw[chunk_i])

        self.show_result(memory, article_raw[chunk_i_0: chunk_i], article[onset:end], decompose=decompose)

    def show_result_with_idx(self, memory, article_raw, article, decompose=False):
        chunk_pos = self.show_reading(memory, article, decompose=decompose)
        chunk_pos_0 = set(accumulate([len(c) for sent in article_raw for c in sent]))

        doc = ''.join(article)
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
                chunk_i_idx = memory.index_with_prior(chunk_i)
            except:
                chunk_i_idx = '___'
            try:
                chunk_j_idx = memory.index_with_prior(chunk_j)
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
