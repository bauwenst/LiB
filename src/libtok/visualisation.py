from itertools import accumulate
from typing import Optional

from .segmentation import TwoGreedySegmenter
from .structures import Document, TrieList


class Visualiser:

    def __init__(self, segmenter: TwoGreedySegmenter, lexicon: TrieList):
        self.segmenter = segmenter
        self.lexicon = lexicon

    def display_segmentation(self, chunks, comb_sents=True):
        if comb_sents:
            print(' '.join(chunks))
        else:
            for concatenated_sentence in chunks:
                print(' '.join(concatenated_sentence))

    def show_result(self, article: Document, decompose=False):
        chunk_pos = self.segmenter.segment(article, decompose=decompose)
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

    def demo(self, article: Document, decompose=False, section=(0,None)):
        start_sentence_idx, end_sentence_idx = section
        sliced = Document(sentences=article.sentences[start_sentence_idx:end_sentence_idx])
        self.show_result(sliced, decompose=decompose)

    def show_result_with_idx(self, article: Document, decompose=False):
        chunk_pos = self.segmenter.segment(article, decompose=decompose)
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
                chunk_i_idx = self.lexicon.index_with_prior(chunk_i)
            except:
                chunk_i_idx = '___'
            try:
                chunk_j_idx = self.lexicon.index_with_prior(chunk_j)
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

    def debug_lexicon(self, lexicon: Optional[TrieList] = None, mode='dict not root'):
        if lexicon is None:
            lexicon = self.lexicon
        if mode == 'dict not root':
            return [i for i in lexicon.par_dict if not lexicon.match(i)==i]
        elif mode == 'dict not list':
            return [i for i in lexicon.par_dict if i not in lexicon.par_list]
        elif mode == 'list not root':
            return [i for i in lexicon.par_list if not lexicon.match(i)==i]
        elif mode == 'list not dict':
            return [i for i in lexicon.par_list if i not in lexicon.par_dict]
        else:
            chunks = []
            curNode = lexicon.root

            def f(pre, curNode):
                for i in curNode:
                    if i!=-1:
                        f(pre+i, curNode[i])
                    else:
                        chunks.append(pre)
            f('', curNode)
            if mode == 'root not list':
                li = [i if isinstance(i,str) else ''.join(i) for i in lexicon.par_list]
            elif mode == 'root not dict':
                li = [i if isinstance(i,str) else ''.join(i) for i in lexicon.par_dict]
            return [i for i in chunks if i not in li]
