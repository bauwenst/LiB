from itertools import accumulate

from .structures import Document, TrieList


class TwoGreedySegmenter:
    """Two-greedy segmentation over a ranked lexicon (TrieList)."""

    def __init__(self, lexicon: TrieList):
        self.lexicon = lexicon

    def find_subs(self, large_chunk: str, level: int = 2) -> tuple[str, ...]:
        """
        Decompose a chunk into ranked subchunks (paper §4.1).

        Among splits of ``large_chunk`` into two lexicon types ``(c1, c2)``,
        choose the split with the lowest pair of ordinals. Recurse up to
        ``level`` if both halves outrank the original chunk; otherwise keep
        ``large_chunk`` intact. Used when ``decompose=True`` in ``segment`` /
        ``show_reading`` (e.g. word-like units inside multi-word chunks).

        Parameters
        ----------
        large_chunk :
            Chunk string to split.
        level :
            Recursion depth. ``1`` returns at most one binary split; the default
            ``2`` may split each half once more.

        Returns
        -------
        tuple of str
            Ordered subchunks, or ``(large_chunk,)`` if no better split exists.
        """
        chunk_1 = large_chunk

        subs = []
        while True:
            chunk_1 = self.lexicon.match(chunk_1[:-1])
            chunk_2 = large_chunk[len(chunk_1):]

            if chunk_1!='' and chunk_2 in self.lexicon:
                subs.append((chunk_1, chunk_2, (self.lexicon.index_with_prior(chunk_1), self.lexicon.index_with_prior(chunk_2))))
            if len(chunk_1) <= 1:
                break

        if len(subs) > 0:
            sub = sorted(subs, key=lambda x:x[2])[0]
            if max(sub[2]) < self.lexicon.index_with_prior(large_chunk, nothing=len(self.lexicon)):
                if level == 1:
                    return sub[:2]
                else:
                    return self.find_subs(sub[0], level-1) + self.find_subs(sub[1], level-1)
            else:
                return (large_chunk,)
        else:
            return (large_chunk,)

    def segment(
        self,
        article: Document,
        max_len: int = 10,
        decompose: bool = False,
        return_chunks: bool = False,
        comb_sents: bool = True,
    ):
        """
        Two-greedy segmentation of a document with the current ranked lexicon.

        At each position, take the longest and second-longest lexicon matches.
        If both exist, compare full continuations (token count, then sum of
        ordinals) and commit the better path — unlike training ``reading``, which
        always commits the primary match and uses the comparison only for rewards.

        Parameters
        ----------
        article :
            Document to segment.
        max_len :
            Maximum match window in characters (historical default ``10``,
            independent of the trainer's ``max_len``).
        decompose :
            If True, run ``find_subs`` on each committed chunk.
        return_chunks :
            If True, return chunk strings; if False, return boundary positions
            (cumulative character offsets) as a ``set``.
        comb_sents :
            If True, flatten sentence-level chunk lists into one sequence
            before returning.

        Returns
        -------
        list or set
            Chunk strings (nested by sentence unless ``comb_sents``), or a set of
            boundary offsets when ``return_chunks`` is False.
        """
        chunks = []

        for sentence in article.sentences:
            concatenated_sentence = sentence.cat()

            sent_chunks = []
            i = 0
            while i < len(concatenated_sentence):
                chunk_2nd, chunk = self.lexicon.match_two(concatenated_sentence[i:i+max_len])
                if len(chunk) > 0:

                    if len(chunk) > 1 and len(chunk_2nd) > 0:
                        onset, end_0, end = i, i + len(chunk), i + len(chunk_2nd)
                        N_unknowns_0, N_chunks_0, N_unknowns, N_chunks =  0, 1, 0, 1
                        chunks_t_0 = [chunk]
                        chunks_t = [chunk_2nd]

                        while end_0 != end:
                            if end_0 > end:
                                next_chunk = self.lexicon.match(concatenated_sentence[end: end+max_len])
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
                                next_chunk = self.lexicon.match(concatenated_sentence[end_0: end_0+max_len])
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
                                        sum(self.lexicon.index_with_prior(i, nothing=len(self.lexicon)) for i in chunks_t_0) >
                                        sum(self.lexicon.index_with_prior(i, nothing=len(self.lexicon)) for i in chunks_t)))

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

        if return_chunks:
            return chunks
        else:
            return set(accumulate([len(c) for c in chunks]))
