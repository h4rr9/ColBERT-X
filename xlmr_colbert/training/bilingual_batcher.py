import ujson

from functools import partial
from xlmr_colbert.utils.utils import print_message
from xlmr_colbert.modeling.tokenization import (
    QueryTokenizer,
    DocTokenizer,
    tensorize_triples,
)

from xlmr_colbert.utils.runs import Run

import numpy as np


class BilingualBatcher:
    def __init__(self, args, rank=0, nranks=1):
        self.bsize, self.accumsteps = args.bsize, args.accumsteps

        self.query_tokenizer = QueryTokenizer(args.query_maxlen)
        self.doc_tokenizer = DocTokenizer(args.doc_maxlen)
        self.tensorize_triples = partial(
            tensorize_triples, self.query_tokenizer, self.doc_tokenizer
        )
        self.position = 0

        self.triples = self._load_triples(args.triples, rank, nranks)
        self.queries_lang_a = self._load_queries(args.queries_lang_a)
        self.queries_lang_b = self._load_queries(args.queries_lang_b)
        self.collection_lang_a = self._load_collection(args.collection_lang_a)
        self.collection_lang_b = self._load_collection(args.collection_lang_b)

        self.rng = np.random.default_rng()

    def _load_triples(self, path, rank, nranks):
        """
        NOTE: For distributed sampling, this isn't equivalent to perfectly uniform sampling.
        In particular, each subset is perfectly represented in every batch! However, since we never
        repeat passes over the data, we never repeat any particular triple, and the split across
        nodes is random (since the underlying file is pre-shuffled), there's no concern here.
        """
        print_message(f"#> Loading triples from {path}...")

        triples = []

        with open(path) as f:
            for line_idx, line in enumerate(f):
                if line_idx % nranks == rank:
                    qid, pos, neg = [int(id) for id in line.strip().split("\t")]
                    triples.append((qid, pos, neg))

        return triples

    def _load_queries(self, path):
        print_message(f"#> Loading queries from {path}...")

        queries = {}

        with open(path) as f:
            for line in f:
                qid, query = line.strip().split("\t")
                qid = int(qid)
                queries[qid] = query

        return queries

    def _load_collection(self, path):
        print_message(f"#> Loading collection from {path}...")

        collection = []

        with open(path) as f:
            for line_idx, line in enumerate(f):
                pid, passage = line.strip().split("\t")
                assert pid == "id" or int(pid) == line_idx

                collection.append(passage)

        return collection

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.triples)

    def __next__(self):
        offset, endpos = self.position, min(
            self.position + self.bsize, len(self.triples)
        )
        self.position = endpos

        if offset + self.bsize > len(self.triples):
            raise StopIteration

        queries, positives, negatives = [], [], []

        for position in range(offset, endpos):
            query, pos, neg = self.triples[position]

            query = (
                self.queries_lang_a[query]
                if self.rng.random() < 0.5
                else self.queries_lang_b[query]
            )

            pos = (
                self.collections_lang_a[query]
                if self.rng.random() < 0.5
                else self.collections_lang_b[query]
            )

            neg = (
                self.collections_lang_a[query]
                if self.rng.random() < 0.5
                else self.collections_lang_b[query]
            )

            query, pos, neg = (
                self.queries[query],
                self.collection[pos],
                self.collection[neg],
            )

            queries.append(query)
            positives.append(pos)
            negatives.append(neg)

        return self.collate(queries, positives, negatives)

    def collate(self, queries, positives, negatives):
        assert len(queries) == len(positives) == len(negatives) == self.bsize

        return self.tensorize_triples(
            queries, positives, negatives, self.bsize // self.accumsteps
        )

    def skip_to_batch(self, batch_idx, intended_batch_size):
        Run.warn(
            f"Skipping to batch #{batch_idx} (with intended_batch_size = {intended_batch_size}) for training."
        )
        self.position = intended_batch_size * batch_idx
