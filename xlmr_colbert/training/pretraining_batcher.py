import ujson

from functools import partial
from xlmr_colbert.utils.utils import print_message
from xlmr_colbert.modeling.tokenization import (
    QueryTokenizer,
    DocTokenizer,
    tensorize_triples,
    tensorize_queries_documents,
)

from xlmr_colbert.utils.runs import Run

import numpy as np


class PreTrainingBatcher:
    def __init__(self, args, rank=0, nranks=1):
        self.bsize, self.accumsteps = args.bsize, args.accumsteps

        self.query_tokenizer = QueryTokenizer(args.query_maxlen)
        self.doc_tokenizer = DocTokenizer(args.doc_maxlen)
        self.tensorize_triples = partial(
            tensorize_triples, self.query_tokenizer, self.doc_tokenizer
        )
        self.tensorize_quries = partial(tensorize_queries, self.query_tokenizer)
        self.position = 0

        self.queries_lang_a = self._load_queries(args.queries_lang_a)
        self.queries_lang_b = self._load_queries(args.queries_lang_b)
        self.collection_lang_a = self._load_collection(args.collection_lang_a)
        self.collection_lang_b = self._load_collection(args.collection_lang_b)

        self.rng = np.random.default_rng()

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
                pid, passage, title, *_ = line.strip().split("\t")
                assert pid == "id" or int(pid) == line_idx

                passage = title + " | " + passage
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

        queries_a, queries_b = [], []
        collections_a, collections_b = [], []

        for position in range(offset, endpos):
            query_a, query_b = (
                self.queries_lang_a[position],
                self.queries_lang_b[position],
            )
            collection_a, collection_b = (
                self.collection_lang_a[position],
                self.collection_lang_b[position],
            )

            queries_a.append(query_a)
            queries_b.append(query_b)
            collections_a.append(collection_a)
            collections_b.append(collection_b)

        return self.collate(queries_a, queries_b, collections_a, collections_b)

    def collate(self, queries_a, queries_b, collections_a, collections_b):
        assert (
            len(queries_a)
            == len(queries_b)
            == len(collections_a)
            == len(collections_b)
            == self.bsize
        )

        return self.tensorize_queries_documents(
            queries_a,
            queries_b,
            collections_a,
            collections_b,
            self.bsize // self.accumsteps,
        )

    def skip_to_batch(self, batch_idx, intended_batch_size):
        Run.warn(
            f"Skipping to batch #{batch_idx} (with intended_batch_size = {intended_batch_size}) for training."
        )
        self.position = intended_batch_size * batch_idx
