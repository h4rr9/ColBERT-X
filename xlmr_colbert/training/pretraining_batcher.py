import ujson

from functools import partial
from xlmr_colbert.utils.utils import print_message
from xlmr_colbert.modeling.tokenization import (
    QueryTokenizer,
    DocTokenizer,
    tensorize_queries_documents,
)

from xlmr_colbert.utils.runs import Run


class PreTrainingBatcher:
    def __init__(self, args, rank=0, nranks=1):
        self.bsize, self.accumsteps = args.bsize, args.accumsteps

        self.query_tokenizer = QueryTokenizer(args.query_maxlen)
        self.doc_tokenizer = DocTokenizer(args.doc_maxlen)
        self.tensorize_queries_documents = partial(
            tensorize_queries_documents, self.query_tokenizer, self.doc_tokenizer
        )
        self.position = 0

        self.query_triples = self._load_triples(args.query_triples, rank, nranks)
        self.collection_triples = self._load_triples(args.collection_triples, rank, nranks)
        self.queries_lang_a = self._load_queries(args.queries_lang_a)
        self.queries_lang_b = self._load_queries(args.queries_lang_b)
        self.collections_lang_a = self._load_collection(args.collection_lang_a)
        self.collections_lang_b = self._load_collection(args.collection_lang_b)

        assert len(self.query_triples) == len(
            self.collection_triples
        ), "Expected query and document triples to have same length"

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
            self.position + self.bsize, len(self.query_triples)
        )
        self.position = endpos

        if offset + self.bsize > len(self.triples):
            raise StopIteration

        queries, queries_positive, queries_negative = [], [], []
        collections, collections_positive, collections_negative = [], []

        for position in range(offset, endpos):
            query, query_pos, query_neg = self.query_triples[position]
            collection, collection_pos, collection_neg = self.collection_triples[
                position
            ]

            query, query_pos, query_neg = (
                self.queries_lang_a[query],
                self.queries_lang_b[query_pos],
                self.queries_lang_b[query_neg],
            )
            collection, collection_pos, collection_neg = (
                self.colletions_lang_a[collection],
                self.collections_lang_b[collection_pos],
                self.collections_lang_b[collection_neg],
            )

            queries.append(query)
            queries_positive.append(query_pos)
            queries_negative.append(query_neg)
            collections.append(collection)
            collections_positive.append(collection_pos)
            collections_negative.append(collection_neg)

        return self.collate(
            queries,
            queries_positive,
            queries_negative,
            collections,
            collections_positive,
            collections_negative,
        )

    def collate(
        self,
        queries,
        queries_positive,
        queries_negative,
        collections,
        collections_positive,
        collections_negative,
    ):
        assert (
            len(queries)
            == len(queries_positive)
            == len(queries_negative)
            == len(collections)
            == len(collections_positive)
            == len(collections_negative)
            == self.bsize
        )

        return self.tensorize_queries_documents(
            queries,
            queries_positive,
            queries_negative,
            collections,
            collections_positive,
            collections_negative,
            self.bsize // self.accumsteps,
        )

    def skip_to_batch(self, batch_idx, intended_batch_size):
        Run.warn(
            f"Skipping to batch #{batch_idx} (with intended_batch_size = {intended_batch_size}) for training."
        )
        self.position = intended_batch_size * batch_idx
