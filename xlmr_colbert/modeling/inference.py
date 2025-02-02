import torch

from xlmr_colbert.modeling.colbert import ColBERT
from xlmr_colbert.modeling.tokenization import QueryTokenizer, DocTokenizer
from xlmr_colbert.utils.amp import MixedPrecisionManager
from xlmr_colbert.parameters import DEVICE


from tqdm.auto import tqdm


class ModelInference:
    def __init__(self, colbert: ColBERT, amp=False):
        assert colbert.training is False

        self.colbert = colbert
        self.query_tokenizer = QueryTokenizer(colbert.query_maxlen)
        self.doc_tokenizer = DocTokenizer(colbert.doc_maxlen)

        self.amp_manager = MixedPrecisionManager(amp)

    def query(self, *args, to_cpu=False, **kw_args):
        with torch.no_grad():
            with self.amp_manager.context():
                Q = self.colbert.query(*args, **kw_args)
                return Q.cpu() if to_cpu else Q

    def doc(self, *args, to_cpu=False, **kw_args):
        with torch.no_grad():
            with self.amp_manager.context():
                D = self.colbert.doc(*args, **kw_args)
                return D.cpu() if to_cpu else D

    #    def queryFromText(self, queries, bsize=None, to_cpu=False):
    #        if bsize:
    #            batches = self.query_tokenizer.tensorize(queries, bsize=bsize)
    #            batches = [self.query(input_ids, attention_mask, to_cpu=to_cpu) for input_ids, attention_mask in batches]
    #            return torch.cat(batches)
    #
    #        input_ids, attention_mask = self.query_tokenizer.tensorize(queries)
    #        return self.query(input_ids, attention_mask)

    def queryFromText(self, queries, bsize=None, to_cpu=False, with_ids=False):
        if bsize:
            batches = self.query_tokenizer.tensorize(queries, bsize=bsize)
            batchesEmbs = [
                self.query(input_ids, attention_mask, to_cpu=to_cpu)
                for input_ids, attention_mask in batches
            ]
            if with_ids:
                return (
                    torch.cat(batchesEmbs),
                    torch.cat([ids for ids, _ in batches]),
                    torch.cat([masks for _, masks in batches]),
                )
            return torch.cat(batchesEmbs)

        input_ids, attention_mask = self.query_tokenizer.tensorize(queries)
        if with_ids:
            return (self.query(input_ids, attention_mask), input_ids, attention_mask)
        return self.query(input_ids, attention_mask)

    def docFromText(
        self, docs, bsize=None, keep_dims=True, to_cpu=False, with_ids=False
    ):
        if bsize:
            # print("docFromText on %d documents" % len(docs))
            batch_ids, reverse_indices = self.doc_tokenizer.tensorize(docs, bsize=bsize)
            # batch_ids contain batches; each batch is a 2-tuple, of which the left is
            # the ids of each document, and the right is the masks of each document
            # print("tokens doc 0: %d" % len(batch_ids[0][0][0]))
            # print("total tokens %d" % sum([len(d) for ids, mark in batch_ids for d in ids]))
            # batch_ids = [ input_ids for input_ids in batches]

            # print("batch_ids len=%d" % len(batch_ids))
            # print("reverse_indices.shape=" + str(reverse_indices.shape))

            batches = [
                self.doc(input_ids, attention_mask, keep_dims=keep_dims, to_cpu=to_cpu)
                for input_ids, attention_mask in tqdm(batch_ids)
            ]
            # print("batches len = %d " % len(batches))

            if keep_dims:
                D = _stack_3D_tensors(batches)
                if with_ids:
                    Dids = _stack_3D_tensors(batch_ids)
                    return D[reverse_indices], Dids
                return D[reverse_indices]
            # print(batches[0][0])
            D = [d for batch in batches for d in batch]
            # print("lenD = %d " % len(D))
            if with_ids:
                # the masking code assumes that args.mask_punctuation is false.
                assert len(self.colbert.skiplist) == 0

                D_i = [
                    d[(mask > 0) & (d != 1)]
                    for input_ids, attention_masks in batch_ids
                    for d, mask in zip(input_ids, attention_masks)
                ]
                # print("len D_i = %d" % len(D_i))
                left = [D[idx] for idx in reverse_indices.tolist()]
                right = [D_i[idx] for idx in reverse_indices.tolist()]
                return left, right
            return [D[idx] for idx in reverse_indices.tolist()]

        input_ids, attention_mask = self.doc_tokenizer.tensorize(docs)
        if with_ids:
            return self.doc(input_ids, attention_mask, keep_dims=keep_dims), input_ids
        return self.doc(input_ids, attention_mask, keep_dims=keep_dims)

    #    def docFromText(self, docs, bsize=None, keep_dims=True, to_cpu=False):
    #        if bsize:
    #            batches, reverse_indices = self.doc_tokenizer.tensorize(docs, bsize=bsize)
    #
    #            batches = [self.doc(input_ids, attention_mask, keep_dims=keep_dims, to_cpu=to_cpu)
    #                       for input_ids, attention_mask in batches]
    #
    #            if keep_dims:
    #                D = _stack_3D_tensors(batches)
    #                return D[reverse_indices]
    #
    #            D = [d for batch in batches for d in batch]
    #            return [D[idx] for idx in reverse_indices.tolist()]
    #
    #        input_ids, attention_mask = self.doc_tokenizer.tensorize(docs)
    #        return self.doc(input_ids, attention_mask, keep_dims=keep_dims)

    def score(self, Q, D, mask=None, lengths=None, explain=False):
        if lengths is not None:
            assert mask is None, "don't supply both mask and lengths"

            mask = torch.arange(D.size(1), device=DEVICE) + 1
            mask = mask.unsqueeze(0) <= lengths.to(DEVICE).unsqueeze(-1)

        scores = D @ Q
        scores = scores if mask is None else scores * mask.unsqueeze(-1)
        scores = scores.max(1)

        if explain:
            assert False, "TODO"

        return scores.values.sum(-1).cpu()


def _stack_3D_tensors(groups):
    bsize = sum([x.size(0) for x in groups])
    maxlen = max([x.size(1) for x in groups])
    hdim = groups[0].size(2)

    output = torch.zeros(
        bsize, maxlen, hdim, device=groups[0].device, dtype=groups[0].dtype
    )

    offset = 0
    for x in groups:
        endpos = offset + x.size(0)
        output[offset:endpos, : x.size(1)] = x
        offset = endpos

    return output
