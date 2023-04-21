import torch


def tensorize_triples(
    query_tokenizer, doc_tokenizer, queries, positives, negatives, bsize
):
    assert len(queries) == len(positives) == len(negatives)
    assert bsize is None or len(queries) % bsize == 0

    N = len(queries)
    Q_ids, Q_mask = query_tokenizer.tensorize(queries)
    D_ids, D_mask = doc_tokenizer.tensorize(positives + negatives)
    D_ids, D_mask = D_ids.view(2, N, -1), D_mask.view(2, N, -1)

    # Compute max among {length of i^th positive, length of i^th negative} for i \in N
    maxlens = D_mask.sum(-1).max(0).values

    # Sort by maxlens
    indices = maxlens.sort().indices
    Q_ids, Q_mask = Q_ids[indices], Q_mask[indices]
    D_ids, D_mask = D_ids[:, indices], D_mask[:, indices]

    (positive_ids, negative_ids), (positive_mask, negative_mask) = D_ids, D_mask

    query_batches = _split_into_batches(Q_ids, Q_mask, bsize)
    positive_batches = _split_into_batches(positive_ids, positive_mask, bsize)
    negative_batches = _split_into_batches(negative_ids, negative_mask, bsize)

    batches = []
    for (q_ids, q_mask), (p_ids, p_mask), (n_ids, n_mask) in zip(
        query_batches, positive_batches, negative_batches
    ):
        Q = (torch.cat((q_ids, q_ids)), torch.cat((q_mask, q_mask)))
        D = (torch.cat((p_ids, n_ids)), torch.cat((p_mask, n_mask)))
        batches.append((Q, D))

    return batches


def tensorize_queries_documents(
    query_tokenizer,
    doc_tokenizer,
    queries,
    queries_positive,
    queries_negative,
    documents,
    documents_positive,
    documents_negative,
    bsize,
):
    # TODO: implement query document tensorize
    assert (
        len(queries)
        == len(queries_positive)
        == len(queries_negative)
        == len(documents)
        == len(documents_positive)
        == len(documents_negative)
    )
    assert bsize is None or len(queries) % bsize == 0

    N = len(queries)

    Q_ids, Q_mask = query_tokenizer.tensorize(queries)
    Qpn_ids, Qpn_mask = query_tokenizer.tensorize(queries_positive + queries_negative)
    Qpn_ids, Qpn_mask = Qpn_ids.view(2, N, -1), Qpn_mask.view(2, N, -1)



    maxlens = Qpn_mask.sum(-1).max(0).values
    indices = maxlens.sort().indices
    Q_ids, Q_mask = Q_ids[indices], Q_mask[indices]
    Qpn_ids, Qpn_mask = Qpn_ids[:, indices], Qpn_mask[:, indices]




    (positive_ids, negative_ids), (positive_mask, negative_mask) = Qpn_ids, Qpn_mask



    query_batches = _split_into_batches(Q_ids, Q_mask, bsize)
    query_positive_batches = _split_into_batches(positive_ids, positive_mask, bsize)
    query_negative_batches = _split_into_batches(negative_ids, negative_mask, bsize)


    print(f"query_batches {len(query_batches)}")
    print(f"query_positive_batches {len(query_positive_batches)}")
    print(f"query_negative_batches {len(query_negative_batches)}")

    query_batches = []
    for (q_ids, q_mask), (qp_ids, qp_mask), (qn_ids, qn_mask) in zip(
        query_batches, query_positive_batches, query_negative_batches
    ):
        Q = (torch.cat((q_ids, q_ids)), torch.cat((q_mask, q_mask)))
        Qpn = (torch.cat((qp_ids, qn_ids)), torch.cat((qp_mask, qn_mask)))
        query_batches.append((Q, Qpn))
    print(f"query_batches {len(query_batches)}")

    D_ids, D_mask = doc_tokenizer.tensorize(documents)
    Dpn_ids, Dpn_mask = doc_tokenizer.tensorize(documents_positive + documents_negative)
    Dpn_ids, Dpn_mask = Dpn_ids.view(2, N, -1), Dpn_mask.view(2, N, -1)

    maxlens = Dpn_mask.sum(-1).max(0).values
    indices = maxlens.sort().indices
    D_ids, D_mask = D_ids[indices], D_mask[indices]
    Dpn_ids, Dpn_mask = Dpn_ids[:, indices], Dpn_mask[:, indices]

    (positive_ids, negative_ids), (positive_mask, negative_mask) = Dpn_ids, Dpn_mask

    doc_batches = _split_into_batches(D_ids, D_mask, bsize)
    doc_positive_batches = _split_into_batches(positive_ids, positive_mask, bsize)
    doc_negative_batches = _split_into_batches(negative_ids, negative_mask, bsize)

    doc_batches = []
    for (d_ids, d_mask), (dp_ids, dp_mask), (dn_ids, dn_mask) in zip(
        doc_batches, doc_positive_batches, doc_negative_batches
    ):
        D = (torch.cat((d_ids, d_ids)), torch.cat((d_mask, d_mask)))
        Dpn = (torch.cat((dp_ids, dn_ids)), torch.cat((dp_mask, dn_mask)))
        doc_batches.append((D, Dpn))
    print(f"doc_batches {len(doc_batches)}")


    return zip(query_batches, doc_batches)


def _sort_by_length(ids, mask, bsize):
    if ids.size(0) <= bsize:
        return ids, mask, torch.arange(ids.size(0))

    indices = mask.sum(-1).sort().indices
    reverse_indices = indices.sort().indices

    return ids[indices], mask[indices], reverse_indices


def _split_into_batches(ids, mask, bsize):
    batches = []
    for offset in range(0, ids.size(0), bsize):
        batches.append((ids[offset : offset + bsize], mask[offset : offset + bsize]))

    return batches
