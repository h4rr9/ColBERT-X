import os
import time
import faiss
import random
import torch
import itertools

from xlmr_colbert.utils.runs import Run
from multiprocessing import Pool
from xlmr_colbert.modeling.inference import ModelInference
from xlmr_colbert.evaluation.ranking_logger import RankingLogger

from xlmr_colbert.utils.utils import print_message, batch
from xlmr_colbert.ranking.rankers import Ranker


def retrieve(args):
    inference = ModelInference(args.colbert, amp=args.amp)
    ranker = Ranker(args, inference, faiss_depth=args.faiss_depth)

    ranking_logger = RankingLogger(Run.path, qrels=None, log_scores=args.log_scores)
    milliseconds = 0

    with ranking_logger.context("ranking.tsv", also_save_annotations=False) as rlogger:
        queries = args.queries
        qids_in_order = list(queries.keys())

        for qoffset, qbatch in batch(qids_in_order, 100, provide_offset=True):
            qbatch_text = [queries[qid] for qid in qbatch]

            rankings = []

            for query_idx, q in enumerate(qbatch_text):
                torch.cuda.synchronize("cuda:0")
                s = time.time()

                Q = ranker.encode([q])
                pids, scores = ranker.rank(Q)

                torch.cuda.synchronize()
                milliseconds += (time.time() - s) * 1000.0

                if len(pids):
                    print(
                        qoffset + query_idx,
                        q,
                        len(scores),
                        len(pids),
                        scores[0],
                        pids[0],
                        milliseconds / (qoffset + query_idx + 1),
                        "ms",
                    )

                rankings.append(zip(pids, scores))

            for query_idx, (qid, ranking) in enumerate(zip(qbatch, rankings)):
                query_idx = qoffset + query_idx

                if query_idx % 100 == 0:
                    print_message(f"#> Logging query #{query_idx} (qid {qid}) now...")

                ranking = [
                    (score, pid, None)
                    for pid, score in itertools.islice(ranking, args.depth)
                ]
                rlogger.log(qid, ranking, is_ranked=True)

    print("\n\n")
    print(ranking_logger.filename)
    print("#> Done.")
    print("\n\n")
