import random
import time
import torch
import torch.nn as nn
import numpy as np

from transformers import AdamW
from xlmr_colbert.utils.runs import Run
from xlmr_colbert.utils.amp import MixedPrecisionManager

from xlmr_colbert.training.bilingual_batcher import BilingualBatcher
from xlmr_colbert.parameters import DEVICE

from xlmr_colbert.modeling.colbert import ColBERT
from xlmr_colbert.utils.utils import print_message
from xlmr_colbert.training.utils import print_progress, manage_checkpoints


def train(args):
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    if args.distributed:
        torch.cuda.manual_seed_all(12345)

    if args.distributed:
        assert args.bsize % args.nranks == 0, (args.bsize, args.nranks)
        assert args.accumsteps == 1
        args.bsize = args.bsize // args.nranks

        print(
            "Using args.bsize =",
            args.bsize,
            "(per process) and args.accumsteps =",
            args.accumsteps,
        )

    reader = BilingualBatcher(args, (0 if args.rank == -1 else args.rank), args.nranks)

    if args.rank not in [-1, 0]:
        torch.distributed.barrier()

    colbert = ColBERT.from_pretrained(
        args.base_model,
        query_maxlen=args.query_maxlen,
        doc_maxlen=args.doc_maxlen,
        dim=args.dim,
        similarity_metric=args.similarity,
        mask_punctuation=args.mask_punctuation,
    )

    colbert.roberta.resize_token_embeddings(len(colbert.tokenizer))

    if args.checkpoint is not None:
        # assert args.resume_optimizer is False, "TODO: This would mean reload optimizer too."
        if not args.resume_optimizer:
            print_message(
                f"#> Starting from checkpoint {args.checkpoint} -- but NOT the optimizer!"
            )

        checkpoint = torch.load(args.checkpoint, map_location="cpu")

        try:
            colbert.load_state_dict(checkpoint["model_state_dict"])
        except:
            print_message("[WARNING] Loading checkpoint with strict=False")
            colbert.load_state_dict(checkpoint["model_state_dict"], strict=False)

    if args.rank == 0:
        torch.distributed.barrier()

    colbert = colbert.to(DEVICE)
    colbert.train()

    if args.distributed:
        colbert = torch.nn.parallel.DistributedDataParallel(
            colbert,
            device_ids=[args.rank],
            output_device=args.rank,
            find_unused_parameters=True,
        )

    # Attempt to load optimizer
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, colbert.parameters()), lr=args.lr, eps=1e-8
    )
    if args.resume_optimizer:
        assert args.checkpoint is not None
        print_message("#> Loading the optimizer")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    optimizer.zero_grad()

    amp = MixedPrecisionManager(args.amp)
    criterion = nn.CrossEntropyLoss()
    labels = torch.zeros(args.bsize, dtype=torch.long, device=DEVICE)

    start_time = time.time()
    train_loss = 0.0

    start_batch_idx = 0

    if args.resume:
        assert args.checkpoint is not None
        start_batch_idx = checkpoint["batch"]

        reader.skip_to_batch(start_batch_idx, checkpoint["arguments"]["bsize"])

    for batch_idx, BatchSteps in zip(range(start_batch_idx, args.maxsteps), reader):
        this_batch_loss = 0.0

        for queries, passages in BatchSteps:
            with amp.context():
                scores = colbert(queries, passages).view(2, -1).permute(1, 0)
                loss = criterion(scores, labels[: scores.size(0)])
                loss = loss / args.accumsteps

            if args.rank < 1:
                print_progress(scores)

            amp.backward(loss)

            train_loss += loss.item()
            this_batch_loss += loss.item()

        amp.step(colbert, optimizer)

        if args.rank < 1:
            avg_loss = train_loss / (batch_idx + 1)

            num_examples_seen = (batch_idx - start_batch_idx) * args.bsize * args.nranks
            elapsed = float(time.time() - start_time)

            log_to_mlflow = batch_idx % 20 == 0
            Run.log_metric(
                "train/avg_loss", avg_loss, step=batch_idx, log_to_mlflow=log_to_mlflow
            )
            Run.log_metric(
                "train/batch_loss",
                this_batch_loss,
                step=batch_idx,
                log_to_mlflow=log_to_mlflow,
            )
            Run.log_metric(
                "train/examples",
                num_examples_seen,
                step=batch_idx,
                log_to_mlflow=log_to_mlflow,
            )
            Run.log_metric(
                "train/throughput",
                num_examples_seen / elapsed,
                step=batch_idx,
                log_to_mlflow=log_to_mlflow,
            )

            print_message(batch_idx, avg_loss)
            manage_checkpoints(args, colbert, optimizer, batch_idx + 1)
