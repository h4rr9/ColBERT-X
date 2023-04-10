from xlmr_colbert.utils.parser import Arguments
from xlmr_colbert.utils.runs import Run
from xlmr_colbert.training.bilingual_training import train


def main():
    parser = Arguments(
        description="Training ColBERT with bilingual <query, positive query, negative query> and <doc, positive doc, negative doc> triples."
    )

    parser.add_model_parameters()
    parser.add_model_training_parameters()
    parser.add_bilingual_training_input()

    args = parser.parse()

    assert args.bsize % args.accumsteps == 0, (
        (args.bsize, args.accumsteps),
        "The batch size must be divisible by the number of gradient accumulation steps.",
    )
    assert args.query_maxlen <= 512
    assert args.doc_maxlen <= 512

    with Run.context(consider_failed_if_interrupted=False):
        train(args)


if __name__ == "__main__":
    main()
