import os
from pathlib import Path
import sys
import tempfile
import urllib.request

import typer
from datasets import load_dataset
import sentencepiece as spm
from tokenizers import SentencePieceUnigramTokenizer


CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
app = typer.Typer(
    help="Train a SentencePiece tokenizer on a fineweb-2-edu-japanese dataset.",
    add_completion=False,
    context_settings=CONTEXT_SETTINGS
)


@app.command()
def main(
    model_prefix: str = typer.Option(
        "trained/tokenizer",
        help="Path to save the trained tokenizer model.",
    ),
    n_vocab: int = typer.Option(
        16_000,
        help="Vocabulary size for the tokenizer.",
    ),
    n_rows: int = typer.Option(
        1_000_000,
        help="Number of rows to use for training the tokenizer.",
    ),
):
    model_prefix = Path(model_prefix)
    model_prefix.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        "hotchpotch/fineweb-2-edu-japanese",
        "small_tokens_cleaned",
        split="train",
        streaming=True,
    )

    def ds_iter():
        it = iter(ds["text"])
        for _ in range(n_rows):
            yield next(it)

    spm.SentencePieceTrainer.Train(
        sentence_iterator=ds_iter(),
        model_prefix=model_prefix,
        vocab_size=n_vocab,
        model_type="unigram",
        character_coverage=0.9995,
        train_extremely_large_corpus=True,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        dst = os.path.join(tmpdir, "sentencepiece_model_pb2.py")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/google/sentencepiece/master/python/src/sentencepiece/sentencepiece_model_pb2.py",
            dst
        )
        sys.path.insert(0, tmpdir)
        tokenizer = SentencePieceUnigramTokenizer.from_spm(
            model_prefix.with_suffix(".model")
        )

    fpath_tokenizer = str(model_prefix.with_suffix(".json"))
    tokenizer.save(fpath_tokenizer)
    print("Tokenizer saved to:", fpath_tokenizer)


if __name__ == "__main__":
    app()
