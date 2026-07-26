import os
import sys; sys.path.insert(0, os.getcwd())

import typer

from pipelines import get_pipeline


CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
app = typer.Typer(
    help=(
        "Calculate the number of training steps based on the model "
        "configuration and the scaling law."
    ),
    add_completion=False,
    context_settings=CONTEXT_SETTINGS,
)

@app.command()
def main(
    fpath_config: str = typer.Argument(help="Path to the configuration file."),
):
    pipeline = get_pipeline(fpath_config)
    n_params = pipeline.n_params
    total_tokens = n_params * 20
    batch_size = pipeline.config.train.batch_size
    max_len = pipeline.config.model.arch["max_len"]
    n_tokens_per_step = max_len * batch_size
    n_steps = total_tokens // n_tokens_per_step

    print(f"Number of parameters: {n_params:,}")
    print(f"Total tokens to train: {total_tokens:,}")
    print(f"Batch size: {batch_size:}")
    print(f"Max sequence length: {max_len:,}")
    print(f"Tokens per training step: {n_tokens_per_step:,}")
    print(f"Total training steps: {n_steps:,}")


if __name__ == "__main__":
    app()
