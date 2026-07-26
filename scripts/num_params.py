import os
import sys; sys.path.insert(0, os.getcwd())

import typer

from pipelines import get_pipeline


CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
app = typer.Typer(
    help="Calculate the number of parameters.",
    add_completion=False,
    context_settings=CONTEXT_SETTINGS,
)

@app.command()
def main(
    fpath_config: str = typer.Argument(help="Path to the configuration file."),
):
    pipeline = get_pipeline(fpath_config)
    print(f"Number of parameters: {pipeline.n_params:,}")


if __name__ == "__main__":
    app()
