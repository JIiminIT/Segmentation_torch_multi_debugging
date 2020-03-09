import click
import tinycat as cat


@click.command()
@click.argument("filename")
@click.option(
    "--compress", default=True, help="If true, save nifti into gz compressed format"
)
def to_nifti_cmd(filename, compress):
    cat.to_nifti(filename, compress=compress)


def main():
    # pylint: disable=E1120
    return to_nifti_cmd()
