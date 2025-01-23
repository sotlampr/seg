import os


def get_pretrained_fname(fname):
    return os.path.join(
        os.path.join(
            os.path.abspath(
                os.path.join(os.path.split(__file__)[0], "..")
            ),
            "pretrained"
        ),
        fname
    )


def check_for_file(pretrained_weights, get_url, *args):
    if pretrained_weights is not None \
            and not os.path.exists(pretrained_weights):
        raise FileNotFoundError(
                    f"Model not found. Download url: {get_url(*args)}"
                )
