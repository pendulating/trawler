import os
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt


def load_ibm_plex_sans(
    search_dirs=(
        "/share/pierson/matt/UAIR/assets/fonts/ibx-plex-sans",
        "/share/pierson/matt/UAIR/assets/fonts/ibm-plex-sans",
    ),
    font_filename="IBMPlexSans-Regular.ttf",
    default_size=11,
):
    """Load IBM Plex Sans for Matplotlib from a locally extracted directory.

    - Searches for `font_filename` within `search_dirs` recursively
    - Registers the font with Matplotlib and sets it as default family
    - Sets default font size to `default_size`

    Raises FileNotFoundError if the font file is not found.
    """
    target_ttf = None
    for base in search_dirs:
        if not os.path.isdir(base):
            continue
        direct = os.path.join(base, font_filename)
        if os.path.exists(direct):
            target_ttf = direct
            break
        for root, _dirs, files in os.walk(base):
            for f in files:
                if f.lower() == font_filename.lower():
                    target_ttf = os.path.join(root, f)
                    break
            if target_ttf:
                break
        if target_ttf:
            break

    if not target_ttf:
        raise FileNotFoundError(
            f"{font_filename} not found in any of: {', '.join(search_dirs)}"
        )

    fm.fontManager.addfont(target_ttf)
    plt.rcParams["font.family"] = "IBM Plex Sans"
    plt.rcParams["font.size"] = default_size
    return target_ttf


