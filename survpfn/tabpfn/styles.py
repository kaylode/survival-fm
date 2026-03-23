import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from cycler import cycler
from functools import partial
from typing import Tuple, Optional, List


def check_paths(in_paths: dict, out_paths: dict) -> None:
    """
    Verify input paths exist and create output directories if needed.
    
    Parameters
    ----------
    in_paths : dict
        Dictionary with input paths. Keys ending with '_dir' are validated as directories.
    out_paths : dict
        Dictionary with output paths. Directories are created if they don't exist.
    """
    for pth_key in in_paths:
        pth = in_paths[pth_key]
        if not os.path.exists(pth):
            print(f'Path [{pth}] does not exist')
        if pth_key.endswith('_dir') and (not os.path.isdir(pth)):
            print(f'Path [{pth}] does not correspond to a directory!')

    for pth_key in out_paths:
        pth = out_paths[pth_key]
        if pth_key.endswith('_dir'):
            abs_path = os.path.abspath(pth)
        else:
            abs_path = os.path.abspath(os.path.dirname(pth))
        if not os.path.exists(abs_path):
            print(f'Creating path: [{abs_path}]')
            os.makedirs(abs_path)


def save_fig(
    fig: Figure, 
    fig_name: str, 
    fig_dir: str, 
    fig_fmt: str = 'pdf',
    fig_size: Tuple[float, float] = (6.4, 4), 
    save: bool = True, 
    dpi: int = 300,
    transparent_png: bool = True,
) -> None:
    """
    Save matplotlib figure with appropriate settings for research papers.
    
    Parameters
    ----------
    fig : Figure
        Matplotlib figure instance
    fig_name : str
        File name where the figure is saved (without extension)
    fig_dir : str
        Path to the directory where the figure is saved
    fig_fmt : str, optional
        Format of the figure (pdf, png, etc.), by default 'pdf'
    fig_size : Tuple[float, float], optional
        Size of the figure in inches, by default (6.4, 4)
    save : bool, optional
        If the figure should be saved, by default True
    dpi : int, optional
        Dots per inch for rasterized formats, by default 300
    transparent_png : bool, optional
        If the background should be transparent for png, by default True
    """
    if not save:
        return
    
    fig.set_size_inches(fig_size, forward=False)
    fig_fmt = fig_fmt.lower()
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    
    pth = os.path.join(fig_dir, f'{fig_name}.{fig_fmt}')
    
    if fig_fmt == 'pdf':
        metadata = {
            'Creator': '',
            'Producer': '',
            'CreationDate': None
        }
        fig.savefig(pth, bbox_inches='tight', metadata=metadata)
    elif fig_fmt == 'png':
        alpha = 0 if transparent_png else 1
        axes = fig.get_axes()
        fig.patch.set_alpha(alpha)
        for ax in axes:
            ax.patch.set_alpha(alpha)
        fig.savefig(pth, bbox_inches='tight', dpi=dpi)
    else:
        try:
            fig.savefig(pth, bbox_inches='tight')
        except Exception as e:
            print(f"Cannot save figure: {e}")


def get_colorblind_cycler() -> cycler:
    """
    Get cycler with colors distinguishable by colorblind people.
    
    Returns
    -------
    cycler
        Cycler object with colorblind-friendly colors
    """
    return cycler(color=plt.rcParams['axes.prop_cycle'].by_key()['color'])


def get_marker_cycler(markers: Optional[List[str]] = None) -> cycler:
    """
    Get cycler for different markers.
    
    Parameters
    ----------
    markers : List[str], optional
        List of marker styles, by default ['o', 'x', 's', 'P']
    
    Returns
    -------
    cycler
        Cycler object with marker styles
    """
    if markers is None:
        markers = ['o', 'x', 's', 'P']
    return cycler(marker=markers)


def get_linestyle_cycler(linestyles: Optional[List[str]] = None) -> cycler:
    """
    Get cycler for different line styles.
    
    Parameters
    ----------
    linestyles : List[str], optional
        List of line styles, by default ['-', '--', ':', '-.']
    
    Returns
    -------
    cycler
        Cycler object with line styles
    """
    if linestyles is None:
        linestyles = ['-', '--', ':', '-.']
    return cycler(ls=linestyles)


def get_bw_cycler() -> cycler:
    """
    Get black and white cycler combining line styles and markers.
    
    Returns
    -------
    cycler
        Cycler object for black and white figures
    """
    return (
        cycler(color=['black']) *
        cycler(ls=['-', '--', ':', '-.']) *
        cycler(marker=['o', 'x', 's', 'P'])
    )


def apply_paper_style(style: str = 'seaborn-v0_8-paper') -> None:
    """
    Apply a matplotlib style suitable for research papers.
    
    Parameters
    ----------
    style : str, optional
        Style name, by default 'seaborn-v0_8-paper'
        Options: 'seaborn-paper', 'seaborn-talk', 'seaborn-notebook', 
                 'seaborn-v0_8-paper', 'tableau-colorblind10'
    """
    plt.style.use(style)


def setup_figure(
    figsize: Tuple[float, float] = (6.4, 4),
    style: str = 'seaborn-v0_8-paper',
    use_cycler: Optional[cycler] = None
) -> Tuple[Figure, plt.Axes]:
    """
    Create a figure with paper-appropriate settings.
    
    Parameters
    ----------
    figsize : Tuple[float, float], optional
        Figure size in inches, by default (6.4, 4) for 16:10 ratio
    style : str, optional
        Matplotlib style to use, by default 'seaborn-v0_8-paper'
    use_cycler : cycler, optional
        Custom cycler for line properties, by default None
    
    Returns
    -------
    Tuple[Figure, plt.Axes]
        Figure and axes objects
    """
    apply_paper_style(style)
    fig, ax = plt.subplots(figsize=figsize)
    
    if use_cycler is not None:
        ax.set_prop_cycle(use_cycler)
    
    return fig, ax


def create_savefig_partial(
    fig_dir: str,
    fig_fmt: str = 'pdf',
    fig_size: Tuple[float, float] = (6.4, 4),
    save: bool = True,
    transparent_png: bool = True,
    dpi: int = 300
):
    """
    Create a partial function with preset save_fig parameters.
    
    Parameters
    ----------
    fig_dir : str
        Directory to save figures
    fig_fmt : str, optional
        Figure format, by default 'pdf'
    fig_size : Tuple[float, float], optional
        Figure size, by default (6.4, 4)
    save : bool, optional
        Whether to save figures, by default True
    transparent_png : bool, optional
        Transparent background for PNG, by default True
    dpi : int, optional
        DPI for rasterized formats, by default 300
    
    Returns
    -------
    partial
        Partial function with preset parameters
    """
    return partial(
        save_fig,
        fig_dir=fig_dir,
        fig_fmt=fig_fmt,
        fig_size=fig_size,
        save=save,
        transparent_png=transparent_png,
        dpi=dpi
    )


# Common figure size presets for different aspect ratios
FIGSIZE_PRESETS = {
    '16:10_default': (6.4, 4),
    '16:10_small': (6, 3.75),
    '16:9_default': (6.4, 3.6),
    '16:9_small': (6, 3.375),
    '4:3_default': (6.4, 4.8),
}