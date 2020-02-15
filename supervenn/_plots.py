# -*- coding: utf-8 -*-
"""
Routines for plotting multiple sets.
"""
import numpy as np
import matplotlib.pyplot as plt

from supervenn._algorithms import (
    get_chunks_and_composition_array,
    get_permutations,
    DEFAULT_MAX_BRUTEFORCE_SIZE,
    DEFAULT_SEEDS,
    DEFAULT_NOISE_PROB)


DEFAULT_FONTSIZE = 12
DEFAULT_FIGSIZE = (12, 7)


def get_alternated_ys(ys_count, low, high):
    """
    A helper function generating y-positions for x-axis annotations, useful when some annotations positioned along the
    x axis are too crowded.
    :param ys_count: integer from 1 to 3.
    :param low: lower bound of the area designated for annotations
    :param high: higher bound for thr area designated for annotations.
    :return:
    """
    if ys_count not in [1, 2, 3]:
        raise ValueError('Argument ys_count should be 1, 2 or 3.')
    if ys_count == 1:
        coefs = [0.5]
        vas = ['center']
    elif ys_count == 2:
        coefs = [0.15, 0.85]
        vas = ['bottom', 'top']
    else:
        coefs = [0.15, 0.5, 0.85]
        vas = ['bottom', 'center', 'top']

    ys = [low + coef * (high - low) for coef in coefs]

    return ys, vas


def plot_binary_array(arr, ax=None, col_widths=None, row_heights=None,
                      row_annotations=None, row_annotations_y=0.5,
                      col_annotations=None, col_annotations_area_height=0.75, col_annotations_ys_count=1,
                      rotate_col_annotations=False,
                      color_by='row', bar_height=1, bar_alpha=0.6, bar_align='edge', color_cycle=None,
                      fontsize=DEFAULT_FONTSIZE):
    """
    Visualize a binary array as a grid with variable sized columns and rows, where cells with 1 are filled using bars
    and cells with 0 are blank.
    :param arr: numpy.array of zeros and ones
    :param ax: axis to plot into (current axis by default)
    :param col_widths: widths for grid columns, must have len equal to arr.shape[1]
    :param row_heights: heights for grid rows, must have len equal to arr.shape[0]
    :param row_annotations: annotations for each row, plotted in the middle of the row
    :param row_annotations_y: a number in (0, 1), position for row annotations in the row. Default 0.5 - center of row.
    :param col_annotations: annotations for columns, plotted in the bottom, below the x axis.
    :param col_annotations_area_height: height of area for column annotations in inches, 1 by default
    :param col_annotations_ys_count: 1 (default), 2, or 3 - use to reduce clutter in column annotations area
    :param rotate_col_annotations: True / False
    :param color_by: 'row' (default) or 'column'. If 'row', all cells in same row are same color, etc.
    :param bar_height: height of cell fill as a fraction of row height, a number in (0, 1).
    :param bar_alpha: alpha for cell fills.
    :param bar_align: vertical alignment of bars, 'edge' (defaulr) or 'center'. Only matters when bar_height < 1.
    :param color_cycle: a list of colors, given as names of matplotlib named colors, or hex codes (e.g. '#1f77b4')
    :param fontsize: font size for annotations (default {}).
    """.format(DEFAULT_FONTSIZE)
    if row_heights is None:
        row_heights = [1] * arr.shape[0]

    if col_widths is None:
        col_widths = [1] * arr.shape[1]

    if len(row_heights) != arr.shape[0]:
        raise ValueError('len(row_heights) doesnt match number of rows of array')

    if len(col_widths) != arr.shape[1]:
        raise ValueError('len(col_widths) doesnt match number of columns of array')

    allowed_argument_values = {
        'bar_align': ['center', 'edge'],
        'color_by': ['row', 'column'],
        'col_annotations_ys_count': [1, 2, 3],
    }

    for argument_name, allowed_argument_values in allowed_argument_values.items():
        if locals()[argument_name] not in allowed_argument_values:
            raise ValueError('Argument {} should be one of {}'.format(argument_name, allowed_argument_values))

    if not 0 <= row_annotations_y <= 1:
        raise ValueError('row_annotations_y should be a number between 0 and 1')

    if color_cycle is None:
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    grid_xs = np.insert(np.cumsum(col_widths), 0, 0)[:-1]
    grid_ys = np.insert(np.cumsum(row_heights), 0, 0)[:-1]

    if ax is not None:
        plt.sca(ax)

    # BARS
    for row_index, (row, grid_y, row_height) in enumerate(zip(arr, grid_ys, row_heights)):

        bar_y = grid_y + 0.5 * row_height if bar_align == 'center' else grid_y

        for col_index, (is_filled, grid_x, col_width) in enumerate(zip(row, grid_xs, col_widths)):
            if is_filled:
                color_index = row_index if color_by == 'row' else col_index
                color = color_cycle[color_index % len(color_cycle)]
                plt.barh(y=bar_y, left=grid_x, width=col_width, height=bar_height * row_height, align=bar_align,
                         color=color, alpha=bar_alpha)

    # ROW ANNOTATIONS
    if row_annotations is not None:
        for row_index, (grid_y, row_height, annotation) in enumerate(zip(grid_ys, row_heights, row_annotations)):
            annot_y = grid_y + row_annotations_y * row_height
            plt.annotate(xy=(0.5 * sum(col_widths), annot_y), s=str(annotation),
                         ha='center', va='center', fontsize=fontsize)

    # COL ANNOTATIONS
    min_y = 0
    if col_annotations is not None:

        min_y = - 1.0 * col_annotations_area_height / plt.gcf().get_size_inches()[1] * arr.shape[0]

        plt.axhline(0, c='k')

        annot_ys, vas = get_alternated_ys(col_annotations_ys_count, min_y, 0)

        for col_index, (grid_x, col_width, annotation) in enumerate(zip(grid_xs, col_widths, col_annotations)):
            annot_y = annot_ys[col_index % len(annot_ys)]
            plt.annotate(xy=(grid_x + col_width * 0.5, annot_y), s=str(annotation),
                         ha='center', va=vas[col_index % len(vas)], fontsize=fontsize,
                         rotation=90 * rotate_col_annotations)

    plt.xlim(0, sum(col_widths))
    plt.ylim(min_y, sum(row_heights))
    plt.xticks(grid_xs, [])
    plt.yticks(grid_ys, [])
    plt.grid()


def side_plot(values, widths, orient, fontsize=DEFAULT_FONTSIZE, min_width_for_annotation=1, rotate_annotations=False,
              color='gray'):
    """
    Barplot with multiple bars of variable width right next to each other, with an option to rotate the plot 90 degrees.
    :param values: the values to be plotted.
    :param widths: Widths of bars
    :param orient: 'h' / 'horizontal' (default) or 'v' / 'vertical'
    :param fontsize: font size for annotations
    :param min_width_for_annotation: for horizontal plot, don't annotate bars of widths less than this value (to avoid
    clutter)
    """
    bar_edges = np.insert(np.cumsum(widths), 0, 0)
    annotation_positions = [0.5 * (begin + end) for begin, end in zip(bar_edges[:-1], bar_edges[1:])]
    max_value = max(values)
    if orient in ['h', 'horizontal']:
        horizontal = True
        plt.bar(x=bar_edges[:-1], height=values, width=widths, align='edge', alpha=0.5, color=color)
        ticks = plt.xticks
        noticks = plt.yticks
        lim = plt.ylim
    elif orient in ['v', 'vertical']:
        horizontal = False
        plt.barh(y=bar_edges[:-1], width=values, height=widths, align='edge', alpha=0.5, color=color)
        ticks = plt.yticks
        noticks = plt.xticks
        lim = plt.xlim
    else:
        raise ValueError('Unknown orient: {} (should be "h" or "v")'.format(orient))

    for i, (annotation_position, value, width) in enumerate(zip(annotation_positions, values, widths)):
        if width < min_width_for_annotation and horizontal:
            continue
        x, y = 0.5 * max_value, annotation_position
        if horizontal:
            x, y = y, x
        plt.annotate(xy=(x, y), s=value, ha='center', va='center', rotation=rotate_annotations * 90, fontsize=fontsize)

    ticks(bar_edges, [])
    noticks([])
    lim(0, max(values))
    plt.grid()


def balance_widths(widths, maxmin_ratio=0.03):
    """
    Given a list of numbers, apply a linear transformation to every element of the array, such that the maximum value
    remains the same, and the minimum value is maxmin_ratio times the maximum value
    :param widths: list of numbers
    :param maxmin_ratio: the desired max / min ratio in the transformed list.
    :return: transformed list of numbers
    """
    if not 0 <= maxmin_ratio <= 1:
        raise ValueError('maxmin_ratio must be between 0 and 1')
    max_size = max(widths)
    intercept = maxmin_ratio * max_size
    slope = 1 - 1.0 * intercept / max_size
    return [slope * width + intercept for width in widths]


def supervenn(sets, set_annotations=None, figsize=DEFAULT_FIGSIZE, side_plots=True,
              chunk_ordering='minimize gaps', sets_ordering=None,
              reverse_chunks_order=True, reverse_sets_order=True,
              max_bruteforce_size=DEFAULT_MAX_BRUTEFORCE_SIZE, seeds=DEFAULT_SEEDS, noise_prob=DEFAULT_NOISE_PROB,
              side_plot_width=1.5, min_width_for_annotation=1, widths_maxmin_ratio=0, side_plot_color='gray', **kw):
    """
    Plot a diagram visualizing relationship of multiple sets.
    :param sets: list of sets
    :param set_annotations: list of annotations for the sets
    :param figsize: figure size
    :param side_plots: True / False: add small barplots on top and on the right. On top, for each chunk it is shown,
    how many sets does this chunk lie inslde. On the right, set sizes are shown.
    :param chunk_ordering: method of ordering the chunks (columns of the grid)
        - 'minimize gaps' (default): use a smart algorithm to find an order of columns giving fewer gaps in each row,
            making the plot as readable as possible.
        - 'size': bigger chunks go first (or last if reverse_chunks_order=False)
        - 'occurence': chunks that are in most sets go first (or last if reverse_chunks_order=False)
        - 'random': randomly shuffle the columns
    :param sets_ordering: method of ordering the sets (rows of the grid)
        - None (default): keep the order as it is passed
        - 'minimize gaps': use a smart algorithm to find an order of rows giving fewer gaps in each column
        - 'size': bigger sets go first (or last if reverse_sets_order = False)
        - 'chunk count': sets that contain most chunks go first (or last if reverse_sets_order = False)
        - 'random': randomly shuffle
    :param reverse_chunks_order: True (default) / False when chunks_ordering is "size" or "occurence",
        chunks with bigger corresponding property go first if reverse_chunks_order=True, smaller go first if False.
    :param reverse_sets_order: True / False, works the same way as reverse_chunks_order
    :param max_bruteforce_size: maximal number of items for which bruteforce method is applied to find permutation
    :param seeds: number of different random seeds for the randomized greedy algorithm to find permutation
    :param noise_prob: probability of given element being equal to 1 in the noise array for randomized greedy algorithm
    :param side_plot_width: width of side plots in inches (default 1.5)
    :param side_plot_color: color of bars in side plots, default 'gray'
    :param min_width_for_annotation: for horizontal plot, don't annotate bars of widths less than this value (to avoid
    clutter)
    :param widths_maxmin_ratio: desired max/min ratio of displayed chunk widths, default 0 (show actual widths)
    :param row_annotations_y: a number in (0, 1), position for row annotations in the row. Default 0.5 - center of row.
    :param col_annotations_area_height: height of area for column annotations in inches, 1 by default
    :param col_annotations_ys_count: 1 (default), 2, or 3 - use to reduce clutter in column annotations area
    :param color_by: 'row' (default) or 'column'. If 'row', all cells in same row are same color, etc.
    :param bar_height: height of cell fill as a fraction of row height, a number in (0, 1).
    :param bar_alpha: alpha for cell fills.
    :param bar_align: vertical alignment of bars, 'edge' (defaulr) or 'center'. Only matters when bar_height < 1.
    :param color_cycle: a list of colors, given as names of matplotlib named colors, or hex codes (e.g. '#1f77b4')
    """
    # Set up axes
    if side_plots:

        fig_width, fig_height = figsize

        height_ratios = [side_plot_width, fig_height - side_plot_width]
        width_ratios = [fig_width - side_plot_width, side_plot_width]

        _, axes = plt.subplots(2, 2, figsize=figsize, gridspec_kw={'height_ratios': height_ratios,
                                                                   'width_ratios': width_ratios,
                                                                   'hspace': 0.0,
                                                                   'wspace': 0.0})
        for side_ax in axes[0, 1], axes[1, 1], axes[0, 0]:
            side_ax.set_xticks([])
            side_ax.set_yticks([])
        main_ax = axes[1, 0]
    else:
        plt.figure(figsize=figsize)
        main_ax = plt.gca()

    if set_annotations is None:
        set_annotations = ['Set_{}'.format(i) for i in range(len(sets))]

    chunks, composition_array = get_chunks_and_composition_array(sets)

    # Find permutations of rows and columns
    permutations_ = get_permutations(
        chunks,
        composition_array,
        chunks_ordering=chunk_ordering,
        sets_ordering=sets_ordering,
        reverse_chunks_order=reverse_chunks_order,
        reverse_sets_order=reverse_sets_order,
        max_bruteforce_size=max_bruteforce_size,
        seeds=seeds,
        noise_prob=noise_prob)

    # Apply permutations
    chunks = [chunks[i] for i in permutations_['chunks_ordering']]
    composition_array = composition_array[:, permutations_['chunks_ordering']]
    composition_array = composition_array[permutations_['sets_ordering'], :]
    set_annotations = [set_annotations[i] for i in permutations_['sets_ordering']]

    # Main plot
    chunk_sizes = [len(chunk) for chunk in chunks]

    col_widths = balance_widths(chunk_sizes, widths_maxmin_ratio) if widths_maxmin_ratio else chunk_sizes

    plot_binary_array(
        arr=composition_array,
        row_annotations=set_annotations,
        col_annotations=chunk_sizes,
        ax=main_ax,
        col_widths=col_widths,
        row_heights=[1] * len(sets),
        **kw)

    xlim = main_ax.get_xlim()
    ylim = main_ax.get_ylim()
    plt.xlabel('ITEMS', fontsize=kw.get('fontsize', DEFAULT_FONTSIZE))
    plt.ylabel('SETS', fontsize=kw.get('fontsize', DEFAULT_FONTSIZE))

    # Side plots
    if side_plots:
        plt.sca(axes[0, 0])
        side_plot(composition_array.sum(0), col_widths, 'h', min_width_for_annotation=min_width_for_annotation,
                  rotate_annotations=kw.get('rotate_col_annotations', False), color=side_plot_color)
        plt.xlim(xlim)

        plt.sca(axes[1, 1])
        side_plot([len(set_) for set_ in sets], [1] * len(sets), 'v', color=side_plot_color)
        plt.ylim(ylim)
