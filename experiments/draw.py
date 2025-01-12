import matplotlib.pyplot as plt

###############################################################################
#                            Helper Functions
###############################################################################

def _create_figure_and_axes(figsize=(6, 8)):
    """
    Creates a matplotlib figure and axes, returns both.
    """
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax

def _configure_axes(ax, total_memory, ylabel='Memory (GB)', title_prefix='Total Memory'):
    """
    Sets y-axis limits, labels, title, and removes x-axis ticks.
    """
    ax.set_ylim(0, total_memory + 10)
    ax.set_ylabel(ylabel)
    ax.set_title(f'{title_prefix}: {total_memory:.1f} GB')
    ax.set_xticks([])
    ax.grid(False)

def _save_and_close(fig, path):
    """
    Calls tight_layout, saves the figure, then closes to free memory.
    """
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close(fig)

def _plot_vertical_stack(ax, x_position, categories, values, colors, bar_width):
    """
    Plots vertical (stacked) bars at x_position and returns the total height (bottom).
    """
    bottom = 0
    for cat, val, col in zip(categories, values, colors):
        ax.bar(
            x=x_position, height=val, width=bar_width*2,
            bottom=bottom, color=col, label=f'{cat} ({val} GB)'
        )
        bottom += val
    return bottom

def _add_vertical_stack_labels(ax, x_position, categories, values, bar_width, text_color='white'):
    """
    Adds text labels centered in each vertical stacked bar.
    """
    bottom = 0
    for cat, val in zip(categories, values):
        ax.text(
            x_position,
            bottom + val / 2,
            f'{cat} ({val:.2f} GB)',
            ha='center', va='center', color=text_color, fontsize=10
        )
        bottom += val
    return bottom

def _plot_horizontal_components(ax, bar_width, bottom_offset, components):
    """
    Plots horizontal bars (offset in x-direction) at the same vertical level (bottom_offset).
    `components` is expected to be a list of tuples: (x_offset, value, color, label_text).
    Example:
        components = [
            (0,  gradient, 'yellow', 'Gradients'),
            (bar_width/2, activation, 'green', 'Activations'),
            ...
        ]
    """
    for x_offset, val, color, label in components:
        ax.bar(
            x_offset, val, width=bar_width,
            bottom=bottom_offset, color=color, label=f'{label} ({val} GB)'
        )

def _add_horizontal_labels(ax, bar_width, bottom_offset, components):
    """
    Adds text labels centered in each horizontal bar.
    Same structure of `components` as in _plot_horizontal_components.
    """
    for x_offset, val, _, label in components:
        ax.text(
            x_offset,
            bottom_offset + val / 2,
            f'{label} ({val:.2f} GB)',
            ha='center', va='center', fontsize=10
        )

###############################################################################
#                1) save_memory_usage_figure_gpu (Unoptimized GPU)
###############################################################################
def save_memory_usage_figure_gpu(path: str, weight, optimizer, gradient, activation):
    """
    Plots memory usage for GPU scenario with separate static (weight, optimizer)
    and dynamic (gradient, activation) components.
    """
    # ------------------ Calculate memory usage ------------------ #
    static_categories = ['Model Parameters', 'Optimizer States']
    static_values = [weight, optimizer]
    static_colors = ['red', 'blue']
    static_total_memory = sum(static_values)
    
    dynamic_categories = ['Gradients', 'Activations']
    dynamic_values = [gradient, activation]
    dynamic_colors = ['yellow', 'green']
    dynamic_total_memory = max(dynamic_values)   # as per original logic
    
    total_memory = static_total_memory + dynamic_total_memory

    # ------------------ Create figure and axes ------------------ #
    fig, ax = _create_figure_and_axes(figsize=(6, 8))
    bar_width = 0.5

    # ------------------ Plot static (vertical stack) ------------ #
    static_bottom = _plot_vertical_stack(
        ax, x_position=0,
        categories=static_categories,
        values=static_values,
        colors=static_colors,
        bar_width=bar_width
    )

    # ------------------ Plot dynamic (horizontal) --------------- #
    # Gradients at x=0, Activations at x=bar_width/2
    components = [
        (0,           dynamic_values[0], dynamic_colors[0], dynamic_categories[0]),  # Gradients
        (bar_width/2, dynamic_values[1], dynamic_colors[1], dynamic_categories[1])   # Activations
    ]
    _plot_horizontal_components(ax, bar_width*2, static_bottom, [components[0]])
    _plot_horizontal_components(ax, bar_width,   static_bottom, [components[1]])
    
    # ------------------ Configure axis and title ---------------- #
    _configure_axes(ax, total_memory)

    # ------------------ Add text labels ------------------------- #
    # Labels for static bars
    static_bottom = _add_vertical_stack_labels(
        ax, x_position=0,
        categories=static_categories,
        values=static_values,
        bar_width=bar_width
    )
    
    # Labels for dynamic bars
    # We must manually align the x-coords for textual annotation 
    # in the same style as in the original code.
    ax.text(
        0 - bar_width/2,
        static_bottom + dynamic_values[0] / 2,
        f'{dynamic_categories[0]} ({dynamic_values[0]:.2f} GB)',
        ha='center', va='center', fontsize=10
    )
    ax.text(
        0 + bar_width/2,
        static_bottom + dynamic_values[1] / 2,
        f'{dynamic_categories[1]} ({dynamic_values[1]:.2f} GB)',
        ha='center', va='center', fontsize=10
    )

    # ------------------ Save and close -------------------------- #
    _save_and_close(fig, path)

###############################################################################
#            2) save_optimized_memory_usage_figure_gpu (Optimized GPU)
###############################################################################
def save_optimized_memory_usage_figure_gpu(path: str, weight, optimizer, gradient, activation, checkpointed):
    """
    Similar GPU memory usage plot, but for an 'optimized' scenario
    where the dynamic memory includes gradient, activation, and checkpointed.
    """
    # ------------------ Calculate memory usage ------------------ #
    static_categories = ['Model Parameters', 'Optimizer States']
    static_values = [weight, optimizer]
    static_colors = ['red', 'blue']
    static_total_memory = sum(static_values)
    
    # As per original logic:
    dynamic_total_memory = max(gradient, activation + checkpointed)
    total_memory = static_total_memory + dynamic_total_memory

    # ------------------ Create figure and axes ------------------ #
    fig, ax = _create_figure_and_axes(figsize=(6, 8))
    bar_width = 0.5

    # ------------------ Plot static (vertical stack) ------------ #
    static_bottom = _plot_vertical_stack(
        ax, x_position=0,
        categories=static_categories,
        values=static_values,
        colors=static_colors,
        bar_width=bar_width
    )

    # ------------------ Plot dynamic (horizontal) --------------- #
    # Original code sets:
    #   Gradients (yellow), x=0
    #   Activations (green), x=bar_width/2
    #   Checkpointed (orange), x=bar_width/2 stacked on top of 'Activations'
    ax.bar(
        0, gradient, width=bar_width*2,
        bottom=static_bottom, color="yellow",
        label=f'Gradients ({gradient} GB)'
    )
    ax.bar(
        bar_width/2, activation, width=bar_width,
        bottom=static_bottom, color='green',
        label=f'Activations ({activation} GB)'
    )
    ax.bar(
        bar_width/2, checkpointed, width=bar_width,
        bottom=static_bottom + activation, color="orange",
        label=f'Checkpointed ({checkpointed} GB)'
    )

    # ------------------ Configure axis and title ---------------- #
    _configure_axes(ax, total_memory)

    # ------------------ Add text labels ------------------------- #
    # Labels for static bars
    static_bottom = _add_vertical_stack_labels(
        ax, x_position=0,
        categories=static_categories,
        values=static_values,
        bar_width=bar_width
    )
    
    # Now text for dynamic
    ax.text(
        0 - bar_width/2,
        static_bottom + gradient / 2,
        f'Gradients ({gradient:.2f} GB)',
        ha='center', va='center', fontsize=10
    )
    ax.text(
        0 + bar_width/2,
        static_bottom + activation / 2,
        f'Activations ({activation:.2f} GB)',
        ha='center', va='center', fontsize=10
    )
    ax.text(
        0 + bar_width/2,
        static_bottom + activation + checkpointed / 2,
        f'Checkpointed ({checkpointed:.2f} GB)',
        ha='center', va='center', fontsize=10
    )

    # ------------------ Save and close -------------------------- #
    _save_and_close(fig, path)

###############################################################################
#               3) save_optimized_memory_usage_figure_cpu (CPU)
###############################################################################
def save_optimized_memory_usage_figure_cpu(path: str, forward, backward):
    """
    Plots memory usage for CPU scenario with forward activations vs. backward grads/activations.
    """
    # ------------------ Calculate memory usage ------------------ #
    total_memory = max(forward, backward)

    # ------------------ Create figure and axes ------------------ #
    fig, ax = _create_figure_and_axes(figsize=(6, 8))
    bar_width = 0.5

    # ------------------ Plot dynamic memory (horizontal) -------- #
    ax.bar(
        0, forward,
        width=bar_width*2, bottom=0,
        color="yellow", label=f'Forward Activations ({forward:.2f} GB)'
    )
    ax.bar(
        bar_width/2, backward,
        width=bar_width, bottom=0,
        color='green', label=f'Backward Gradients/Activations ({backward:.2f} GB)'
    )

    # ------------------ Configure axis and title ---------------- #
    _configure_axes(ax, total_memory)

    # ------------------ Add text labels ------------------------- #
    ax.text(
        0 - bar_width/2,
        forward / 2,
        f'Forward Activations ({forward:.2f} GB)',
        ha='center', va='center', fontsize=10
    )
    ax.text(
        0 + bar_width/2,
        backward / 2,
        f'Backward Gradients/Activations ({backward:.2f} GB)',
        ha='center', va='center', fontsize=10
    )

    # ------------------ Save and close -------------------------- #
    _save_and_close(fig, path)

###############################################################################
#               4) save_cpu_memory_usage_figure_cpu (CPU)
###############################################################################
def save_cpu_memory_usage_figure_cpu(path: str, 
                                     fp32_weight, 
                                     fp32_grad, 
                                     fp32_optimizer, 
                                     fp16_weight, 
                                     fp16_grad, 
                                     checkpointed, 
                                     others):
    """
    Plots a stacked bar for various CPU memory categories (FP32, FP16, checkpointed, others, etc.).
    """
    # ------------------ Setup categories and values ------------- #
    categories = [
        'FP32 Model Parameters',
        'FP32 Gradients',
        'FP32 Optimizer States',
        'FP16 Model Parameters',
        'FP16 Gradients',
        'Checkpointed States',
        'Other'
    ]
    values = [
        fp32_weight,
        fp32_grad,
        fp32_optimizer,
        fp16_weight,
        fp16_grad,
        checkpointed,
        others
    ]
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'gray']
    
    total_memory = sum(values)

    # ------------------ Create figure and axes ------------------ #
    fig, ax = _create_figure_and_axes(figsize=(6, 8))
    bar_width = 0.5

    # ------------------ Plot stacked bars (vertical) ------------ #
    bottom = 0
    for category, value, color in zip(categories, values, colors):
        ax.bar(
            x=0, height=value, width=bar_width*2,
            bottom=bottom, color=color,
            label=f'{category} ({value} GB)'
        )
        bottom += value

    # ------------------ Configure axis and title ---------------- #
    _configure_axes(ax, total_memory)

    # ------------------ Add stacked bar labels ------------------ #
    bottom = 0
    for category, value in zip(categories, values):
        ax.text(
            0, bottom + value / 2,
            f'{category} ({value:.2f} GB)',
            ha='center', va='center', color='white', fontsize=10
        )
        bottom += value

    # ------------------ Save and close -------------------------- #
    _save_and_close(fig, path)
