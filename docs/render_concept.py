"""Render an editorial diagram of the mechanism; no experimental data is used."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'heritable_social_learning/.mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def render():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none'})
    paper, ink, muted, teal = '#faf8f3', '#243532', '#68726c', '#2b7469'
    fig, ax = plt.subplots(figsize=(16, 8.5))
    fig.patch.set_facecolor(paper)
    ax.set(xlim=(0, 16), ylim=(0, 8.5))
    ax.axis('off')
    ax.text(.9, 7.98, 'RESEARCH NOTE  /  HERITABLE SOCIAL LEARNING', fontsize=10, color=teal)
    ax.text(.9, 7.18, 'What survives a generation?', fontsize=32, color=ink, fontfamily='DejaVu Serif')
    ax.text(.9, 6.64, 'A prior for learning from imperfect evidence.', fontsize=15, color=muted)
    ax.plot([.9, 15.1], [6.2, 6.2], color='#d9ddd5', linewidth=1)

    for x, number, title, lines in [
        (.95, '01', 'A newborn', ['Fresh neural weights.', 'An inherited learning prior.', 'No inherited experience.']),
        (6.0, '02', 'A lifetime', ['Private observations.', 'Noisy peer evidence.', 'The prior shapes learning.']),
        (11.1, '03', 'An evaluation', ['Fresh held-out episodes.', 'Weights are frozen.', 'Performance guides selection.']),
    ]:
        ax.text(x, 5.70, number, fontsize=11, color=teal)
        ax.text(x, 5.10, title, fontsize=21, color=ink, fontfamily='DejaVu Serif')
        for i, line in enumerate(lines):
            ax.text(x, 4.57-i*.34, line, fontsize=12, color=muted)

    def arrow(start, end, color=teal):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=15,
                                    linewidth=1.6, color=color))
    arrow((4.15, 5.25), (5.42, 5.25))
    arrow((9.40, 5.25), (10.53, 5.25))
    ax.text(.95, 3.36, 'WITHIN ONE LIFE', fontsize=9, color=muted)
    ax.plot([3.15, 15.1], [3.40, 3.40], color='#d9ddd5', linewidth=1, linestyle=(0,(3,5)))

    # The return path is the only transmission channel between generations.
    ax.plot([13.00, 13.00], [3.05, 2.03], color=teal, linewidth=1.6)
    arrow((13.00, 2.03), (10.56, 2.03))
    ax.plot([5.44, 2.55], [2.03, 2.03], color=teal, linewidth=1.6)
    arrow((2.55, 2.03), (2.55, 3.05))
    ax.text(11.45, 2.30, 'Fitness', fontsize=10, color=muted)
    ax.text(3.27, 2.30, 'Only the genome', fontsize=10, color=teal)
    ax.add_patch(FancyBboxPatch((5.48, 1.23), 5.04, 1.60,
                               boxstyle='round,pad=.02,rounding_size=.06',
                               facecolor='#edf1e9', edgecolor='none'))
    ax.text(8, 2.42, 'SELECTION + MUTATION', fontsize=10, color=teal, ha='center')
    ax.text(8, 1.92, 'α   ·   β   ·   γ', fontsize=24, color=ink, ha='center', fontfamily='DejaVu Serif')
    ax.text(8, 1.52, 'private   /   social   /   selectivity', fontsize=10, color=muted, ha='center')
    ax.text(.95, .69, 'Across generations, the learning prior changes. Every newborn learns for itself.',
            fontsize=12, color=ink, fontfamily='DejaVu Serif')
    ax.text(.95, .25, 'Conceptual diagram · Neural weights, memories and optimizer state reset at birth · No claim of cumulative culture',
            fontsize=9, color=muted)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    dest = ROOT / 'docs/figures'
    dest.mkdir(parents=True, exist_ok=True)
    for ext in ['svg', 'png']:
        fig.savefig(dest / f'learning-prior-cycle.{ext}', dpi=160, facecolor=paper)
    svg = dest / 'learning-prior-cycle.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)


if __name__ == '__main__':
    render()
