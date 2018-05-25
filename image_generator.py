from collections import namedtuple
import random
import math
from matplotlib.patches import Circle
import matplotlib.pyplot as plt

# TODO: document!
Dot = namedtuple('Dot', ['x', 'y', 'radius', 'color'])


def clip(val, min_val, max_val):
    return max(min(val, max_val), min_val)


def polar_to_cartesian(theta, r):
    return r*math.cos(theta), r*math.sin(theta)


def no_overlap(dots, x, y, radius):
    can_add = True
    for dot in dots:
        if ((x - dot.x)**2 + (y - dot.y)**2 <
                (radius + dot.radius)**2):
            can_add = False
    return can_add


# TODO: area vs size control, i.e. smart radius
# TODO: smart file names, batch generation
def scattered_random(colors_dict, num_pixels=256, padding=16,
                     min_radius=2, max_radius=8):

    x_min, y_min = padding, padding
    x_max, y_max = num_pixels - padding, num_pixels - padding
    dots = []
    for color in colors_dict:
        for _ in range(colors_dict[color]):
            new_dot_added = False
            while not new_dot_added:
                x = random.uniform(x_min, x_max)
                y = random.uniform(y_min, y_max)
                r = clip(random.gauss(5, 1), min_radius, max_radius)
                # avoid overlap with existing circles
                if no_overlap(dots, x, y, r):
                    dots.append(Dot(x=x, y=y, radius=r, color=color))
                    new_dot_added = True
    return dots


def scattered_pairs(colors_dict, num_pixels=256, padding=16,
                    min_radius=2, max_radius=8):

    x_min, y_min = padding, padding
    x_max, y_max = num_pixels - padding, num_pixels - padding

    assert len(colors_dict) == 2
    sort_dict = sorted(colors_dict.items(), key=lambda pair: pair[1])
    num_leftover = sort_dict[1][1] - sort_dict[0][1]

    dots = []
    for _ in range(sort_dict[0][1]):
        new_pair_added = False
        while not new_pair_added:
            # one circle
            x1 = random.uniform(x_min, x_max)
            y1 = random.uniform(y_min, y_max)
            r1 = clip(random.gauss(5, 1), min_radius, max_radius)
            # get second dot coordinates
            r2 = clip(random.gauss(5, 1), min_radius, max_radius)
            theta = random.uniform(0, 2*math.pi)
            dist = r1 + r2
            tmp_x2, tmp_y2 = polar_to_cartesian(theta, dist)
            x2, y2 = tmp_x2 + x1, tmp_y2 + y1
            if no_overlap(dots, x1, y1, r1) and no_overlap(dots, x2, y2, r2):
                dots.append(Dot(x=x1, y=y1, radius=r1, color=sort_dict[0][0]))
                dots.append(Dot(x=x2, y=y2, radius=r2, color=sort_dict[1][0]))
                new_pair_added = True

    for _ in range(num_leftover):
        new_dot_added = False
        while not new_dot_added:
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            r = clip(random.gauss(5, 1), min_radius, max_radius)
            # avoid overlap with existing circles
            if no_overlap(dots, x, y, r):
                dots.append(Dot(x=x, y=y, radius=r, color=sort_dict[1][0]))
                new_dot_added = True

    return dots


def column_pairs_mixed(colors_dict, num_pixels=256, pad=5,
                       min_radius=2, max_radius=8):

    assert len(colors_dict) == 2
    sort_dict = sorted(colors_dict.items(), key=lambda pair: pair[1])
    center = num_pixels / 2
    x_vals = (center - max_radius - pad/2, center + max_radius + pad/2)
    y_step = max_radius + 2*pad
    y0 = center + int(sort_dict[1][1] / 2)*y_step
    num_lower = sort_dict[0][1]

    dots = []
    for row_num in range(sort_dict[1][1]):
        first_side = random.choice([0, 1])
        x1 = x_vals[first_side]
        y1 = y0 - row_num*y_step
        r1 = clip(random.gauss(5, 1), min_radius, max_radius)
        dots.append(Dot(x=x1, y=y1, radius=r1, color=sort_dict[1][0]))

        add_second = False
        if num_lower < sort_dict[1][1] - row_num:
            if random.random() < 0.5:
                add_second = True
        else:
            add_second = True

        if add_second:
            x2 = x_vals[0 if first_side == 1 else 1]
            y2 = y1
            r2 = clip(random.gauss(5, 1), min_radius, max_radius)
            dots.append(Dot(x=x2, y=y2, radius=r2, color=sort_dict[0][0]))
            num_lower -= 1

    return dots


def column_pairs_sorted(colors_dict, num_pixels=256, pad=5,
                        min_radius=2, max_radius=8):
    assert len(colors_dict) == 2
    sort_dict = sorted(colors_dict.items(), key=lambda pair: pair[1])
    center = num_pixels / 2
    x_vals = (center - max_radius - pad/2, center + max_radius + pad/2)
    y_step = max_radius + 2*pad
    y0 = center + int(sort_dict[1][1] / 2)*y_step
    num_lower = sort_dict[0][1]
    lower_color_side = random.choice([0, 1])
    higher_color_side = 1 if lower_color_side == 0 else 0

    dots = []
    for row_num in range(sort_dict[1][1]):
        x1 = x_vals[higher_color_side]
        y1 = y0 - row_num*y_step
        r1 = clip(random.gauss(5, 1), min_radius, max_radius)
        dots.append(Dot(x=x1, y=y1, radius=r1, color=sort_dict[1][0]))

        if row_num < num_lower:
            x2 = x_vals[lower_color_side]
            y2 = y1
            r2 = clip(random.gauss(5, 1), min_radius, max_radius)
            dots.append(Dot(x=x2, y=y2, radius=r2, color=sort_dict[0][0]))

    return dots


def make_image(file_name, dots, num_pixels=256):

    plt.rcParams['axes.facecolor'] = 'grey'
    plt.rcParams['axes.linewidth'] = 0

    fig, ax = plt.subplots(figsize=(1, 1), dpi=num_pixels)

    ax.set_xlim((0, num_pixels))
    ax.set_ylim((0, num_pixels))
    for dot in dots:
        ax.add_patch(
            Circle((dot.x, dot.y), radius=dot.radius, fc=dot.color))

    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout(pad=0)
    fig.savefig(file_name, facecolor='grey', pad_inches=0)


if __name__ == '__main__':

    test_image = [
        Dot(x=0, y=0, radius=2, color='y'),
        Dot(x=20, y=53, radius=1.5, color='b'),
        Dot(x=67, y=22, radius=4, color='y')
    ]

    make_image('test.png', column_pairs_sorted({'y': 10, 'b': 9}))
