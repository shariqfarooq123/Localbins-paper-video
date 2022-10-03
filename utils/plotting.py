import numpy as np
from manim import *

def get_density_bars(points, **kwargs):
    """
    Plots a bar chart of the density of the points
    Bars are colored blue to red according to the histogram value. Higher the value, redder the color
    """
    data, _ = np.histogram(points, bins=10)
    return bar_chart(data, **kwargs)

def bar_chart(bar_heights, ends=None, align=UP, max_height=None):
    """
    Custom bar chart. 
    Plots rectangles with heights given by bar_heights side by side with a gap of 0.1 between them.
    Bars are colored blue to red according to the height value. Higher the value, redder the color
    """
    bars = VGroup()
    # normalize the data
    bar_heights = bar_heights / np.max(bar_heights)
    # convert the data to a color by interpolating between blue and red
    colors = [interpolate_color(BLUE, RED, value) for value in bar_heights]
    for i in range(len(bar_heights)):
        bar = Rectangle(width=0.1, height=bar_heights[i], color=colors[i], fill_color=colors[i], fill_opacity=0.5)
        bars.add(bar)

    # scale the bars to fit in the max_height
    if max_height is not None:
        for bar in bars:
            # set bar height
            new_height = max_height * bar.get_height() / bars.get_height()
            bar.stretch_to_fit_height(new_height)
            
    # add the bars side by side with a gap of 0.1 between them
    bars.arrange(RIGHT, buff=0.1)

    # align the bars to the bottom
    for bar in bars:
        bar.align_to(bars[0], align)

    # if ends are given, spread the bars to cover from ends[0] to ends[1]
    if ends is not None:
        bars.stretch_to_fit_width(ends[1] - ends[0])
        bars.shift(np.array([(ends[1] - ends[0])/2,0,0]))



    # Add a horizontal line at the bottom covering the range of the bars
    line = Line(bars.get_left(), bars.get_right(), color=WHITE)
    line.next_to(bars, align, buff=0.05)

    return VGroup(bars, line)
