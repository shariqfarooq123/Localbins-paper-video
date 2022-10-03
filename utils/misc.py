from manim import *

def get_labeled_rect(label, width=1, height=1, color=BLUE, fill_opacity=0.5, **kwargs):
    rect = Rectangle(width=width, height=height, color=color, fill_color=color, fill_opacity=fill_opacity, **kwargs)
    label = Text(label, font="Montserrat", color=WHITE)
    label.scale(0.5)
    label.move_to(rect.get_center())
    return VGroup(rect, label)

def get_updown_arrow_between(mob1, mob2, buff=0.1, **kwargs):
    arrow = Arrow(UP, DOWN, **kwargs)
    arrow.set_length(np.abs(mob1.get_bottom()[1] - mob2.get_top()[1]) - 2 * buff)
    arrow.next_to(mob1, DOWN, buff=buff)
    return arrow

def get_right_arrow_between(mob1, mob2, buff=0.1, **kwargs):
    arrow = Arrow(LEFT, RIGHT, **kwargs)
    arrow.set_length(np.abs(mob1.get_right()[0] - mob2.get_left()[0]) - 2 * buff)
    arrow.next_to(mob1, RIGHT, buff=buff)
    return arrow

def get_row_vec(values, color=WHITE):
    values = np.round(values, 2)
    values = values.reshape(1, -1)
    row_vec = Matrix(values, element_alignment_corner=UL).set_row_colors(color)
    row_vec.scale(0.5)
    return row_vec

