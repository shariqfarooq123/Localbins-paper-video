from manim import *

# Unused in this project
class TransformerBlock(object):
    def __init__(self, label_text="Transformer"):
        """
        Returns the Transformer architecture outline (VGroup) as .renderable

        The architecture has a main rectangle in the middle and small rectangles (long and thin) on the sides
        The sequence length is the number of small rectangles
        """

        # Main rectangle
        main_rect = Rectangle(width=2, height=2, color=BLUE, fill_color=BLUE, fill_opacity=1)

        # Write label on the main rectangle
        label = Text(label_text, font="Montserrat", color=WHITE)
        label.scale(0.5)
        label.move_to(main_rect.get_center())


        # Small rectangles on the left
        n_rects = 8
        rects_left = [Rectangle(width=1, height=0.2, color=BLUE, fill_color=BLUE, fill_opacity=1) for i in range(n_rects)]
        rects_left[0].move_to(ORIGIN)
        for i in range(1, n_rects):
            rects_left[i].next_to(rects_left[i-1], DOWN, buff=0.1)

        # Group the small rectangles on the left
        rects_left = VGroup(*rects_left)

        # Place the small rectangles on the left to the left of the main rectangle
        rects_left.next_to(main_rect, LEFT, buff=0.1)

        # Small rectangles on the right
        rects_right = [Rectangle(width=1, height=0.2, color=BLUE, fill_color=BLUE, fill_opacity=1) for i in range(n_rects)]
        rects_right[0].move_to(ORIGIN)
        for i in range(1, n_rects):
            rects_right[i].next_to(rects_right[i-1], DOWN, buff=0.1)

        # Group the small rectangles on the right
        rects_right = VGroup(*rects_right)

        # Place the small rectangles on the right to the right of the main rectangle
        rects_right.next_to(main_rect, RIGHT, buff=0.1)

        # Group the main rectangle and the small rectangles
        transformer = VGroup(main_rect, rects_left, rects_right, label)

        self.rects_left = rects_left
        self.rects_right = rects_right
        self.transformer = transformer

    @property
    def renderable(self):
        return self.transformer

    @property
    def input_embeddings(self):
        return self.rects_left

    @property
    def output_embeddings(self):
        return self.rects_right