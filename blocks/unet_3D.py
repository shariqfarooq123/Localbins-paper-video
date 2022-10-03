"""
Like unet_3D but with a 3D version of the U-Net architecture.
It uses Cuboids (Prism class) instead of Rectangles.
"""

from manim import *

class Unet3DBlock(object):
    def __init__(self):
        self.blocks = self.build_blocks()
        self.unet = VGroup(*self.blocks)

    def build_blocks(self):
        """
        Returns the VGroup of the U-Net architecture using Cuboids (Prism class).

        Its a series of Cuboids (or blocks) aligned on the horizontal axis.
        The height of the blocks decrease from max to min and then min to max to create a U shape
        The width of the blocks increase from min to max and then max to min. 
        """

        # Define the number of blocks
        n_blocks = 7

        # Define the width and height of the blocks
        block_height = [2, 1.5, 1, 0.5, 1, 1.5, 2]
        block_width = [0.5, 1, 1.5, 2, 1.5, 1, 0.5]

        # Define the background color of the blocks
        block_color = [BLUE, BLUE_E, BLUE_D, BLUE_C, BLUE_B, BLUE_A, BLUE]

        # Define the size of the blocks
        blocks = [Prism(dimensions=[block_width[i], block_height[i], block_height[i]], color=block_color[i], fill_color=block_color[i], fill_opacity=0.7) for i in range(n_blocks)]

        # Define the position of the blocks
        blocks[0].move_to(ORIGIN)
        for i in range(1, n_blocks):
            # Give the block a position relative to the previous block with a small offset
            blocks[i].next_to(blocks[i-1], RIGHT, buff=0.1)

            # Shade in 3d
            blocks[i].set_shade_in_3d(True)

        return blocks

    @property
    def bottleneck(self):
        """
        Returns the bottleneck block of the U-Net architecture
        """
        return self.blocks[3]

    @property
    def encoder(self):
        """
        Returns the encoder blocks of the U-Net architecture
        """
        return VGroup(*self.blocks[:3])

    @property
    def decoder(self):
        """
        Returns the decoder blocks of the U-Net architecture
        """
        return VGroup(*self.blocks[4:])

    @property
    def renderable(self):
        """
        Returns the VGroup of the U-Net architecture
        """
        return self.unet