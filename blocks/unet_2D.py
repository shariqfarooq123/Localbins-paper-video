from manim import *


class Unet2DBlock(object):
    def __init__(self):
        self.blocks = self.build_blocks()
        self.unet = VGroup(self.encoder, self.bottleneck, self.decoder)

    def build_blocks(self):
        """
        Returns the VGroup of  the U-Net architecture using Rectangles.

        Its a series of Rectangles (or blocks) aligned on the horizontal axis.
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
        blocks = [Rectangle(width=block_width[i], height=block_height[i], color=block_color[i], fill_color=block_color[i], fill_opacity=1) for i in range(n_blocks)]

        # Define the position of the blocks
        blocks[0].move_to(ORIGIN)
        for i in range(1, n_blocks):
            # Give the block a position relative to the previous block with a small offset
            blocks[i].next_to(blocks[i-1], RIGHT, buff=0.1)
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
        return self.unet

