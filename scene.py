"""
Main file for the project.
All the "scenes" are defined under one gigantic class "Main". This is generally considered bad practice, but is quick and easy for this project.

To generate a video, run the following command:

manim -pql scene.py Main

For high quality videos, use:

manim -pqh scene.py Main
"""
from manim import *
from blocks import Unet3DBlock, Unet2DBlock, AdaptiveBins, TransformerBlock
from utils import get_density_bars, get_labeled_rect, get_updown_arrow_between, get_right_arrow_between, get_row_vec
import numpy as np
import unreal

# filter all warnings
import warnings
warnings.filterwarnings("ignore")
 
selected_unreal = unreal.UnrealExamples()

FONT = "Montserrat"

class Main(ThreeDScene):
    def construct(self):
        self.intro()
        self.our_method()
        self.results()
        self.thank_you()

    def intro(self):
        self.title_page()
        self.main_diagram()
        self.background_page1()
        self.background_page2()
        self.background_page3()

    def our_method(self):
        self.framework_comparison()
        self.our_architecture()
        self.training()

    def results(self):
        title = Title("Results")
        self.play(FadeIn(title))
        self.show_result_images()
        self.adaptive_neighborhood_result()

    def thank_you(self):
        text = Text("Thank you for your attention!", font=FONT, color=WHITE).scale(0.5)
        self.play(FadeIn(text))
        self.wait(5)

    def title_page(self):
        PAPER_TITLE_1 = "LocalBins"
        PAPER_TITLE_2 = "Improving Depth Estimation by learning local distributions"
        PAPER_AUTHORS = "Shariq F. Bhat<sup>1</sup>, Ibraheem Alhashim<sup>2</sup>, and Peter Wonka<sup>1</sup>"
        PAPER_AFFILIATION = "<sup>1</sup>KAUST, Saudi Arabia    <sup>2</sup> SDAIA, Saudi Arabia"

        title1 = MarkupText(PAPER_TITLE_1, font=FONT, color=WHITE)
        title2 = MarkupText(PAPER_TITLE_2, font=FONT, color=WHITE)
        authors = MarkupText(PAPER_AUTHORS, font=FONT, color=WHITE)
        affiliation = MarkupText(PAPER_AFFILIATION, font=FONT, color=WHITE)

        title1.scale(0.7)
        title2.scale(0.7)
        authors.scale(0.5)
        affiliation.scale(0.5)

        title2.next_to(title1, DOWN)
        authors.next_to(title2, DOWN)
        affiliation.next_to(authors, DOWN)

        self.play(LaggedStart(Write(title1), Write(title2)))
        self.play(FadeIn(authors), run_time=1)
        self.play(FadeIn(affiliation), run_time=1)

        self.wait(2)
        self.play(FadeOut(title1), FadeOut(title2), FadeOut(authors), FadeOut(affiliation))
        self.wait(1)

    def main_diagram(self):
        model = get_labeled_rect("Model", width=2, height=0.5)

        # input_image = get_labeled_rect("Input Image", width=2, height=2)
        input_image, output_image = self.get_unreal_sample()

        input_image.next_to(model, LEFT, buff=1)

        # Arrow from input image to the model
        arrow = Arrow(input_image.get_right(), model.get_left(), buff=0.1, color=BLUE)
        arrow.set_stroke(width=2)

        # output_image = get_labeled_rect("Depth map", width=2, height=2)
        output_image.next_to(model, RIGHT, buff=1)

        # Arrow from the model to the output image
        arrow2 = Arrow(model.get_right(), output_image.get_left(), buff=0.1, color=BLUE)
        arrow2.set_stroke(width=2)

        animation = AnimationGroup(
            FadeIn(input_image),
            Create(arrow),
            FadeIn(model),
            Create(arrow2),
            FadeIn(output_image),
            lag_ratio=0.5
        )

        self.play(animation, run_time=2)
        self.wait(2)

        main_group = Group(input_image, arrow, model, arrow2, output_image)
        self.play(main_group.animate.scale(0.8).to_edge(UP, buff=0.5))
        self.wait(0.5)

        framework_text = "Adaptive bins framework of Bhat et al. (AdaBins - CVPR'21)"
        framework_text = Text(framework_text, font=FONT, color=WHITE)
        framework_text.scale(0.5)
        framework_text.next_to(main_group, DOWN, buff=0.5)

        self.play(FadeIn(framework_text))

        self.wait(5)
        return main_group, framework_text

    def obervation(self, framework_text):
        # Place three images side by side and some random plots on the bottom of each image
        images = Group()
        for i in range(3):
            # image = get_labeled_rect("Image", width=2, height=2)
            image, _ = self.get_unreal_sample()
            # image.scale(0.7)
            images.add(image)
        
        images.arrange(RIGHT, buff=0.5)
        images.next_to(framework_text, DOWN, buff=0.5)

        self.play(FadeIn(images))

        # Place the plots
        plots = VGroup()
        attractors = np.array([
            [0, 0.1, 0.4],
            [0.3, 0.5, 0.52],
            [0, 0.7, 0.8]

        ])
        points = np.random.rand(16)
        for i in range(3):
            helper_bins = AdaptiveBins(points)
            ends = [images[i].get_left()[0], images[i].get_right()[0]]
            plot = get_density_bars(helper_bins.attract_bins(attractors[i], step=0.4).bin_centers, ends=ends, max_height=1)
            plots.add(plot)

        plots.arrange(RIGHT, buff=0.5)
        plots.next_to(images, DOWN, buff=0.5)

        self.play(FadeIn(plots))

        self.wait(8)

        # Return the group of all items
        return Group(images, plots)

    def background_page1(self):
        main_group, framework_text = self.main_diagram()
        ob_out = self.obervation(framework_text)

        # FadeOut everything
        # self.play(FadeOut(main_group), FadeOut(framework_text), FadeOut(ob_out))
        self.clear_screen(1)

    def continuous_regression(self):
        centers = np.random.rand(16)
        bins = AdaptiveBins(centers).scale(4)

        # Illustrate continuous regression
        
        interval = bins.interval.copy()
        interval.set_stroke(color=BLUE)
        # # place the interval on the left
        # interval.shift(LEFT * 2)

        # Label the ends of interval as min depth and max depth
        min_depth = Text("Min depth", font=FONT, color=WHITE)
        min_depth.scale(0.5)
        min_depth.next_to(interval.get_left(), DOWN, buff=0.4)

        max_depth = Text("Max depth", font=FONT, color=WHITE)
        max_depth.scale(0.5)
        max_depth.next_to(interval.get_right(), DOWN, buff=0.4)

        label_group = VGroup(min_depth, max_depth)

        # Group 
        interval_group = VGroup(interval, label_group)

        # Show a moving arrow on top of the interval, pointing downwards to the points on interval
        arrow = Arrow(interval.get_center() + UP * 1, interval.get_center(), buff=0, color=BLUE)
        arrow.set_stroke(width=2)
        
        # show the projection point of the arrow on the interval line
        projection = Dot(arrow.get_end(), color=BLUE)
        projection.set_stroke(width=2)

        # Write the arrow label on top of the arrow as 'Depth {value}' with value from 0 to 1
        get_depth_label = lambda x: Text("Depth " + str(round(x, 2)), font=FONT, color=WHITE).scale(0.5)
        arrow_label = get_depth_label(0.5)
        arrow_label.next_to(arrow, UP, buff=0.1)

        self.play(FadeIn(interval_group), Create(arrow), FadeIn(projection), FadeIn(arrow_label))

        # Keep the projection attached according to the arrow
        projection.add_updater(lambda m: m.move_to(arrow.get_end()))

        # Keep the arrow label attached according to the arrow and update the label
        def label_updater(m):
            m.become(get_depth_label((arrow.get_end()[0] - interval.get_left()[0]) / interval.get_width()))
            m.next_to(arrow, UP, buff=0.1)

        arrow_label.add_updater(label_updater)

        # animate the arrow moving from left to right of the interval and back
        step = interval.get_right()[0] - arrow.get_center()[0]
        step = np.array([step, 0 ,0])
        arrow_anim = Succession(
            ApplyMethod(arrow.shift, step),
            ApplyMethod(arrow.shift, -2*step),
            ApplyMethod(arrow.shift, step),
        run_time=3)
        self.play(arrow_anim, run_time=3)

        # Remove the arrow and projection
        self.play(FadeOut(arrow), FadeOut(projection), FadeOut(arrow_label))

        # return the interval, arrow and projection
        return interval_group, arrow, projection, bins

    def adabins_inference_anim(self, bins, num_images=2, play=True, image_scale=0.7, real_images=False):
        anims = []
        for i in range(num_images):
            if real_images:
                fade_in_image, _ = self.get_unreal_sample()
            else:
                fade_in_image = get_labeled_rect(f"Image {i+1}", width=2, height=2)

            fade_in_image.scale(image_scale)
            fade_in_image.next_to(bins, UP, buff=0.5)

            anims.append(Succession(
                AnimationGroup(FadeIn(fade_in_image, shift=RIGHT), run_time=0.5),
                AnimationGroup(ApplyMethod(bins.attract_bins_to_random), run_time=1),
                AnimationGroup(FadeOut(fade_in_image, shift=RIGHT*2), run_time=0.5)
            ))          
            
        if play:
            for anim in anims:
                self.play(anim)
        else:
            return anims

    def adabins_training_anim(self, bins):
        # fade_in_train_img = get_labeled_rect("Input Image", width=2, height=2)
        fade_in_train_img, gt_depth = self.get_unreal_sample()
        # fade_in_train_img.scale(0.7)
        fade_in_train_img.next_to(bins, UP, buff=0.5)

        gt_points = np.random.rand(10)
        # max height distance between bottom of bins and top of gt_depth
        max_height = 0.8
        # ends are interval ends
        ends = [bins.interval_line.get_left()[0], bins.interval_line.get_right()[0]]
        dist_plot = get_density_bars(gt_points, ends=ends, max_height=max_height)
        # dist_plot.scale(0.3)
        dist_plot.next_to(bins, DOWN, buff=0.5)

        # gt_depth = get_labeled_rect("Ground Truth\n\tDepth", width=2, height=2)
        # gt_depth.scale(0.7)
        gt_depth.next_to(dist_plot, LEFT, buff=0.5).shift(DOWN*0.3)

        self.play(FadeIn(fade_in_train_img, shift=RIGHT), FadeIn(gt_depth, shift=RIGHT), run_time=0.5)
        self.play(FadeIn(dist_plot, shift=UP), run_time=0.5)
        self.play(ApplyMethod(bins.attract_bins, gt_points))
        self.wait(1)

        # Fade out image, gt_depth and plot
        self.play(FadeOut(fade_in_train_img, shift=RIGHT*2), FadeOut(gt_depth, shift=RIGHT*2), FadeOut(dist_plot, shift=DOWN*2), run_time=0.5)

    def background_page2(self):
        interval_group, arrow, projection, bins = self.continuous_regression()
        self.wait(0.5)

        self.play(FadeOut(interval_group[0]), run_time=0.5)

        self.play(Create(bins))

        self.adabins_inference_anim(bins, num_images=2, real_images=True, image_scale=1)

        self.wait(2)

        chamfer_tex = MathTex("Chamfer1D(")
        chamfer_tex.add(MathTex("bin\_centers(Img)", color=ORANGE).scale(0.7).next_to(chamfer_tex, RIGHT, buff=0.1))
        chamfer_tex.add(MathTex(",\, depth\_values(Img)", color=WHITE).scale(0.7).next_to(chamfer_tex, RIGHT, buff=0.1))
        chamfer_tex.add(MathTex(")", color=WHITE).next_to(chamfer_tex, RIGHT, buff=0.1))
        chamfer_tex.scale(0.8)
        chamfer_tex.to_corner(UL).shift(0.5*DOWN + 0.5*RIGHT)

        self.play(FadeIn(chamfer_tex, shift=RIGHT), run_time=0.5)

        self.adabins_training_anim(bins)
        self.adabins_training_anim(bins)
        self.adabins_training_anim(bins)

        self.wait(2)

        self.play(FadeOut(bins), FadeOut(chamfer_tex), FadeOut(interval_group[1:]), run_time=0.5)

    def background_page3(self):
        # Add title at the top
        title = Text("AdaBins", font=FONT, color=WHITE).scale(0.7)
        title.to_edge(UP)

        unet = Unet2DBlock().renderable
        # unet.scale(0.8)
        # Move slightly to the left
        unet.stretch_to_fit_width(5)
        unet.shift(LEFT*5)

        mvit_block = get_labeled_rect("mViT", height=unet.get_height(), width=1, color=ORANGE)
        mvit_block.next_to(unet, RIGHT, buff=0.5)

        self.play(
            FadeIn(title),
            FadeIn(unet),
            FadeIn(mvit_block),
            run_time=0.5
        )

        self.wait(3)

        bins = AdaptiveBins()
        bins.scale(3)
        bins.renderable.stretch_to_fit_width(2)
        bins.renderable.next_to(mvit_block, RIGHT, buff=0.5).shift(UP*1)

        probs = Prism([unet.get_height(), unet.get_height(), 1], color=GREEN, fill_opacity=0.5)
        probs.next_to(mvit_block, RIGHT, buff=0.5).next_to(bins.renderable, DOWN, buff=0.5)
        probs.rotate(PI/8, UP).rotate(PI/32, RIGHT)


        self.play(
            FadeIn(bins.renderable, shift=RIGHT),
            FadeIn(probs, shift=RIGHT),
            run_time=0.5
        )
        self.wait(2)

        pixel_size = probs.get_height() * 0.1
        pixel_probs = Prism([pixel_size, pixel_size, 1], color=GREEN, fill_opacity=0.9, fill_color=GREEN)
        pixel_probs.move_to(probs.get_corner(UL) + 3*pixel_size * RIGHT + 4*pixel_size * DOWN)

        self.play(FadeIn(pixel_probs), run_time=0.5)

        self.play(
            FadeOut(mvit_block),
            FadeOut(unet),
            run_time=0.5
        )

        tprobs = pixel_probs.copy()
        prob_values = np.random.rand(8)
        prob_values = prob_values / np.sum(prob_values)
        prob_row_vec = get_row_vec(prob_values, color=GREEN)

        # Place prob_row_vec at the top left of the screen
        prob_row_vec.to_edge(UL).shift(RIGHT*2 + DOWN*2)
        prob_label = Tex("p(x) = ", color=GREEN).next_to(prob_row_vec, LEFT, buff=0.1)

        self.play(
            FadeIn(prob_label)
        )


        tbins = bins.renderable.copy()
        center_values = np.random.rand(8)
        center_values = 10 * center_values / np.max(center_values)
        center_row_vec = get_row_vec(center_values, color=ORANGE)

        center_row_vec.next_to(prob_row_vec, DOWN, buff=0.5).stretch_to_fit_width(prob_row_vec.get_width())
        center_label = Tex("c(Img) = ", color=ORANGE).next_to(center_row_vec, LEFT, buff=0.1)

        self.play(
            FadeIn(center_label)
        )

        self.play(Indicate(pixel_probs, 1.7))
        self.play(Transform(tprobs, prob_row_vec))
        self.wait(1)
        self.play(Indicate(bins.renderable, 1.7))
        self.play(Transform(tbins, center_row_vec))
        self.wait(1)

        # Write depth equation as depth(x) = \sum_i p_i * c_i
        # pi is colored green and all ci is colored orange
        depth_eq = MathTex("d(x) = ", color=WHITE).next_to(center_row_vec, DOWN, buff=0.5).align_to(prob_row_vec, LEFT)
        depth_eq.add(MathTex("\\sum_{i=1} ", color=WHITE).next_to(depth_eq, RIGHT, buff=0.1))
        depth_eq.add(MathTex("p_i(x)", color=GREEN).next_to(depth_eq, RIGHT, buff=0.1))
        depth_eq.add(MathTex("c_i(Img)", color=ORANGE).next_to(depth_eq, RIGHT, buff=0.1))


        self.play(
            FadeIn(depth_eq, shift=DOWN)
        )

        self.wait(5)
        self.clear_screen()

    def localbins_inference_anim(self, bins, num_points=4, play=True):
        image = get_labeled_rect("Image", height=2, width=2)
        image.next_to(bins, UP, buff=0.5)

        self.play(FadeIn(image))
        
        # pick random points on the image
        points = np.random.rand(num_points, 3)
        points = points * 2 - 1
        points = points * 0.5
        
        # transform points to the image coordinates with origin at the center, and scale to fit the image
        points[:, 0] = points[:, 0] * image.get_width() / 2 + image.get_center()[0]
        points[:, 1] = points[:, 1] * image.get_height() / 2 + image.get_center()[1]

        # Draw the first point
        point = points[0]
        point = Dot(point, color=RED)
        self.play(FadeIn(point))
        self.play(Indicate(point, 1.7))

        # Simultaneously animate movement of the point the point to next location and animate attraction to random bins
        anims = []
        for i in range(1, num_points):
            next_point = points[i]
            next_point = Dot(next_point, color=RED)
            anims.append(AnimationGroup(ApplyMethod(point.move_to, next_point), ApplyMethod(bins.attract_bins_to_random), run_time=2))
            self.wait(0.5)

        anims.append(AnimationGroup(FadeOut(image), FadeOut(point), run_time=0.5))
        if play:
            for anim in anims:
                self.play(anim)
        else:
            return anims, image, point

    def framework_comparison(self):

        # Create a vertical separation line in the middle of the screen
        line = Line(ORIGIN, 6*UP, color=WHITE)
        line.to_edge(DOWN, buff=0.5)
        
        self.play(FadeIn(line))
        
        num_images = 4
        num_points = num_images + 1

        bins = AdaptiveBins().scale(3)
        bins.next_to(line, LEFT, buff=0.5).recalculate()

        # Text label as "AdaBins (Bhat et al.)"
        ab_label = Text("AdaBins (Bhat et al.)", font=FONT, color=WHITE).scale(0.5)
        ab_label.next_to(bins, DOWN, buff=0.5)

        self.play(FadeIn(bins), FadeIn(ab_label))

        adabins_anims = self.adabins_inference_anim(bins, play=False, num_images=num_images, image_scale=1)

        local_bins = AdaptiveBins(interval_color=GREEN).scale(3)
        local_bins.next_to(line, RIGHT, buff=0.5).recalculate()

        # Text label as "LocalBins (Ours)"
        lb_label = Text("LocalBins (Ours)", font=FONT, color=WHITE).scale(0.5)
        lb_label.next_to(local_bins, DOWN, buff=0.5)

        self.play(FadeIn(local_bins), FadeIn(lb_label))
        localbins_anims, image, point = self.localbins_inference_anim(local_bins, play=False, num_points=num_points)
        localbins_anims = localbins_anims[:-1]  # Dont fade out image and point

        # simulataneously play the two animations, until the longest one is done
        longest_anim = max(adabins_anims, localbins_anims, key=len)
        for i in range(len(longest_anim)):
            anims_to_play = []
            if i < len(adabins_anims):
                anims_to_play.append(adabins_anims[i])
            if i < len(localbins_anims):
                anims_to_play.append(localbins_anims[i])
            self.play(*anims_to_play)

        self.wait(1)

        self.play(FadeOut(bins), FadeOut(ab_label), FadeOut(line))
        localbins_group = VGroup(local_bins, lb_label, image, point)
        self.play(localbins_group.animate.move_to(ORIGIN).scale(1.2))

        neighborhood_circle = Circle(radius=0.2 * image.get_width(), color=GREEN, fill_color=GREEN, fill_opacity=0.5)
        neighborhood_circle.move_to(point)
        self.play(ReplacementTransform(local_bins.copy(), neighborhood_circle), run_time=1.5)
        self.wait(3)

        self.clear_screen()

    def our_architecture(self):
        title = "Our Architecture"
        title = Text(title, font=FONT, color=WHITE).scale(0.5)
        title.to_edge(UP, buff=0.5)

        self.play(FadeIn(title))

        unet = Unet2DBlock()
        unet.renderable.scale(0.8)
        # Move to center of screen
        unet.renderable.move_to(ORIGIN)
        unet.renderable.shift(UP*1.5)

        # Draw a outline rectangle with width from bottleneck to the last layer, labelled as "1x1 Convs"
        convs = get_labeled_rect("LocalBins module", height=unet.renderable.get_height() * 0.5, width=unet.bottleneck.get_width() + unet.decoder.get_width(), fill_opacity=0.1)
        convs.next_to(unet.renderable, DOWN, buff=1)
        # Align left of convs box with the left of the bottleneck
        convs.align_to(unet.bottleneck, LEFT)

        # Draw arrows straight down from bottleneck and decoder to the convs box
        bottleneck_arrow = get_updown_arrow_between(unet.bottleneck, convs, color=WHITE)
        decoder_arrows = []
        for decoder_layer in unet.decoder:
            decoder_arrows.append(get_updown_arrow_between(decoder_layer, convs, color=WHITE))

        all_arrows = VGroup(*([bottleneck_arrow] + decoder_arrows))

        input_image = get_labeled_rect("Input Image", height=2, width=2).scale(0.7)
        input_image.next_to(unet.renderable, LEFT, buff=0.2)

        model_group = VGroup(unet.renderable, convs, all_arrows, input_image)
        abins = AdaptiveBins().scale(3)
        bins = abins.renderable
        bins.next_to(convs, RIGHT, buff=1).shift(LEFT*2)

        binarrow = get_right_arrow_between(convs, abins.interval_ends[0], color=WHITE).shift(LEFT*2)

        self.play(AnimationGroup(FadeIn(unet.renderable), FadeIn(convs, shift=UP), lag_ratio=1, run_time=2))
        self.wait(2)
        self.play(AnimationGroup(FadeIn(input_image, shift=RIGHT), FadeIn(all_arrows, shift=DOWN), lag_ratio=1, run_time=3))
        self.wait(1)
        self.play(AnimationGroup(
            model_group.animate.shift(LEFT*2),
            FadeIn(binarrow, shift=RIGHT),
            FadeIn(bins),
            lag_ratio=0.5,
        ))

        self.wait(2)

        self.play(FadeOut(unet.renderable), FadeOut(input_image), FadeOut(binarrow), FadeOut(bins))

        to_remove_later = []
        # Draw a big outline rectangle 
        big_rect = Rectangle(height=7, width=10, fill_opacity=0.1, color=WHITE)
        big_rect.move_to(ORIGIN).shift(0.5*RIGHT)
        to_remove_later.append(big_rect)
        to_remove_later.append(all_arrows)

        # Transform the convs into big_rect but keep arrows always on top
        def arrow_updaters(mob):
            mob.next_to(convs, UP, buff=0.1)
        
        all_arrows.add_updater(arrow_updaters)
        
        self.play(FadeOut(convs[1]),
                ReplacementTransform(convs[0], big_rect),
                title.animate.to_edge(LEFT, buff=0.2),
                AnimationGroup(all_arrows.animate.stretch_to_fit_width(big_rect.get_width() - 2), lag_ratio=0.5)
                )
        all_arrows.remove_updater(arrow_updaters)
        self.play(all_arrows.animate.arrange(RIGHT, buff=1.5, center=False, aligned_edge=DOWN), run_time=0.5)
        self.play(all_arrows.animate.shift(DOWN*0.5), run_time=0.5)
        feat_rects = VGroup()
        for i in range(len(all_arrows)):
            feat_rects.add(Rectangle(height=0.5, width=2.2, fill_opacity=0.1, color=YELLOW))
            feat_rects[i].next_to(all_arrows[i], DOWN, buff=0.1)

        to_remove_later.append(feat_rects)

        self.play(*[FadeIn(rect, shift=DOWN*0.5) for rect in feat_rects], run_time=0.5)
        self.flash_label("Pixel-level features", feat_rects, wait_time=1)


        bin_embedding_rects = VGroup()
        for i in range(len(all_arrows)):
            bin_embedding_rects.add(Rectangle(height=0.5, width=2, fill_opacity=0.1, color=BLUE))
            bin_embedding_rects[i].next_to(feat_rects[i], DOWN, buff=1)

        to_remove_later.append(bin_embedding_rects)

        self.play(*[FadeIn(rect, shift=DOWN*0.5) for rect in bin_embedding_rects], run_time=0.5)
        self.flash_label("Bin embeddings", bin_embedding_rects, wait_time=3)

        centers = np.random.rand(4)
        abins = AdaptiveBins(centers)
        to_remove_later.append(abins)
        # Create splits
        split_abins = []
        root_bins = abins
        for i in range(len(all_arrows)-1):
            s = root_bins.split()
            split_abins.append(s)
            root_bins = s

        to_remove_later.append(VGroup(*split_abins))
        
        # Scale all bins
        abins.scale(1.9)
        for i in range(len(split_abins)):
            split_abins[i].scale(1.9)

        
        abins.next_to(bin_embedding_rects[0], DOWN, buff=1)
        abins.recalculate()

        self.play(FadeIn(abins, shift=DOWN))
        self.wait(4)

        splitter_boxes = VGroup()
        for i in range(1,len(all_arrows)):
            splitter_boxes.add(Rectangle(height=1, width=2, fill_opacity=0.1, color=RED))
            splitter_boxes[i-1].next_to(bin_embedding_rects[i], DOWN, buff=1)
            split_abins[i-1].move_to(splitter_boxes[i-1])

        to_remove_later.append(splitter_boxes)

        self.play(*[FadeIn(rect, shift=DOWN*0.5) for rect in splitter_boxes], run_time=0.5)
        self.wait(1)

        root = abins
        # Move splitter_boxes to the back of all objects except for 0
        for i in range(1, len(splitter_boxes)):
            # set z index
            splitter_boxes[i].set_z_index(-2)

        for i in range(len(split_abins)):
            rcopy = root.copy()
            self.play(rcopy.animate.move_to(splitter_boxes[i]))
            if i == 0:  # First one is zoomed in
                split_abins[i].scale(4)
                self.play(splitter_boxes[i].animate.scale(4).set_fill(opacity=1, color=BLACK).set_stroke(color=RED), rcopy.animate.scale(4))

            self.play(ReplacementTransform(rcopy, split_abins[i]), Circumscribe(splitter_boxes[i]), run_time=2)
            root = split_abins[i]

            if i == 0:
                self.wait(2)
                self.play(splitter_boxes[i].animate.scale(0.25).set_fill(opacity=0.1, color=RED).set_stroke(color=RED), split_abins[i].animate.scale(0.25))
            
        result_bins = split_abins[-1].copy()
        self.play(result_bins.animate.shift(DOWN*2))
        self.wait(1)

        # FadeOut everything else
        self.play(*[FadeOut(obj) for obj in to_remove_later], run_time=0.5)

        # Show the final result
        # Move the title to the center
        self.play(title.animate.to_edge(UP, buff=0.2))
        probs = Prism([2, 2, 1], color=GREEN, fill_opacity=0.5)
        probs.next_to(result_bins, UP, buff=1)
        probs.rotate(PI/8, UP).rotate(PI/32, RIGHT)

        self.play(
            FadeIn(probs, shift=RIGHT),
            run_time=0.5
        )
        self.wait(2)

        pixel_size = probs.get_height() * 0.1
        pixel_probs = Prism([pixel_size, pixel_size, 1], color=GREEN, fill_opacity=0.9, fill_color=GREEN)
        pixel_probs.move_to(probs.get_corner(UL) + 3*pixel_size * RIGHT + 4*pixel_size * DOWN)

        self.play(FadeIn(pixel_probs), run_time=0.5)
        self.play(Indicate(pixel_probs, 1.7))
        self.wait(1)

        tprobs = pixel_probs.copy()
        prob_values = np.random.rand(8)
        prob_values = prob_values / np.sum(prob_values)
        prob_row_vec = get_row_vec(prob_values, color=GREEN)

        # Place prob_row_vec at the top left of the screen
        prob_row_vec.to_edge(UL).shift(RIGHT*2 + DOWN*2)
        prob_label = Tex("p(x) = ", color=GREEN).next_to(prob_row_vec, LEFT, buff=0.1)

        self.play(
            FadeIn(prob_label)
        )

        tbins = result_bins.renderable.copy()
        center_values = np.random.rand(8)
        center_values = 10 * center_values / np.max(center_values)
        center_row_vec = get_row_vec(center_values, color=ORANGE)

        center_row_vec.next_to(prob_row_vec, DOWN, buff=0.5).stretch_to_fit_width(prob_row_vec.get_width())
        center_label = Tex("c(x) = ", color=ORANGE).next_to(center_row_vec, LEFT, buff=0.1)

        self.play(
            FadeIn(center_label)
        )

        self.play(Transform(tprobs, prob_row_vec))
        self.play(Transform(tbins, center_row_vec))

        # Write depth equation as depth(x) = \sum_i p_i * c_i
        # pi is colored green and all ci is colored orange
        depth_eq = MathTex("d(x) = ", color=WHITE).next_to(center_row_vec, DOWN, buff=0.5).align_to(prob_row_vec, LEFT)
        depth_eq.add(MathTex("\\sum_{i=1} ", color=WHITE).next_to(depth_eq, RIGHT, buff=0.1))
        depth_eq.add(MathTex("p_i(x)", color=GREEN).next_to(depth_eq, RIGHT, buff=0.1))
        depth_eq.add(MathTex("c_i(x)", color=ORANGE).next_to(depth_eq, RIGHT, buff=0.1))


        self.play(
            FadeIn(depth_eq, shift=DOWN)
        )

        self.wait(2)

        self.clear_screen()

    def what_neighborhood(self):
        input_image = get_labeled_rect("Input Image", 2, 2, color=BLUE)
        self.play(FadeIn(input_image))
        self.wait(1)
        # Text beside the image
        input_image_text = Text(
            """How to define a neighborhood?
               What size?""", color=WHITE, t2c={"size": RED, "neighborhood": RED}).scale(0.7).next_to(input_image, RIGHT, buff=-1)
        
        self.play(AnimationGroup(input_image.animate.shift(LEFT*2), FadeIn(input_image_text), lag_ratio=0.5))

        point = Dot(color=RED)
        point.move_to(input_image.get_corner(UL) + 0.5*input_image.get_width()*RIGHT + 0.5*input_image.get_height()*DOWN)
        self.play(FadeIn(point))
        self.play(Indicate(point))

        # Show the neighborhood
        init_size = 0.1 * input_image.get_width()
        neighborhood = Rectangle(width=init_size, height=init_size, color=RED, fill_opacity=0.5, fill_color=RED)
        neighborhood.move_to(point)

        self.play(FadeIn(neighborhood))
        self.wait(1)
        self.play(neighborhood.animate.scale(2))
        self.wait(1)
        self.play(neighborhood.animate.scale(2))
        self.wait(2)

        # Show a bbox on the input_image
        bbox_size = 0.2 * input_image.get_width()
        bbox = Rectangle(width=bbox_size, height=bbox_size, color=RED, fill_opacity=0.5, fill_color=RED)
        bbox.move_to(input_image.get_corner(UL) + 0.5*bbox_size*RIGHT + 0.5*bbox_size*DOWN)

        self.play(ReplacementTransform(neighborhood, bbox), FadeOut(point))

        # Chamfer tex
        chamfer_tex = Tex("Chamfer1D(c(x),", color=WHITE).scale(0.7).next_to(input_image_text, UP, buff=0.5).align_to(input_image_text, LEFT)
        chamfer_tex.set_color_by_tex("c(x)", ORANGE)
        depth_tex = Tex("depth\_values", color=RED).scale(0.7).next_to(chamfer_tex, RIGHT, buff=0.1)
        chamfer_tex.add(depth_tex)
        chamfer_tex.add(Tex(")", color=WHITE).scale(0.7).next_to(depth_tex, RIGHT, buff=0.1))

        self.play(FadeIn(chamfer_tex))


        # Slide the bbox from left to right, top to bottom. Use two for loops
        num_rows = 6
        num_cols = 6
        row_step = input_image.get_height() / num_rows
        col_step = input_image.get_width() / num_cols 

        rows_to_show = 2  # Controls the length of animation

        for i in range(num_rows - 1):
            for j in range(num_cols - 1):
                self.play(bbox.animate.shift(col_step*RIGHT), run_time=0.2, rate_func=linear)
                self.play(ReplacementTransform(bbox.copy(), depth_tex), run_time=0.5)
                # self.wait(0.1)
            self.play(bbox.animate.shift(row_step*DOWN + col_step*(num_cols-1)*LEFT))
            # self.wait(0.1)
            if i == rows_to_show - 1:
                break

        self.wait(1)

        # Fade out everything
        # self.play(FadeOut(input_image), FadeOut(input_image_text), FadeOut(chamfer_tex), FadeOut(bbox))
        self.clear_screen(1)

    def query_response_training(self):
        title = Text("Query-Response Training", color=WHITE).scale(1.5)
        self.play(FadeIn(title))
        self.wait(3)
        self.play(title.animate.to_edge(UP))

        # Show the query-response training
        input_image = get_labeled_rect("Input Image", 2, 2, color=BLUE)
        unet = Unet3DBlock()
        self.play(FadeIn(input_image))
        self.wait(1)
        self.play(ApplyMethod(input_image.shift, LEFT*4))
        unet.renderable.next_to(input_image, RIGHT, buff=0.3)
        self.play(FadeIn(unet.renderable))

        # Show the bbox
        bbox_size = 0.4 * input_image.get_width()
        bbox = Rectangle(width=bbox_size, height=bbox_size, color=RED, fill_opacity=0.5, fill_color=RED)
        get_bbox_location = lambda img : img.get_corner(UL) + 0.4*img.get_width()*RIGHT + 0.4*img.get_height()*DOWN
        bbox.move_to(get_bbox_location(input_image))

        self.play(FadeIn(bbox))
        self.play(Indicate(bbox, 1.8))
        self.wait(2)

        bbox_targets = []
        bbox_copies = []
        for block in unet.blocks[3:]:
            bbox_copies.append(bbox.copy())
            prism_copy = block.copy().scale(0.5)
            # rescale along depth to match original block
            prism_copy.rescale_to_fit(block.get_width(), 0, stretch=True)
            prism_copy.move_to(block)
            bbox_targets.append(prism_copy)
        
        # Transform bbox copies to respective targets
        anims = []
        for i in range(len(bbox_copies)):
            anims.append(ReplacementTransform(bbox_copies[i], bbox_targets[i]))

        self.play(*anims)
        self.wait(1)

        # Create pooled vectors and thin and long vertically
        pooled_vectors = VGroup(*[Rectangle(width=0.2, height=1, color=RED, fill_opacity=0.5, fill_color=RED) for _ in range(len(bbox_targets))])
        pooled_vectors.arrange(RIGHT, buff=1, aligned_edge=DOWN)
        pooled_vectors.next_to(unet.renderable, DOWN, buff=0.3).shift(LEFT*0.5)

        # Transform bbox_targets to pooled_vectors
        anims = []
        for i in range(len(bbox_targets)):
            anims.append(ReplacementTransform(bbox_targets[i], pooled_vectors[i]))
        
        self.play(*anims)
        pooled_text = Text("ROI-Align Pooled features", font=FONT).scale(0.7)
        pooled_text.next_to(pooled_vectors, LEFT, buff=0.7)
        self.play(FadeIn(pooled_text))
        self.wait(1)

        # Create localbins module
        localbins = get_labeled_rect("LocalBins Module", width=5, height=1, color=BLUE, fill_opacity=0.5)
        localbins.next_to(pooled_vectors, DOWN, buff=0.4)

        self.play(FadeIn(localbins, shift=UP))
        self.wait(1)

        result_bins = AdaptiveBins().scale(2.7)
        result_bins.renderable.next_to(localbins, RIGHT, buff=0.4)
        self.play(FadeOut(pooled_vectors, shift=DOWN), FadeOut(pooled_text),  FadeIn(result_bins.renderable, shift=RIGHT))
        self.wait(2)

        self.play(FadeOut(unet.renderable), FadeOut(localbins))

        gt_depth = get_labeled_rect("GT_depth", 2, 2, color=GREEN).next_to(input_image, RIGHT, buff=0.3)
        # Add bbox
        gt_bbox = Rectangle(width=bbox_size, height=bbox_size, color=RED, fill_opacity=0.5, fill_color=RED)
        gt_bbox.move_to(get_bbox_location(gt_depth))
        self.play(FadeIn(gt_depth), FadeIn(gt_bbox))

        gt_points = np.random.rand(6)
        distribution = get_density_bars(gt_points, align=DOWN).scale(1.2)
        distribution.next_to(gt_depth, RIGHT, buff=2).shift(UP*0.1)

        loading_circle = Circle(0.2, color=RED, fill_color=RED, fill_opacity=1).next_to(distribution, 0.8*DOWN)

        gt_bbox_copy = gt_bbox.copy()
        self.play(ReplacementTransform(gt_bbox_copy, distribution), result_bins.renderable.animate.next_to(loading_circle, 0.8*DOWN))
        self.wait(0.5)

        self.play(AnimationGroup(
            loading_circle.animate.set_color(GREEN),
            ApplyMethod(result_bins.attract_bins, gt_points),
            run_time=5
            )
        )
        self.wait(1)
        self.clear_screen()

    def training(self):
        self.what_neighborhood()
        self.query_response_training()

    def show_result_images(self):

        # NYU images
        nyu_root = os.path.join("assets", "comp_nyu")
        nyu_bases = ["comp4.jpg", "comp3.jpg", "comp2.jpg"]
        nyu_images = [ImageMobject(os.path.join(nyu_root, base)).scale_to_fit_width(config.frame_width - 0.4) for base in nyu_bases]

        nyu_label = Text("NYU Depth V2", font=FONT).scale(0.7).next_to(nyu_images[0], UP, buff=0.2)
        self.play(FadeIn(nyu_label, shift=UP))
        label_texts = VGroup(*[Tex(t).scale(0.6) for t in ["Input", "GT", "AdaBins", "Ours", "AdaBins $\Delta$", "Ours $\Delta$"]])
        label_texts.arrange(RIGHT, buff=1.5).next_to(nyu_images[0], DOWN, buff=0.5)
        self.play(FadeIn(label_texts, shift=UP))
        for im in nyu_images:
            self.play(FadeIn(im))
            self.wait(3)
            self.play(FadeOut(im))
        
        self.clear_screen()
        # iBims images
        ibims_root = os.path.join("assets", "comp_ibims")
        ibims_bases = ["comp1.jpg", "comp2.jpg"]
        ibims_images = [ImageMobject(os.path.join(ibims_root, base)).scale_to_fit_width(config.frame_width - 0.4) for base in ibims_bases]

        ibims_label = Text("iBims (Zero-shot transfer)", font=FONT).scale(0.7).next_to(ibims_images[0], UP, buff=0.2)
        self.play(FadeIn(ibims_label, shift=UP))
        label_texts.next_to(ibims_images[0], DOWN, buff=0.5)
        self.play(FadeIn(label_texts, shift=UP))
        for im in ibims_images:
            self.play(FadeIn(im))
            self.wait(3)
            self.play(FadeOut(im))

        self.clear_screen()

    def adaptive_neighborhood_result(self):

        ada_nbd_root = os.path.join("assets", "ada_nbd")
        ada_nbd_bases = ["ada_nbd1.jpg", "ada_nbd2.jpg"]
        im_width = 3
        ada_nbd_images = [ImageMobject(os.path.join(ada_nbd_root, base)).scale_to_fit_width(im_width) for base in ada_nbd_bases]
        ada_nbd_pred = [ImageMobject(os.path.join(ada_nbd_root, base.replace(".", "_gt."))).scale_to_fit_width(im_width) for base in ada_nbd_bases]

        # Arrange images in a grid
        ada_nbd_all = Group(*[Group(*[ada_nbd_images[i], ada_nbd_pred[i]]).arrange(DOWN, buff=0.5) for i in range(len(ada_nbd_images))])
        ada_nbd_all.arrange(RIGHT, buff=0.9)


        text = Text("Context-dependent\nneighborhood size", font=FONT, color=WHITE)
        self.play(Write(text))

        self.play(AnimationGroup(text.animate.scale(0.5).next_to(ada_nbd_all, LEFT, buff=0.5), FadeIn(ada_nbd_all), lag_ratio=0.5, run_time=2))

        viz_points = np.array([[0.8, 0.45, 0], [0.8, 0.75, 0]])
        # Place points on the images
        dots = []
        for i in range(len(ada_nbd_images)):
            point = viz_points[i]
            point = ada_nbd_images[i].get_corner(UL) + point*ada_nbd_images[i].get_width()*RIGHT + point*ada_nbd_images[i].get_height()*DOWN
            dot = Dot(point, color=RED, radius=0.1)
            dots.append(dot)
            self.play(FadeIn(dot))
        
        # Indicate animation on dots
        self.play(*[Flash(d) for d in dots], run_time=1)
        self.wait(1)

        im_group = Group(*ada_nbd_images, *dots)
        gt_group = Group(*ada_nbd_pred)

        # Push im_group up and slightly scale down each image. Push gt_group down and slightly scale down each image
        self.play(
            im_group.animate.shift(0.5*UP).scale(0.9),
            gt_group.animate.shift(0.5*DOWN).scale(0.9)
        )

        nbd_bins = [AdaptiveBins() for _ in range(len(ada_nbd_images))]
        anims = []
        for i in range(len(ada_nbd_images)):
            nbd_bins[i].renderable.set_width(ada_nbd_images[i].get_width())
            nbd_bins[i].renderable.next_to(ada_nbd_images[i], DOWN, buff=0.5)
            anims.append(FadeIn(nbd_bins[i].renderable))

        self.play(*anims)

        bbox_sizes = [0.7, 0.3]
        bboxes = []
        for i in range(len(ada_nbd_pred)):
            bbox = Square(side_length=bbox_sizes[i], color=BLUE_E)
            bbox.move_to(ada_nbd_pred[i].get_corner(UL) + viz_points[i]*ada_nbd_pred[i].get_width()*RIGHT + viz_points[i]*ada_nbd_pred[i].get_height()*DOWN)
            bboxes.append(bbox)
        
        anims = []
        for bins, bbox in zip(nbd_bins, bboxes):
            anims.append(ReplacementTransform(bins.renderable.copy(), bbox))
        
        self.play(*anims)

        self.wait(5)
        self.clear_screen()
            
    def flash_label(self, text, mob, wait_time=2, direction=DOWN, buff=0.1):

        feat_label = Text(text, font="Montserrat", color=WHITE).scale(0.5)
        feat_label.next_to(mob, direction, buff=buff)

        self.play(FadeIn(feat_label, shift=DOWN*0.5))
        self.wait(wait_time)
        self.play(FadeOut(feat_label))

    def get_unreal_sample(self):
        img, gt = selected_unreal.get()  # numpy arrays
        # Get manim image objects
        img = ImageMobject(img)
        gt = ImageMobject(gt)
        return img, gt

    def clear_screen(self, run_time=0.5):
        # FadeOut everything
        self.play(*[FadeOut(obj) for obj in self.mobjects], run_time=run_time)





