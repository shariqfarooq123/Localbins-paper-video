from manim import *

class AdaptiveBins(VMobject):
    def __init__(self, bin_centers=None, interval_color=BLUE, bins_color=ORANGE, **kwargs):
        """
        Illustration of the adaptive bins.
        If bin_centers is not specified, then 10 bins are initialized randomly

        bin center values range from 0 to 1. bin center of value 0 is placed at the start of the interval and bin center of value 1 is placed at the end of the interval
        Plots :
        - a horizontal line segement that represents the interval. The ends are small vertical line segments.
        - a small vetical line segment for each bin center. 
        """

        # Define the number of bins
        self.n_bins = 16 if bin_centers is None else len(bin_centers)

        # Define the bin centers
        self.bin_centers = np.random.rand(self.n_bins) if bin_centers is None else bin_centers
        self.interval_color = interval_color
        self.bins_color = bins_color
        
        # Inits. They are set in the build_interval and build_bins methods
        self.interval_line = None
        self.interval_ends = None
        self.interval = None
        self.bin_centers_obj = None
        self.bins = None

        self.build_interval()
        self.build_bins()

        super().__init__(**kwargs)
        self.set_colors()

    def set_colors(self):
        """
        Set the colors of the bins and interval lines
        """
        for i in range(self.n_bins):
            self.bin_centers_obj[i].set_color(self.bins_color)
        self.interval.set_color(self.interval_color)
        return self

    def _scale_points(self, points):
        return points * self.interval_line.get_length() + self.interval_line.get_start()[0]

    def scale(self, scale_factor, **kwargs):
        """
        Scale the bin centers by the given scale factor and scale the objects
        """
        mob = super().scale(scale_factor, **kwargs)
        # Update the bin centers by the positions of the bin center line segments
        self.recalculate()        
        return mob

    def recalculate(self):
        self.bin_centers = np.array([self.bin_centers_obj[i].get_start()[0] for i in range(self.n_bins)])
        return self

    def generate_points(self):
        self.set_submobjects(self.renderable.submobjects)
        return super().generate_points()

    def build_interval(self):
        # Define the horizontal line segment that represents the interval
        self.interval_line = Line(start=np.array([0, 0, 0]), end=np.array([1, 0, 0]), color=self.interval_color)
        # set stroke width 2
        self.interval_line.set_stroke(width=2, color=self.interval_color)

        # Define the small vertical line segments at the ends of the interval
        self.interval_ends = VGroup(
                        Line(start=np.array([0, 0.1, 0]), end=np.array([0, -0.1, 0]), color=self.interval_color),
                        Line(start=np.array([1, 0.1, 0]), end=np.array([1, -0.1, 0]), color=self.interval_color)
                        )
        # set the stroke width to 2
        self.interval_ends.set_stroke(width=2)
        self.interval = VGroup(self.interval_line, self.interval_ends)

    def build_bins(self):
        self.bin_centers_obj = VGroup(*[Line(start=np.array([self.bin_centers[i], 0.05, 0]), end=np.array([self.bin_centers[i], -0.05, 0]), color=self.bins_color) for i in range(self.n_bins)])
        self.bins = VGroup(self.interval, self.bin_centers_obj)

    def attract_bins(self, points, step=0.8):
        """
        Attract the bins to the points
        """
        self.recalculate()
        points = self._scale_points(points)
        new_bin_centers = np.zeros(self.n_bins)
        
        for i in range(self.n_bins):
            # Find the closest point to the bin center
            closest_point = np.argmin(np.abs(points - self.bin_centers[i]))
            
            # Move the bin center to the closest point by fraction step
            new_bin_centers[i] = self.bin_centers[i] + step * (points[closest_point] - self.bin_centers[i])

        # shift the bin centers to the new bin centers
        return self.shift_bins(new_bin_centers)
    
    def shift_bins(self, new_bin_centers):
        """
        Shift the line segments of bins vgroup to the new bin centers
        """
        # Get the y coordinate of the current interval line
        y = self.interval_line.get_start()[1]
        for i in range(self.n_bins):
            new_pos = np.array([new_bin_centers[i], y, 0])
            self.bin_centers_obj[i].move_to(new_pos)

        self.bin_centers = new_bin_centers
        return self
    
    def attract_bins_to_random(self):
        """
        Attract the bins to random points
        """
        return self.attract_bins(np.random.rand(self.n_bins))

    def split(self):
        """
        Returns a new instance where each bin is split into two (randomly)
        """
        # assert bin centers are in range 0 to 1
        assert np.all(self.bin_centers >= 0) and np.all(self.bin_centers <= 1), "Only bins with centers in range 0 to 1 are allowed"
        new_bin_centers = np.zeros(2 * self.n_bins)
        for i in range(self.n_bins):
            new_bin_centers[2 * i] = self.bin_centers[i] - 0.1 * np.random.rand()
            new_bin_centers[2 * i + 1] = self.bin_centers[i] + 0.1 * np.random.rand()
        # clip the new bin centers to be between 0 and 1
        new_bin_centers = np.clip(new_bin_centers, 0, 1)
        return AdaptiveBins(bin_centers=new_bin_centers)

    @property
    def renderable(self):
        """
        Returns the VGroup of the renderable objects
        """
        return self.bins