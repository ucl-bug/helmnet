import numpy as np
import torch


class Source:
    """Defines a (complex) monochromatic source. This is made to work easily with
    pytorch, so some outputs may have some extra dimension which appear counter-intuitive.
    """

    def __init__(
        self,
        image_size,
        omega=1,
        location=[180, 50],
        amplitude=1.0,
        phase=0.0,
        smooth=True,
    ):
        """Initializes source

        Args:
            image_size ([type]): Image dimension
            omega (int, optional): Angular frequency of the source, i.e. 2*pi*f. Defaults to 1.
            location (list, optional): Source location. Defaults to [180,50].
            amplitude ([type], optional): Source amplitude. Defaults to 1..
            phase ([type], optional): Source phase. Defaults to 0..
            smooth (bool, optional): If `True`, the source is smoothed in the spatial
                frequency domain using a Blackman window. Defaults to True.
        """
        self.L = image_size
        self.location = location
        self.t = None
        self.omega = omega
        self.amplitude = amplitude
        self.phase = phase
        self.make_abs_spatial_map(smooth=smooth)

    def make_abs_spatial_map(self, smooth=True):
        """Defines the spatial amplitude map in absolute value. This should ideally be
        a complex map if one wants to have multiple monochromatic sources, however
        for the momen we are dealing only with single point sources

        Args:
            smooth (bool, optional): If `True`, the source is smoothed in the spatial
                frequency domain using a Blackman window. Defaults to True.
        """
        # TODO: Make complex such that whatever spatial map can be defined.
        spatial_map = np.zeros((self.L, self.L))
        spatial_map[self.location[0], self.location[1]] = self.amplitude

        # Balckman smoothing in frequency
        sp_map_frequency = np.fft.fftshift(np.fft.fft2(spatial_map))
        if smooth:
            blackman = np.blackman(self.L)
            blackman_2d = np.outer(blackman, blackman)
            sp_map_frequency *= blackman_2d
        # This is a complex map and that's fine
        complex_spatial_map = np.fft.ifft2(np.fft.ifftshift(sp_map_frequency))
        self._abs_spatial_map = torch.from_numpy(np.abs(complex_spatial_map))

    def spatial_map(self, t: float):
        """Builds the complex spatial map at time t.

        Args:
            t (float): Time value

        Returns:
            torch.tensor: The source wavefield at time t.
        """
        curr_time = self.omega * t + self.phase
        with torch.no_grad():
            real = self._abs_spatial_map * np.cos(curr_time)
            imag = self._abs_spatial_map * np.sin(curr_time)
            source = torch.stack([real, imag], dim=2)
        return source.unsqueeze(0)
