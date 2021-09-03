from numpy import linspace
import torch
from torch import nn
import numpy as np
import torch.fft

@torch.jit.script
def complex_mul(x, y):
    """Extend the elementwise product to complex
    tensors, i.e. tensors whose last shape has dimension of 2,
    representing real and imaginary part.

    Args:
        x (tensor): First operand
        y (tensor): Second operand
    """
    real = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    imag = x[..., 1] * y[..., 0] + x[..., 0] * y[..., 1]
    return torch.stack([real, imag], dim=-1)


@torch.jit.script
def conj(x):
    if len(x.shape) == 1:
        rx = x[0]
        ix = x[1]
    else:
        rx = x[..., 0]
        ix = x[..., 1]
    return torch.stack([rx, -ix], dim=-1)

@torch.jit.script
def fast_laplacian_with_pml(u, kx, ky, kx_sq, ky_sq, ax, bx, ay, by):
    """
    ax,bx are the 1/gamma and gamma'/gamma^3 coefficients in the laplacian operator of the paper, for the x axis
    """
    # TODO: redo this function before 9pm
    # Make 2d fourier transform of signal

    #u_fft = torch.fft(u, signal_ndim=2, normalized=False)
    #since 1.6 the fft is a module with different API
    u_fft = torch.view_as_real(
        torch.fft.fftn(
            torch.view_as_complex(u), 
            dim = (-2, -1), 
            norm = "backward"
        )
    )

    # get derivatives
    dx  = complex_mul(u_fft, kx)
    dy  = complex_mul(u_fft, ky)
    ddx = complex_mul(u_fft, kx_sq)
    ddy = complex_mul(u_fft, ky_sq)
    # derivatives = torch.ifft(
    #     torch.stack([dx, dy, ddx, ddy], dim=0),
    #     signal_ndim=2,
    #     normalized=False,
    # )
    
    derivatives = torch.view_as_real(
        torch.fft.ifftn(
            torch.view_as_complex(
                torch.stack([dx, dy, ddx, ddy], dim = 0)
            ), 
            dim = (-2, -1), 
            norm = "backward"
        )
    )

    dx = derivatives[0]
    dy = derivatives[1]
    ddx = derivatives[2]
    ddy = derivatives[3]
    return (
        complex_mul(ax, dx)
        + complex_mul(ay, dy)
        + complex_mul(bx, ddx)
        + complex_mul(by, ddy)
    )


'''
class FourierDerivative(nn.Module):
    def __init__(self, size: int, direction="x"):
        super().__init__()

        # Defining the spectral 1d operator
        k = 2 * np.pi * linspace(-0.5, 0.5, size, endpoint=False)
        k = np.concatenate((k[size // 2 :], k[: size // 2]))

        # Make it 2D on the right direction
        if direction == "x":
            kx = k
            ky = kx * 0.0
            kx, ky = np.meshgrid(kx, ky)
            k = kx
        else:
            ky = k
            kx = ky * 0.0
            kx, ky = np.meshgrid(kx, ky)
            k = ky

        k_tensor = torch.from_numpy(k).unsqueeze(0).unsqueeze(3).float()
        k_tensor = torch.cat([-k_tensor, k_tensor], dim=3)

        #  Save as parameter for automatic GPU loading, non learnable
        self.k = torch.nn.Parameter(k_tensor, requires_grad=False)

    def forward(self, x):
        """x must be [batch, x, y, real/imag]"""
        # Move to fourier basis
        Fx = torch.fft(x, signal_ndim=2, normalized=False)
        # Make derivative
        DFx = self.k * torch.flip(Fx, dims=[3])
        # Back to spatial domain
        Dx = torch.ifft(DFx, signal_ndim=2, normalized=False)
        return Dx
'''


class FourierDerivative(nn.Module):
    def __init__(self, size: int, direction="x"):
        super().__init__()

        # Defining the spectral 1d operator
        k = 2 * np.pi * np.linspace(-0.5, 0.5, size, endpoint=False)
        k = np.concatenate((k[size // 2 :], k[: size // 2]))

        # Make it 2D on the right direction
        if direction == "x":
            kx = k
            ky = kx * 0.0
            kx, ky = np.meshgrid(kx, ky)
            k = kx
        else:
            ky = k
            kx = ky * 0.0
            kx, ky = np.meshgrid(kx, ky)
            k = ky

        k_tensor = torch.from_numpy(k).unsqueeze(0).unsqueeze(3).float()
        self.k_tensor = k_tensor
        k_tensor = torch.cat([-k_tensor, k_tensor], dim=3)

        #  Save as parameter for automatic GPU loading, non learnable
        self.k = torch.nn.Parameter(k_tensor, requires_grad=False)

    def forward(self, x):
        """x must be [batch, x, y, real/imag]"""
        return torch.view_as_real(torch.fft.ifftn(
            torch.view_as_complex(torch.view_as_real(torch.fft.fftn(torch.view_as_complex(x), dim=(-2, -1), norm="backward")).flip(dims=[3]).mul(self.k)),
            dim=(-2, -1),
            norm="backward")
        )   
   
        # return torch.ifft(
        #     torch.fft(x, signal_ndim=2, normalized=False).flip(dims=[3]).mul(self.k),
        #     signal_ndim=2,
        #     normalized=False,
        # )
        x[..., 1] *= -1
        """
        # Move to fourier basis
        Fx = torch.fft(x, signal_ndim=2, normalized=False)
        # Make derivative
        DFx =  torch.flip(Fx, dims=[3]).mul(self.k)
        # Back to spatial domain
        Dx = torch.ifft(DFx, signal_ndim=2, normalized=False)
        return Dx
        """


class LaplacianWithPML(nn.Module):
    def __init__(self, domain_size: int, PMLsize: int, k: float, sigma_max: float):
        super().__init__()

        #  Settings
        self.PMLsize = PMLsize
        self.domain_size = domain_size
        self.sigma_max = sigma_max
        self.k = k

        # Calculating the gamma functions for the PML using
        # quadratic sigmas and
        # https://www.sciencedirect.com/science/article/pii/S0021999106004487
        self.gamma_x, self.gamma_y = self.get_gamma_functions()
        self.gamma_x = torch.nn.Parameter(self.gamma_x, requires_grad=False)
        self.gamma_y = torch.nn.Parameter(self.gamma_y, requires_grad=False)

        # Derivative operators
        self.dx = FourierDerivative(size=domain_size, direction="x")
        self.dy = FourierDerivative(size=domain_size, direction="y")

    def pure_derivatives(self, f):
        # X direction
        dx = self.dx(f)
        dy = self.dy(f)
        return dx, dy

    def sigmas(self):
        return self.sigma_x, self.sigma_y

    def get_gamma_functions(self):
        """Builds the gamma functions for the PML

        Returns:
            torch.tensor, torch.tensor: The gamma_x and gamma_y required by the PML
        """
        pml_coord = np.arange(self.PMLsize)
        sigma_outer = self.sigma_max * (np.abs(1 - pml_coord / self.PMLsize) ** 2)
        sigma = np.zeros((self.domain_size,))
        sigma[: self.PMLsize] = sigma_outer
        sigma[-self.PMLsize :] = np.flip(sigma_outer)
        sigma_x, sigma_y = np.meshgrid(sigma, sigma)
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

        # Making gammas
        gamma_x = 1.0 / (np.ones_like(sigma_x) + (1j / self.k) * sigma_x)
        gamma_y = 1.0 / (np.ones_like(sigma_y) + (1j / self.k) * sigma_y)

        # Turning into tensors
        real = torch.from_numpy(np.real(gamma_x))
        imag = torch.from_numpy(np.imag(gamma_x))
        gamma_x = torch.stack([real, imag], dim=-1).unsqueeze(0)

        real = torch.from_numpy(np.real(gamma_y))
        imag = torch.from_numpy(np.imag(gamma_y))
        gamma_y = torch.stack([real, imag], dim=-1).unsqueeze(0)

        # Return
        return gamma_x.float(), gamma_y.float()

    def forward(self, f):
        # X direction
        gx_f = complex_mul(self.gamma_x, self.dx(f))
        gxgx_f = complex_mul(self.gamma_x, self.dx(gx_f))

        # Y direction
        gy_f = complex_mul(self.gamma_y, self.dy(f))
        gygy_f = complex_mul(self.gamma_y, self.dy(gy_f))

        return gxgx_f + gygy_f


class FastLaplacianWithPML(nn.Module):
    def __init__(self, domain_size: int, PMLsize: int, k: float, sigma_max: float):
        super().__init__()
        self.init_variables(PMLsize, domain_size, sigma_max, k)

    def forward(self, x):
        return fast_laplacian_with_pml(
            x,
            self.kx,
            self.ky,
            self.kx_sq,
            self.ky_sq,
            self.ax,
            self.bx,
            self.ay,
            self.by,
        )

    def sigmas(self):
        return self.sigma_x, self.sigma_y

    def init_variables(self, PMLsize, domain_size, sigma_max, k):
        #  Settings
        self.PMLsize = PMLsize
        self.domain_size = domain_size
        self.sigma_max = sigma_max
        self.k = k

        self.get_gamma_functions()

        # Derivative operators in fourier domain
        self.dx = FourierDerivative(size=domain_size, direction="x")
        self.dy = FourierDerivative(size=domain_size, direction="y")
        kx = self.dx.k_tensor
        ky = self.dy.k_tensor
        kx_sq = kx.pow(2)
        ky_sq = ky.pow(2)
        zeros = torch.zeros_like(kx)
        kx = torch.cat([zeros, kx], dim=-1)  # kx is imaginary
        ky = torch.cat([zeros, ky], dim=-1)
        kx_sq = torch.cat([-kx_sq, zeros], dim=-1)  # k_sq is negated
        ky_sq = torch.cat([-ky_sq, zeros], dim=-1)

        self.kx = torch.nn.Parameter(kx, requires_grad=False)
        self.ky = torch.nn.Parameter(ky, requires_grad=False)
        self.kx_sq = torch.nn.Parameter(kx_sq, requires_grad=False)
        self.ky_sq = torch.nn.Parameter(ky_sq, requires_grad=False)

        # Gamma functions
        del self.dx
        del self.dy

    def get_gamma_functions(self):
        """Builds the gamma functions for the PML using
        quadratic sigmas
        https://www.sciencedirect.com/science/article/pii/S0021999106004487

        Returns:
            torch.tensor, torch.tensor: The gamma_x and gamma_y required by the PML
        """
        # Constructing sigmas
        pml_coord = np.arange(self.PMLsize)
        sigma_outer = self.sigma_max * (np.abs(1 - pml_coord / self.PMLsize) ** 2)
        sigma = np.zeros((self.domain_size,))
        sigma[: self.PMLsize] = sigma_outer
        sigma[-self.PMLsize :] = np.flip(sigma_outer)
        sigma_x, sigma_y = np.meshgrid(sigma, sigma)
        self.sigma_x = torch.tensor(sigma_x).float()
        self.sigma_y = torch.tensor(sigma_y).float()

        # Making inverse gammas
        inv_gamma_x = 1.0 / (
            np.ones_like(sigma_x) + (1j / self.k) * sigma_x
        )  # TODO: this works because w=c0=k=1
        inv_gamma_y = 1.0 / (np.ones_like(sigma_y) + (1j / self.k) * sigma_y)

        # Making gamma_prime
        sigma_prime = (
            -2 * self.sigma_max * (1 - pml_coord / self.PMLsize) / self.PMLsize
        )
        sigma = np.zeros((self.domain_size,))
        sigma[: self.PMLsize] = sigma_prime
        sigma[-self.PMLsize :] = -np.flip(sigma_prime)
        sigma_x_prime, sigma_y_prime = np.meshgrid(sigma, sigma)
        gamma_x_prime = (1j / self.k) * sigma_x_prime
        gamma_y_prime = (1j / self.k) * sigma_y_prime

        # Making coefficients for the modified laplacian as
        # L = ax dx' + bx dx'' + ay dy' + by dy''
        self.ax = -gamma_x_prime * (inv_gamma_x ** 3)
        self.bx = inv_gamma_x ** 2
        self.ay = -gamma_y_prime * (inv_gamma_y ** 3)
        self.by = inv_gamma_y ** 2

        # Turning into tensors
        real = torch.from_numpy(np.real(self.ax))
        imag = torch.from_numpy(np.imag(self.ax))
        self.ax = torch.stack([real, imag], dim=-1).unsqueeze(0).float()

        real = torch.from_numpy(np.real(self.bx))
        imag = torch.from_numpy(np.imag(self.bx))
        self.bx = torch.stack([real, imag], dim=-1).unsqueeze(0).float()

        real = torch.from_numpy(np.real(self.ay))
        imag = torch.from_numpy(np.imag(self.ay))
        self.ay = torch.stack([real, imag], dim=-1).unsqueeze(0).float()

        real = torch.from_numpy(np.real(self.by))
        imag = torch.from_numpy(np.imag(self.by))
        self.by = torch.stack([real, imag], dim=-1).unsqueeze(0).float()

        # Make them parameters for automatic device assignment
        self.sigma_x = torch.nn.Parameter(self.sigma_x, requires_grad=False)
        self.sigma_y = torch.nn.Parameter(self.sigma_y, requires_grad=False)
        self.ax = torch.nn.Parameter(self.ax, requires_grad=False)
        self.bx = torch.nn.Parameter(self.bx, requires_grad=False)
        self.ay = torch.nn.Parameter(self.ay, requires_grad=False)
        self.by = torch.nn.Parameter(self.by, requires_grad=False)
