import random
import pathlib
import sys
from typing import Tuple, Callable


import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nt
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.absolute()))


# Import the functions you'll be testing from your beam_analysis module
import beam_analysis as ba


SFD = Callable[[float], float]
BMD = Callable[[float], float]
LOAD = Callable[[float], float]

FUNCS = Tuple[LOAD, SFD, BMD]


@pytest.fixture
def beam_length_m() -> float:
    return random.uniform(0.1, 2)  # Sample beam length


@pytest.fixture
def n_interval() -> int:
    return random.randint(100, 200) * 2


@pytest.fixture
def x_m_array(beam_length_m:float, n_interval:int) -> np.ndarray:
    return np.linspace(0, beam_length_m, n_interval+1)  # Array of positions for calculation


@pytest.fixture
def a() -> float:
    return random.uniform(0.1, 2)


@pytest.fixture
def b() -> float:
    return random.uniform(0.1, 2)


@pytest.fixture
def c() -> float:
    return random.uniform(0.1, 2)


@pytest.fixture
def half_a(a:float) -> float:
    return 0.5*a


@pytest.fixture
def half_b(b:float) -> float:
    return 0.5*b


@pytest.fixture
def gen_case_const(a:float, half_a:float) -> FUNCS:
    # load -----------------------------
    def load(x:np.ndarray) -> np.ndarray:
        return a * np.ones_like(x)

    # sfd ------------------------------
    def sfd(x:np.ndarray) -> np.ndarray:
        return a*x

    # bmd ------------------------------
    def bmd(x:np.ndarray) -> np.ndarray:
        return half_a*(x**2)
    return load, sfd, bmd


@pytest.fixture
def gen_case_linear(
        a:float, b:float,
        half_a:float, half_b:float,
    ) -> FUNCS:
    # load -----------------------------
    def load(x:np.ndarray) -> np.ndarray:
        return a*x + b

    # sfd ------------------------------
    def sfd(x:np.ndarray) -> np.ndarray:
        return half_a*(x**2) + b*x

    a_sixth = (1.0/6.0)*a
    # bmd ------------------------------
    def bmd(x:np.ndarray) -> np.ndarray:
        return a_sixth*(x**3) + half_b*(x**2)

    return load, sfd, bmd


@pytest.fixture
def gen_case_quadratic(
        a:float, b:float, c:float,
        half_b:float,
    ) -> FUNCS:
    # load -----------------------------
    def load(x:np.ndarray) -> np.ndarray:
        return a*(x**2) + b*x + c

    a_third = (1.0/3.0)*a

    # sfd ------------------------------
    def sfd(x:np.ndarray) -> np.ndarray:
        return a_third*(x**3) + half_b*(x**2) + c*x

    a_twelfth = (1.0/12.0)*a
    b_sixth = (1.0/6.0)*b
    half_c = 0.5*c

    # bmd ------------------------------
    def bmd(x:np.ndarray) -> np.ndarray:
        return a_twelfth*(x**4) + b_sixth*(x**3) + half_c*(x**2)
    return load, sfd, bmd


@pytest.fixture
def a_b(a:float, b:float) -> float:
    return a/b


@pytest.fixture
def a_bb(a:float, b:float) -> float:
    return a/(b*b)


@pytest.fixture
def gen_case_sinusoidal(
        a:float, b:float, c:float,
        a_b:float, a_bb:float,
    ) -> FUNCS:
    # load -----------------------------
    def load(x:np.ndarray) -> np.ndarray:
        return a*np.sin(b*x + c)

    a_b_cos_c = a_b * np.cos(c)
    # sfd ------------------------------
    def sfd(x:np.ndarray) -> np.ndarray:
        return (a_b_cos_c - a_b*np.cos(b*x + c))

    a_bb_sin_c = a_bb * np.sin(c)
    # bmd ------------------------------
    def bmd(x:np.ndarray) -> np.ndarray:
        return (a_b_cos_c*x + a_bb_sin_c - a_bb*np.sin(b*x + c))

    return load, sfd, bmd


@pytest.fixture
def gen_case_exp(
        a:float, b:float, c:float,
        a_b:float, a_bb:float,
    ) -> FUNCS:
    # load -----------------------------
    def load(x:np.ndarray) -> np.ndarray:
        return a*np.exp(-b*x + c)

    a_b_exp_c = a_b * np.exp(c)
    # sfd ------------------------------
    def sfd(x:np.ndarray) -> np.ndarray:
        return (a_b_exp_c - a_b*np.exp(-b*x + c))

    a_bb_exp_c = a_bb * np.exp(c)
    # bmd ------------------------------
    def bmd(x:np.ndarray) -> np.ndarray:
        return (a_b_exp_c*x - a_bb_exp_c + a_bb*np.exp(-b*x + c))

    return load, sfd, bmd


@pytest.fixture(params=['const', 'linear', 'quadratic', 'sinusoidal', 'exp'])
def load_sfd_bmd(request, gen_case_const, gen_case_linear, gen_case_quadratic, gen_case_sinusoidal, gen_case_exp) -> Tuple[str, FUNCS]:
    d = {
        'const': gen_case_const,
        'linear': gen_case_linear,
        'quadratic': gen_case_quadratic,
        'sinusoidal': gen_case_sinusoidal,
        'exp': gen_case_exp,
    }
    return request.param, d[request.param]


def test_calculate_shear_force(load_sfd_bmd:Tuple[str, FUNCS], beam_length_m:float, x_m_array:np.ndarray):
    # 1. Calculate expected SFD using analytical solutions (if possible)
    name, (load_function, expected_sfd, _) = load_sfd_bmd

    # 2. Calculate SFD using numerical integration
    calculated_sfd = ba.calculate_shear_force(x_m_array, beam_length_m, load_function)

    # 3. Assertions
    try:
        nt.assert_allclose(calculated_sfd, expected_sfd(x_m_array), rtol=1e-5, atol=1e-5)  # Adjust tolerances
    except AssertionError as e:
        plt.clf()
        plt.plot(x_m_array, calculated_sfd, label=f'{name}calculated_sfd')
        plt.plot(x_m_array, expected_sfd(x_m_array), label=f'{name}expected_sfd')
        plt.legend(loc=0)
        plt.xlabel('x (m)')
        plt.ylabel('SFD (N)')
        plt.grid(True)
        plt.savefig(name+'.png')
        raise e


def test_calculate_bending_moment(load_sfd_bmd:Tuple[str, FUNCS], beam_length_m:float, x_m_array:np.ndarray):
    # 1. Calculate expected BMD using analytical solutions (if possible)
    name, (load_function, _, expected_bmd) = load_sfd_bmd

    # 2. Calculate BMD using numerical integration
    calculated_bmd = ba.calculate_bending_moment(x_m_array, beam_length_m, load_function)

    # 3. Assertions
    try:
        nt.assert_allclose(calculated_bmd, expected_bmd(x_m_array), rtol=1e-5, atol=1e-5)  # Adjust tolerances
    except AssertionError as e:
        plt.clf()
        plt.plot(x_m_array, calculated_bmd, label=f'{name} calculated_bmd')
        plt.plot(x_m_array, expected_bmd(x_m_array), label=f'{name} expected_bmd')
        plt.legend(loc=0)
        plt.xlabel('x (m)')
        plt.ylabel('BMD (Nm)')
        plt.grid(True)
        plt.savefig(name+'.png')
        plt.close()
        raise e


if "__main__" == __name__:
    pytest.main([__file__])
