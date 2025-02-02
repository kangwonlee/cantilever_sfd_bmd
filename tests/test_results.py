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
import exercise


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
def gen_case_sinusoidal(
        a:float, b:float, c:float,
        beam_length_m:float,
    ) -> FUNCS:
    freq = (2*np.pi*b)/beam_length_m

    # load -----------------------------
    def load(x:np.ndarray) -> np.ndarray:
        return a*np.sin(freq*x + c)

    a_freq = a/freq
    a_freq_cos_c = a_freq * np.cos(c)
    # sfd ------------------------------
    def sfd(x:np.ndarray) -> np.ndarray:
        return (a_freq_cos_c - a_freq*np.cos(freq*x + c))

    a_freq2 = a/(freq*freq)
    a_freq2_sin_c = a_freq2 * np.sin(c)
    # bmd ------------------------------
    def bmd(x:np.ndarray) -> np.ndarray:
        return (a_freq_cos_c*x + a_freq2_sin_c - a_freq2*np.sin(freq*x + c))

    return load, sfd, bmd


@pytest.fixture
def gen_case_exp(
        a:float, b:float, c:float,
    ) -> FUNCS:
    # load -----------------------------
    def load(x:np.ndarray) -> np.ndarray:
        return a*np.exp(-b*x + c)

    a_b = a/b
    a_bb = a/(b*b)
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
def load_function_type(request) -> str:
    return request.param


@pytest.fixture
def load_sfd_bmd(load_function_type, gen_case_const, gen_case_linear, gen_case_quadratic, gen_case_sinusoidal, gen_case_exp) -> FUNCS:
    d = {
        'const': gen_case_const,
        'linear': gen_case_linear,
        'quadratic': gen_case_quadratic,
        'sinusoidal': gen_case_sinusoidal,
        'exp': gen_case_exp,
    }
    return d[load_function_type]


@pytest.fixture
def result_sfd(load_sfd_bmd:FUNCS, beam_length_m:float, x_m_array:np.ndarray) -> np.ndarray:
    return exercise.calculate_shear_force(x_m_array, beam_length_m, load_sfd_bmd[0])


@pytest.fixture
def expected_sfd(load_sfd_bmd:FUNCS, x_m_array:np.ndarray) -> np.ndarray:
    return load_sfd_bmd[1](x_m_array)


def test_sfd_type(result_sfd:np.ndarray):
    assert isinstance(result_sfd, np.ndarray), (
        f"Expected np.ndarray, but got {type(result_sfd)}\n"
        f"numpy array 를 반환할 것을 예상했으나 {type(result_sfd)} 이(가) 반환됨"
    )


def test_sfd_shape(result_sfd:np.ndarray, x_m_array:np.ndarray):
    assert result_sfd.shape == x_m_array.shape, (
        f"Expected shape {x_m_array.shape}, but got {result_sfd.shape}\n"
        f"반환된 array 길이가 {x_m_array.shape} 일 것으로 예상했지만 {result_sfd.shape} (으)로 반환됨"
    )


def test_calculate_shear_force(load_function_type:str, result_sfd:np.ndarray, expected_sfd:np.ndarray, x_m_array:np.ndarray):
    try:
        nt.assert_allclose(result_sfd, expected_sfd, rtol=1e-5, atol=1e-5)  # Adjust tolerances
    except AssertionError as e:
        plt.clf()
        plt.plot(x_m_array, result_sfd, label=f'{load_function_type}calculated_sfd')
        plt.plot(x_m_array, expected_sfd, label=f'{load_function_type}expected_sfd')
        plt.legend(loc=0)
        plt.xlabel('x (m)')
        plt.ylabel('SFD (N)')
        plt.grid(True)
        plt.savefig(load_function_type+'.png')
        raise e


@pytest.fixture
def result_bmd(load_sfd_bmd:FUNCS, beam_length_m:float, x_m_array:np.ndarray) -> np.ndarray:
    return exercise.calculate_bending_moment(x_m_array, beam_length_m, load_sfd_bmd[0])


@pytest.fixture
def expected_bmd(load_sfd_bmd:Tuple[str, FUNCS], x_m_array:np.ndarray) -> np.ndarray:
    return load_sfd_bmd[2](x_m_array)


def test_bmd_type(result_bmd:np.ndarray):
    assert isinstance(result_bmd, np.ndarray), (
        f"Expected np.ndarray, but got {type(result_bmd)}\n"
        f"numpy array 를 반환할 것을 예상했으나 {type(result_bmd)} 이(가) 반환됨"
    )


def test_bmd_shape(result_bmd:np.ndarray, x_m_array:np.ndarray):
    assert result_bmd.shape == x_m_array.shape, (
        f"Expected shape {x_m_array.shape}, but got {result_sfd.shape}\n"
        f"반환된 array 길이가 {x_m_array.shape} 일 것으로 예상했지만 {result_sfd.shape} (으)로 반환됨"
    )


def test_calculate_bending_moment(load_function_type:str, result_bmd:np.ndarray, expected_bmd:np.ndarray, x_m_array:np.ndarray):
    try:
        nt.assert_allclose(result_bmd, expected_bmd, rtol=1e-5, atol=1e-5)  # Adjust tolerances
    except AssertionError as e:
        plt.clf()
        plt.plot(x_m_array, result_bmd, label=f'{load_function_type} calculated_bmd')
        plt.plot(x_m_array, expected_bmd, label=f'{load_function_type} expected_bmd')
        plt.legend(loc=0)
        plt.xlabel('x (m)')
        plt.ylabel('BMD (Nm)')
        plt.grid(True)
        plt.savefig(load_function_type+'.png')
        plt.close()
        raise e


if "__main__" == __name__:
    pytest.main([__file__])
