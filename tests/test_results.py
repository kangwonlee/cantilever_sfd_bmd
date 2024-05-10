import random

import pytest
import numpy as np
import numpy.testing as nt

# Import the functions you'll be testing from your beam_analysis module
import beam_analysis as ba


@pytest.fixture
def beam_length_m() -> float:
    return random.uniform(0.1, 10)  # Sample beam length


@pytest.fixture
def n_interval() -> int:
    return random.randint(20, 50) * 2


@pytest.fixture
def x_m_array(beam_length_m:float, n_interval:int) -> np.ndarray:
    return np.linspace(0, beam_length_m, n_interval+1)  # Array of positions for calculation


# Parameterized tests for different load functions
@pytest.fixture(params=[
    lambda x: 1000.0,  # Constant load
    lambda x: x,       # Linear load 
    lambda x: x**2,    # Quadratic load 
    lambda x: np.sin(x)  # Sinusoidal load
])
def load_function(request):
    return request.param


@pytest.mark.parametrize("load_function", [
    lambda x: 1000.0,  # Constant load
    lambda x: x**2,  # Quadratic load
    lambda x: np.sin(x)  # Sinusoidal load
])
def test_calculate_shear_force(load_function, beam_length_m, x_m_array):
    # 1. Calculate expected SFD using analytical solutions (if possible)
    expected_sfd = ...  # Replace with your analytical calculation

    # 2. Calculate SFD using numerical integration
    calculated_sfd = ba.calculate_shear_force(x_m_array, beam_length_m, load_function)

    # 3. Assertions
    nt.assert_allclose(calculated_sfd, expected_sfd, rtol=1e-5, atol=1e-5)  # Adjust tolerances


@pytest.mark.parametrize("load_function", [
    # ... Same load function variations as above
])
def test_calculate_bending_moment(load_function, beam_length_m, x_m_array):
    # ... Similar structure to test_calculate_shear_force 

    bmd_result = ba.calculate_bending_moment(x_m_array, beam_length_m, load_function)
