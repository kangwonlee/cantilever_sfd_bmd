import math
import pathlib
import random
import sys


from typing import Callable, Dict, Union


import numpy as np
import numpy.testing as nt
import pytest


RESULT = Dict[str, Union[float, np.ndarray]]
METHOD = Callable[[float, float, int], RESULT]
TABLE = Dict[str, np.array]


file_path = pathlib.Path(__file__)
test_folder = file_path.parent.absolute()
proj_folder = test_folder.parent.absolute()


sys.path.insert(0, str(proj_folder))


import numerical_integration


random.seed()


@pytest.fixture
def x_begin() -> float:
    return random.uniform(1e-7, 1e-5)


@pytest.fixture
def x_end() -> float:
    return random.uniform(1, 5)


@pytest.fixture
def n_interval() -> int:
    '''
    number of numerical integration intervals
    '''
    return random.randint(10, 50) * 2


@pytest.fixture
def x_array(x_begin, x_end, n_interval) -> np.array:
    return np.linspace(x_begin, x_end, (n_interval + 1))


@pytest.fixture
def f_array(x_array:np.array) -> np.array:
    return np.exp(-1.0 * (x_array ** 2))


@pytest.fixture
def c_table(f_array:np.ndarray) -> TABLE:
    return {
        '0': f_array[:-1],
        '1': np.array((1, 1)) @ np.vstack((f_array[:-1], f_array[1:])),
        '2': np.array((1, 4, 1)) @ np.vstack((f_array[:-1:2], f_array[1::2], f_array[2::2])),
    }


@pytest.fixture
def delta_x(x_array:np.ndarray) -> float:
    return x_array[1] - x_array[0]


@pytest.fixture(params=[numerical_integration.gauss_int_2])
def int_method(request):
    return request.param


@pytest.fixture
def method_number(int_method:METHOD) -> str:
    return int_method.__name__[-1]


@pytest.fixture
def c_array(c_table:TABLE, method_number:str) -> np.array:
    return c_table[method_number]


@pytest.fixture
def division(method_number:np.ndarray, delta_x:float) -> float:
    return (int(method_number) + 1) / delta_x


@pytest.fixture
def result_dict(int_method:METHOD, x_begin:float, x_end:float, n_interval:int) -> RESULT:
    return int_method(x_begin, x_end, n_interval)


@pytest.fixture
def int_function_name(method_number:str) -> str:
    return f"gauss_int_{method_number}()"


def test_result_type(result_dict:RESULT, int_function_name:str):
    assert isinstance(result_dict, dict), (
        f"{int_function_name} returned result is not a `dict`\n"
        f"{int_function_name} 반환 결과가 `dict`가 아님\n"
        f"{result_dict}"
    )


@pytest.fixture(params=["a_array", "area"])
def dict_key_prefix(request) -> str:
    return request.param


@pytest.fixture
def dict_value_type(dict_key_prefix) -> type:
    return {
        "a_array": np.ndarray,
        "area": float,
    }[dict_key_prefix]


def test_result_has_key(result_dict:RESULT, dict_key_prefix:str, method_number:str, int_function_name:str):
    dict_key = '_'.join((dict_key_prefix, method_number))

    assert dict_key in result_dict, (
        f"{int_function_name} returned result without `{dict_key}`\n"
        f"{int_function_name} 반환값에 `{dict_key}`가 없음\n"
        f"{result_dict}"
    )


@pytest.fixture
def result_a_array(result_dict:RESULT, method_number:str) -> np.ndarray:
    return result_dict[f'a_array_{method_number}']


@pytest.fixture
def result_area(result_dict:RESULT, method_number:str) -> float:
    return result_dict[f'area_{method_number}']


@pytest.fixture
def a_array_key(method_number:str) -> str:
    return 'a_array_' + method_number


@pytest.fixture
def area_key(method_number:str) -> str:
    return 'area_' + method_number


def test_a_array_type(result_a_array:np.ndarray, int_function_name:str, a_array_key:str):
    assert isinstance(result_a_array, np.ndarray,), (
        f"{int_function_name} returned result '{a_array_key}' is not an instance of `np.ndarray`\n"
        f"{int_function_name} 반환 결과 '{a_array_key}' 가 `np.ndarray`가 아님\n"
        f"{result_a_array}"
    )


@pytest.fixture
def expected_number_of_areas(method_number:str, n_interval:int) -> int:
    '''
    expected number of areas of numerical integration
    '''
    if method_number in '01':
        expected_len = (n_interval)
    elif method_number == '2':
        assert n_interval % 2 == 0
        expected_len = (n_interval // 2)
    else:
        raise NotImplementedError

    return expected_len


def test_a_array_dim(result_a_array:np.ndarray, expected_number_of_areas:int, int_function_name:str, a_array_key:str):
    assert len(result_a_array) == (expected_number_of_areas), (
        f"{int_function_name} returned result '{a_array_key}' has {len(result_a_array)} areas. expected {expected_number_of_areas}\n"
        f"{int_function_name} 반환 결과 '{a_array_key}' 에 {len(result_a_array)} 개의 넓이가 저장됨. 예상 갯수 {expected_number_of_areas}\n"
        f"{result_a_array}"
    )


def test_area_type(result_area:float, int_function_name:str, area_key:str):
    assert isinstance(result_area, float), (
        f"{int_function_name} returned result '{area_key}' is not an instance of `float`\n"
        f"{int_function_name} 반환 결과 '{area_key}'가 `float`가 아님\n"
        f"{result_area}"
    )


def test_a_array_value(
        int_function_name:str, result_a_array:np.ndarray, c_array:np.array, division:float,
        x_begin:float, x_end:float, n_interval:int
    ):
    q = result_a_array * division

    nt.assert_allclose(
        actual=q, desired=c_array,
        err_msg=(
            f'{int_function_name} : please verify the area of at each coordinate\n'
            f'{int_function_name} : 각각의 좌표값에서 넓이 계산을 확인 바랍니다\n'
            f'({x_begin} ~ {x_end}, {n_interval} intervals)'
        )
    )


def test_area_value(
        int_function_name:str, result_area:float, c_array:np.array, division:float,
        x_begin:float, x_end:float, n_interval:int
    ):
    q = result_area * division
    expected_q = np.sum(c_array)

    assert math.isclose(q, expected_q), (
        f"{int_function_name} : please verify numerical integration result\n"
        f"{int_function_name} : 적분 결과를 확인 바랍니다\n"
        f"({x_begin} ~ {x_end}, {n_interval} intervals) result = {result_area}"
    )


if "__main__" == __name__:
    pytest.main([__file__])
