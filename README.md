
# Cantilever Beam Analysis: Shear Force and Bending Moment Diagrams (Numerical Integration)<br>외팔보 해석: 전단력 선도와 굽힘모멘트 선도 (수치적분)

* This assignment focuses on numerically calculating and plotting shear force diagrams (SFD) and bending moment diagrams (BMD) for a cantilever beam under a distributed load represented by a callback function.<br>이 과제는 함수로 주어지는 분포 하중을 받는 외팔보의 전단력선도(SFD)와 굽힘모멘트선도(BMD)를 수치적분으로 계산하여 그리는 것을 목표로 함.

## Description<br>설명

* A cantilever beam with variable length (L) is subjected to a distributed load defined by a function `load_function()`.<br> 함수 `load_function()` 으로 주어지는 분산하중을 받고 있는 길이 `L`인 임의의 외팔보가 있다.
* It is known that the shear force diagram (SFD) and bending moment diagram (BMD) can be calculated by integrating the distributed load function.<br>분산하중 함수를 적분하여 전단력선도(SFD)와 굽힘모멘트선도(BMD)를 계산할 수 있다.

$$
SFD(x) = -\int_{0}^{x} q(\tau) d\tau
$$

$$
BMD(x) = -\int_{0}^{x} SFD(\tau) d\tau
$$

## Implementation<br>구현

* In `beam_analysis.py` file, please implement following python functions to calculate SFD and BMD numerically using appropriate integration methods.<br>`beam_analysis.py` 파일에 아래 파이썬 함수를 구현하여 SFD와 BMD를 수치적으로 계산하시오. 적절한 적분 방법을 사용하시오.


| function<br>함수 | description<br>설명 |
|:----------------:|:------------------:|
| `calculate_shear_force()` | calculate the shear force at the given positions<br>주어진 위치에서 전단력을 계산 |
| `calculate_bending_moment()` | calculate the bending moment at the given positions<br>주어진 위치에서 굽힘모멘트를 계산 |

* The functions will take `x_m_array` (position array), `L_m` (beam length), and `load_function_Npm()` as arguments.<br>해당 함수는 위치 좌표 array `x_m_array`, 보 길이 `L_m`, 그리고 `load_function_Npm()` 을 매개변수로 받음.

| argument<br>매개변수 | type<br>형 | unit<br>단위 | description<br>설명 |
|:-----------------:|:----------:|:----------:|:------------------:|
| `x_m_array` | `numpy.array` | m | array of positions from the free end where SFD and BMD are to be calculated<br>SFD, BMD 값을 구하고자 하는 자유단으로부터의 위치 좌표 array |
| `L_m` | `float` | m | length of the beam<br>보의 길이 |
| `load_function_Npm()` | `function` | N/m | load function<br>분포 하중 함수 |

* Here, the load function `load_function_Npm()` will take `x_m` as an argument and return the load in N/m at that position.<br>여기서, 분포 하중 함수 `load_function_Npm()` 는 `x_m` 을 매개변수로 받아 해당 위치에서의 하중을 N/m 단위로 반환함.

| argument<br>매개변수 | type<br>형 | unit<br>단위 | description<br>설명 |
|:-----------------:|:----------:|:----------:|:------------------:|
| `x_m` | `numpy.array` | m | position(s) from the free end where the distributed load(s) to be calculated<br>분포 하중 값을 구하고자 하는 자유단으로부터의 위치 |

* Regarding the return values of each function, please see the table below.<br>각 함수의 반환값에 대해서는 아래 표를 참고하시오.

| function<br>함수 | return value<br>반환값 | type<br>형 | unit<br>단위 |
|:----------------:|:------------------:|:------------------:|:------------------:|
| `calculate_shear_force()` | the shear force at `x_m_array`<br>`x_m_array` 위치에서 전단력 | `numpy.array` | N |
| `calculate_bending_moment()` | the bending moment at `x_m_array`<br>`x_m_array` 위치에서 굽힘모멘트 | `numpy.array` | Nm |

* Please see `sample.py` file for an example.<br>사용 예에 대해서는 `sample.py` 파일을 참고하시오.
* In `beam_analysis.py` file, every python code line must belong to one of functions.<br>`beam_analysis.py` 파일에서 모든 파이썬 코드 라인은 반드시 함수 중 하나에 속해야 함.
* Students may use `numpy`, `scipy`, or `matplotlib` for this assignment.<br>이 과제에서는 `numpy`, `scipy`, `matplotlib` 라이브러리를 사용할 수 있음.


## Grading<br>평가

|       | points<br>배점 |
|:-----:|:-------------:|
| python grammar<br>파이썬 문법 | 2 |
| all lines of `beam_analysis.py` in the function<br>`beam_analysis.py` 파일에는 함수만 포함 | 1 |
| results<br>결과값 | 2 |

## Example<br>예

```python
import matplotlib.pyplot as plt
import numpy as np
import beam_analysis as beam

x_begin = 0.0
x_end = 3.0

x_m_array = np.linspace(x_begin, x_end)

# constant distributed load function
# 균일 분포 하중 함수
# q(x) = 1000.0 N/m
def w_Npm(x_m):
    return 1000.0
# if interested, please try other functions
# 관심이 있다면 다른 함수를 시도해보세요

# calculate SFD and BMD

sfd = beam.calculate_shear_force(
    x_m_array, x_end, w_Npm
)

bmd = beam.calculate_shear_force(
    x_m_array, x_end, w_Npm
)

plt.subplot(2, 1, 1)
plt.plot(x_m_array, sfd, label='SFD')
plt.title('SFD (N)')
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(x_m_array, bmd, label='BMD')
plt.title('BMD (Nm)')
plt.xlabel('x (m)')
plt.grid()

plt.savefig('result.png')
```
