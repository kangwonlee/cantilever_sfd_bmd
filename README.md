
# The Gaussian Integral<br>가우스 적분

* This repository contains the template code and instructions for your numerical integration assignment. You will implement numerical integration techniques to approximate the value of the Gaussian integral.<br>이 저장소는 수치 적분 과제의 시작 코드와 안내사항을 담고 있음. 해당 과제는 가우스 적분을 수치적으로 추정하는 함수를 구현할 것임.
* The Gaussian Integral is different from the **Gaussian Quadrature**.<br>가우스 적분은 **가우스 구적법**과는 다름.
* References 참고문헌: [[1](https://mathworld.wolfram.com/GaussianIntegral.html)] [[2](https://en.wikipedia.org/wiki/Gaussian_integral)] [[3](https://mathworld.wolfram.com/GaussianQuadrature.html)]

## Description<br>설명

* The Gaussian integral has no closed-form solution in terms of elementary functions.<br>해당 가우스 적분은 사칙연산과 지수 로그 함수로는 닫힌 형태의 해가 없음.
$$
\int_{a}^{b} e^{-x^2} dx
$$

* However it is known that the following definite integral has a finite value.<br>하지만 다음의 정적분은 유한한 값을 가지는 것이 알려져 있음.
$$
\int_{-\infty}^{\infty} e^{-x^2} dx=\sqrt{\pi}
$$

* Using numerical integration, please calculate the Gaussian Integral over a specified range.<br>수치 적분을 사용하여 주어진 구간에서 가우스 적분을 계산하시오.

## Implementation<br>구현

* Implement following functions in `numerical_integration.py` file.<br>다음 함수를 `numerical_integration.py` 파일에 구현하시오.

| function<br>함수 | description<br>설명 |
|:----------------:|:------------------:|
| `gauss_int_2()` | calculate the Gaussian Integration using numerical integration of 2th order<br>2차 수치 적분을 사용하여 가우스 적분을 계산 |

* Please see `sample.py` file for an example.<br>사용 예에 대해서는 `sample.py` 파일을 참고하시오.
* In `numerical_integration.py` file, every python code line must belong to one of functions.<br>`numerical_integration.py` 파일에서 모든 파이썬 코드 라인은 반드시 함수 중 하나에 속해야 함.

### 2nd order integration<br>2차 적분
* Function `gauss_int_2()` has three argument : `x_begin`, `x_end`, and `n`.<br>함수 `gauss_int_2()` 의 매개변수는 `x_begin`, `x_end`, 그리고 `n` 이다.

| argument<br>매개변수 | type<br>형 | description<br>설명 |
|:-----------------:|:----------:|:------------------:|
| `x_begin` | `float` | lower bound of the integral<br>적분구간의 하한 |
| `x_end` | `float` | upper bound of the integral<br>적분구간의 상한 |
| `n` | `int` | number of equally spaced intervals between the bounds. always an even number<br>적분 구간을 나눈 등간격 수. 항상 짝수임 |

* Please return a `dict` containing `numpy` array of the `n // 2` areas below the parabolas of 2nd order integration and the sum of all areas as the numerical integration in `float`.<br>`dict`를 반환하시오. value 로 `n // 2`개의 2차 적분의 각 포물선 아래의 넓이를 담은 배열과 해당 면적의 합을 `float`로 담으시오.

| key (str)<br>키 (문자열) | type of value<br>value 의 형 | description<br>설명 |
|:-----------------:|:----------:|:------------------:|
| `'a_array_2'` | `numpy` array | the `n // 2` areas below parabolas of 2nd order integration<br>2차 적분의 `n // 2`개의 포물선 아래의 넓이 |
| `'area_2'` | `float` | the numerical integration<br>해당 수치 적분 값 |

## Grading<br>평가

|       | points<br>배점 |
|:-----:|:-------------:|
| python grammar<br>파이썬 문법 | 2 |
| all lines of `numerical_integration.py` in the function<br>`numerical_integration.py` 파일에는 함수만 포함 | 1 |
| results<br>결과값 | 2 |

## Example<br>예

```python
x_begin = 0.0
x_end = 10.0
n = 100

print(
    gauss_int_2(
        x_begin, x_end, n
    )
)
```
