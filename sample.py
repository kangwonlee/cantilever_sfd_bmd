import matplotlib.pyplot as plt
import numpy as np


import numerical_integration


def sample_main():
    x_begin = 0
    x_end_array = np.logspace(-1, 1)

    area_list = []

    for x_end in x_end_array:

        n = max(int(x_end - x_begin) * 10, 10)

        result_2 = numerical_integration.gauss_int_2(x_begin, x_end, n)

        area_list.append(result_2['area_2'])

    area_array = np.array(area_list)
    going_where = (area_array * 2) ** 2

    plt.clf()

    plt.semilogx(x_end_array, going_where)
    plt.xlabel('integration range end point')
    plt.title(r'$\left(2\int_0^{x_e} exp(-x^2)dx\right)^2$')
    plt.grid(True)
    plt.savefig('sample.png')


if "__main__" == __name__:
    sample_main()
