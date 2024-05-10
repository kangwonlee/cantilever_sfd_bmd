import matplotlib.pyplot as plt
import numpy as np


import beam_analysis as beam


def w_Npm(x_m):
    # constant distributed load function
    # 균일 분포 하중 함수
    # q(x) = 1000.0 N/m
    return 1000.0 * np.ones_like(x_m)

# if interested, please try other functions
# 관심이 있다면 다른 함수를 시도해보세요


def sample_main():
    x_begin_m = 0.0
    x_end_m = 3.0

    x_m_array = np.linspace(x_begin_m, x_end_m)

    # calculate SFD and BMD
    sfd = beam.calculate_shear_force(
        x_m_array, x_end_m, w_Npm
    )

    bmd = beam.calculate_shear_force(
        x_m_array, x_end_m, w_Npm
    )

    # plot SFD and BMD
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


if "__main__" == __name__:
    sample_main()
