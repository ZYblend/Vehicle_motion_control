#!/usr/bin/env python3

from __future__ import division, print_function, absolute_import

import numpy as np
import pandas as pd
from math import sin, cos


class FilterClass(object):
    def __init__(self, alpha=0.98, epsilon=1, n_theta=4, wheel_base=10.1 * 0.0254, tire_radius=0.5 * 2.6 * 0.0254,
                 drive_ratio=(70.0*37.0)/(13.0*19.0), counts_max=8390000.0, cpr=720.0*4.0):
        super().__init__()
        self.parameters = {"alpha": alpha,  # forgetting factor for all recursive least squares
                           "epsilon": epsilon,  # value for initializing the covariance P_k
                           # Essentially, a smoothing factor for the resulting estimate.
                           "n_theta": n_theta,  # dimension of unknown parameter \Theta
                           "wheel_base": wheel_base,  # Wheelbase 10.1 inches
                           "tire_radius": tire_radius,  # radius of the wheels (assumed constant for front and rear)
                           "drive_ratio": drive_ratio,  # overall drive ratio
                           "counts_max": counts_max,  # maximum encoder counts for normalization
                           "cpr": cpr  # counts per revolution
                           }

        self.meas_k_1 = np.zeros((7, 1), dtype=float)  # placeholder for measurement history.
        # the measurement vector = [delta, pwm, steer, enc_count, gyr_wz, acc_ax, acc_ay]

        # Observer gain matrix update parameter
        self.P_prev = epsilon * np.eye(n_theta)
        self.P_current = np.zeros((n_theta, n_theta))

        # Estimated parameter vector
        self.Theta_hat_prev = np.zeros((n_theta, 1))
        self.Theta_hat_current = np.zeros((n_theta, 1))

        # State estimate variables: x1 = [yaw_rate, rotor_speed], x2 = [longitudinal speed, lateral speed]
        self.x1_hat_prev = np.zeros((2, 1))
        self.x1_hat_current = np.zeros((2, 1))
        self.x2_hat_prev = np.zeros((2, 1))
        self.x2_hat_current = np.zeros((2, 1))

    def test_run(self, df_in):
        vec_wz_hat = np.zeros(len(df_in))
        vec_wm_hat = np.zeros(len(df_in))
        vec_vx_hat = np.zeros(len(df_in))
        vec_vy_hat = np.zeros(len(df_in))

        idx = 0
        for row in df_in.to_numpy():
            # print(row)
            vec_wz_hat[idx], vec_wm_hat[idx], vec_vx_hat[idx], vec_vy_hat[idx] = self.run(row)
            idx += 1
        # results = [self.run(row) for row in df.to_numpy()]
        return vec_wz_hat, vec_wm_hat, vec_vx_hat, vec_vy_hat

    def run(self, meas_k, count_thresh=1.98):
        # calls filter implementations, updates necessary parameters and output filtered results
        counts2rad = 2.0 * np.pi / self.parameters["cpr"]
        d_count_k = (meas_k[3] - self.meas_k_1[3]) * counts2rad
        if np.abs(d_count_k) < (count_thresh * self.parameters["counts_max"] * counts2rad):
            # calling filter for wz_hat and wm_hat
            self.__rls_filer_gyr_enc(meas_k)

        wz_hat_out = self.x1_hat_current[0]
        wm_hat_out = self.x1_hat_current[1]

        vx_hat_out = (self.parameters['tire_radius']/self.parameters['drive_ratio'])*wm_hat_out
        vy_hat_out = 0.5*self.parameters['wheel_base']*wz_hat_out

        # updating parameters for next iterate
        self.x1_hat_prev = self.x1_hat_current
        self.P_prev = self.P_current
        self.Theta_hat_prev = self.Theta_hat_current

        self.meas_k_1 = meas_k

        return wz_hat_out[0, 0], wm_hat_out[0, 0], vx_hat_out[0, 0], vy_hat_out[0, 0]

    def __rls_filer_gyr_enc(self, meas_k):
        # Implements recursive least squares with exponential forgetting factor to
        # estimate yaw rate and encoder counts per second from gyroscope measurements and encoder counts
        #
        # Inputs:
        #   - meas_k [7-by-1 vector]: current measurement vector
        #                           ** meas_k = [delta_k = t_k - t_k_1 :  elapsed time since previous execution
        #                                        pwm_k                 :  current pwm signal (ratio btw 0 and 1)
        #                                        steer_k               :  current steering command (radians)
        #                                        enc_count_k           :  current encoder count
        #                                        gyr_wz_k              :  current gyroscope yaw rate reading (rad/sec)
        #                                        acc_ax_k              :  current accelerometer longitudinal acceleration reading (m/s^2)
        #                                        acc_ay_k              :  current accelerometer lateral acceleration reading (m/s^2)

        # Olugbenga Moses Anubi:  12/24/2021

        # **** Input preprocessing ******
        delta_k = meas_k[0] - self.meas_k_1[0]
        pwm_k_1 = self.meas_k_1[1]
        steer_k_1 = self.meas_k_1[2]
        d_count_k = (meas_k[3] - self.meas_k_1[3]) * 2.0 * np.pi / self.parameters["cpr"]  # diff counts (rad)
        # if np.abs(d_count_k) > 0.98:
        # return
        gyr_wz_k = meas_k[4]

        x1_hat_k_1 = self.x1_hat_prev

        alpha = self.parameters['alpha']

        mat_phi_bar_k = np.eye(2)
        mat_beta_bar_k = delta_k * np.eye(2)
        mat_phi_k = np.matrix(np.diag(np.array([1.0, delta_k], dtype=float)))
        mat_beta_k = np.matrix(np.diag(np.array([delta_k, 0.5 * delta_k ** 2])))
        # print('mat_beta_k',mat_beta_k.shape)
        a = mat_beta_k.T * mat_beta_k
        # print('a',a.shape)
        mat_h_k_1 = np.matrix(np.array([[-1.0 * x1_hat_k_1[0, 0], x1_hat_k_1[1, 0] * np.sin(steer_k_1), 0, 0],
                                        [0, 0, -1.0 * x1_hat_k_1[1, 0], pwm_k_1]], dtype=float))

        # **** Parameters updates ***********
        mat_g_k = self.P_prev * mat_h_k_1.T * FilterClass.inv_2by2(
            alpha * FilterClass.inv_2by2(mat_beta_k.T * mat_beta_k) + mat_h_k_1 * self.P_prev * mat_h_k_1.T
        )
        self.P_current = (self.P_prev - mat_g_k * mat_h_k_1 * self.P_prev) / alpha

        vec_z_k = np.array([[gyr_wz_k], [d_count_k]], dtype=float)

        self.Theta_hat_current = self.Theta_hat_prev + mat_g_k * (
                FilterClass.inv_2by2(mat_beta_k.T * mat_beta_k) * mat_beta_k.T * (
                 vec_z_k - mat_phi_k * x1_hat_k_1) - mat_h_k_1 * self.Theta_hat_prev)

        # **** Predict *******************
        self.x1_hat_current = mat_phi_bar_k * self.x1_hat_prev + mat_beta_bar_k * mat_h_k_1 * self.Theta_hat_current

    @staticmethod
    def inv_2by2(mat_in):
        # print(f'1, {mat_in.shape}')
        my_det = mat_in[0, 0] * mat_in[1, 1] - mat_in[1, 0] * mat_in[0, 1]
        return np.matrix(np.array([[mat_in[1, 1], -mat_in[0, 1]], [-mat_in[1, 0], mat_in[0, 0]]], dtype=float) / my_det)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    file_path = r'C:\Users\Abelb\OneDrive - Florida State Students\Attachments\Tallahassee\At FSU\projects\Automation System\Qcar\motion  control\low_level_controller\data\mix2.csv'
    df = pd.read_csv(file_path, header=None)
    df_meas = df.iloc[:, [0, 1, 2, 11, 8, 3, 4]].copy(deep=False)  # 0: time, 1:PWM, 2: steering, 11: encoder counts, 8: raw rate, 3: ax, 4: ay
    # delta_ks = np.diff(df_meas.iloc[:, 0], prepend=0)
    # df_meas.iloc[:, 0] = delta_ks
    my_filter = FilterClass(alpha=0.95)
    my_filter.Theta_hat_prev = np.array([[8.79580942e+01, 8.79580942e+01],
        [2.53996905e+00, 2.53996905e+00],
        [1.29835263e+01, 1.29835263e+01],
        [3.72696260e+04, 3.72696260e+04]])
    my_filter.meas_k_1 = df_meas.iloc[0, :]
    wz_hat, wm_hat, vx_hat, vy_hat = my_filter.test_run(df_meas.iloc[1:, :])

    t = df.iloc[1:, 0].to_numpy()
    wz_raw = df.iloc[1:, 8].to_numpy()

    # save all sensor data
    np.savetxt('sensor_data_mix2.csv', np.array([wz_hat, wm_hat, vx_hat, vy_hat]).T, delimiter=',', fmt='% s')

    fig1, ax1 = plt.subplots(2, sharex=True)
    fig1.suptitle('Yaw rate and encoder rate')
    ax1[0].plot(t, wz_raw, '.', linewidth=0.5)
    ax1[0].plot(t, wz_hat)
    ax1[1].plot(t, wm_hat)
    plt.show()

    fig2, ax2 = plt.subplots(2, sharex=True)
    ax2[0].plot(t, vx_hat)
    ax2[1].plot(t, vy_hat)

    plt.show()

    plt.plot(np.diff(df_meas.iloc[:,0]))
    plt.show()

    print("Done")
