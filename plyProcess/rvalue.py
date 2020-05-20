import numpy as np
import pytemperature as ptemp


class RCalculation:
    def __init__(self, T_inside_air, T_outside_air, air_velocity=1.0, epsilon=0.99):
        # air_velocity should be in meters / second
        # alpha is the convective heat transfer coefficient
        # epsilon is the thermal emissivity
        # sigma is the Stefan-Boltzmann constant
        self.alpha = 3.81*air_velocity
        self.epsilon = epsilon
        self.sigma = 5.67e-8  # W.(m^-2).(K^-4)
        self.T_inside_air = ptemp.c2k(T_inside_air)  # K
        self.T_outside_air = ptemp.c2k(T_outside_air)  # K
        self.R_values = []

    def pixel_to_temperature(self, pixel_mean, low_bound=15.0, upper_bound=35.0):
        return (pixel_mean/255)*(upper_bound-low_bound) + low_bound


    def calculate_R_value(self, T_inside_wall, T_inside_reflected):
        T_inside_wall_C = self.pixel_to_temperature(T_inside_wall)
        T_inside_reflected_C = self.pixel_to_temperature(T_inside_reflected)
        self.T_inside_wall = ptemp.c2k(T_inside_wall_C)
        self.T_inside_reflected = ptemp.c2k(T_inside_reflected_C)
        self.R = abs(self.T_inside_air - self.T_outside_air) / \
                 (self.alpha * abs(self.T_inside_air - self.T_inside_wall)
                  + self.epsilon * self.sigma * abs(self.T_inside_wall ** 4 - self.T_inside_reflected ** 4))

        return self.R

    def calculate_R_value_list(self, T_inside_walls, T_inside_reflected):
        T_inside_reflected = ptemp.c2k(T_inside_reflected)
        for T_inside_wall in T_inside_walls:
            T_inside_wall_C = self.pixel_to_temperature(T_inside_wall)
            print("The inside wall temps is ", T_inside_wall_C)
            # T_inside_reflected_C = self.pixel_to_temperature(T_inside_reflected)
            T_inside_wall_K = ptemp.c2k(T_inside_wall_C)
            R = abs(self.T_inside_air - self.T_outside_air) / \
                     (self.alpha * abs(self.T_inside_air - T_inside_wall_K)
                      + self.epsilon * self.sigma * abs(T_inside_wall_K ** 4 - T_inside_reflected ** 4))
            self.R_values.append(R)

        return self.R_values



