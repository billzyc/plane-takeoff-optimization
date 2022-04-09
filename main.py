import copy
import math
import numpy as np
import random
import time
import pickle


class Flight:
    def __init__(self):
        # Constants
        self.g = 9.81  # m/s^2
        self.fuel_m = 20100 / 1.2  # Mass of fuel; kg
        self.m = 60000 + self.fuel_m  # kg
        self.initial_mass = 60000 + self.fuel_m  # kg
        self.flap_factor = 1.5  # Flap factor
        self.max_cl = 0.5  # Lift coefficient
        self.min_cd = 0.05  # Drag coefficient
        self.cruise_altitude = 10000  # m
        self.air_density = 1.225  # kg/m^2
        self.wing_SA = 2 * 125  # Wing surface area; m^2
        self.frontal_SA = 3.54 * 12  # Frontal surface area; m^2
        self.underbelly_SA = 42 * 12  # Underbelly surface area; m^2
        self.Vc = 258  # Critical airplane speed; m/s
        self.Vs = 343  # Speed of sound; m/s
        self.Mc = 0.78  # Critical Mach number
        self.max_T = 2 * 120000  # N
        self.SFC = 0.627  # Specific fuel consumptionl lb/(lbf.h)

        # Dynamics
        self.T = [0.0, 0.0]  # N
        self.P = [(0.0, 1.0)]  # m
        self.V = [(65.75, 5.75)]  # m/s
        self.A = [(0.0, 0.0)]  # m/s^2
        self.dt = 0.02  # s

        # Storing values
        self.cl_list = []
        self.cd_list = []
        self.theta_list = []
        self.gamma_list = []
        self.alpha_list = []
        self.SA_perp_list = []
        self.SA_parallel_list = []
        self.m_list = [self.m]
        self.fuel_m_list = [self.fuel_m]

    # Calculate temperature
    def temp(self, h):
        return (15.04 - 0.00649 * h if h < 11000 else -56.46) + 273.15

    # Calculate the change in fuel
    def fuel_consumption(self, h, V, T):
        fuel_m_dot = (
            3.338 * 10 ** (-8) * np.linalg.norm(V)
            + 1.04 * 10 ** (-5) * math.sqrt(288.15 / self.temp(h))
        ) * T
        fuel_m_change = fuel_m_dot * self.dt
        return fuel_m_change

    # Get Mach number
    def M(self, vx):
        return vx / self.Vs

    # Slope of climb
    def gamma(self, v_components):
        V = np.linalg.norm(v_components)
        return 0 if V <= 0 else np.arcsin(v_components[1] / V)

    # Angle of attack
    def alpha(self, gamma, theta):
        return theta - gamma

    # Surface area orthogonal to the speed vector
    def SA_perp(self, alpha):
        alpha = np.abs(alpha)
        return (self.wing_SA + self.underbelly_SA) * math.sin(
            alpha
        ) + self.frontal_SA * math.cos(alpha)

    # Surface area parallel to the speed vector
    def SA_parallel(self, alpha):
        alpha = np.abs(alpha)
        return (self.wing_SA + self.underbelly_SA) * math.cos(
            alpha
        ) + self.frontal_SA * math.sin(alpha)

    # Lift coefficient based on Mach number
    def mach_cl(self, cl, vx):
        M = self.M(vx)
        if M <= self.Mc:
            return cl

        Mdd = self.Mc + (1 - self.Mc) / 4
        return (
            cl + 0.1 * (M - self.Mc)
            if M <= Mdd
            else cl + 0.1 * (Mdd - self.Mc) - 0.8 * (M - Mdd)
        )

    # Drag coefficient based on Mach number
    def mach_cd(self, cd, vx):
        M = self.M(vx)
        if M < self.Mc:
            return cd / math.sqrt(1 - (M**2))
        return cd * 15 * (M - self.Mc) + cd / math.sqrt(1 - self.Mc**2)

    # Lift coefficient
    def cl(self, alpha, vx):
        alpha += np.radians(5)
        sign = np.sign(np.degrees(alpha))
        degrees = np.abs(np.degrees(alpha))

        cl = 0
        if degrees < 15:
            cl = np.degrees(alpha) * self.max_cl / 15
        elif np.abs(degrees) < 20:
            cl = np.abs((1 - (np.degrees(alpha) - 15) / 15) * self.max_cl)

        cl *= sign
        return self.mach_cl(cl, vx)

    # Drag coefficient
    def cd(self, alpha, vx):
        return self.mach_cd((np.degrees(alpha) * 0.02) ** 2 + self.min_cd, vx)

    def altitude_factor(self, Py):
        x = 1 / math.exp(Py / 7500)
        return max(0, min(1, x**0.7))

    # Drag/lift force
    def drag_lift(self, S, V, C, Py):
        return self.air_density * self.altitude_factor(Py) * C * S * V**2 / 2

    # Calculate new acceleration
    def get_acceleration(self, T, theta, v_components, mass, Py):
        # Angles
        gamma = self.gamma(v_components)
        alpha = self.alpha(gamma, theta)

        # Properties
        weight = self.g * mass
        SA_perp = self.SA_perp(alpha)
        SA_parallel = self.SA_parallel(alpha)
        cl = self.cl(alpha, v_components[0])
        cd = self.cd(alpha, v_components[0])

        # Lift and drag forces
        V = np.linalg.norm(v_components)
        Fd = self.flap_factor * self.drag_lift(SA_perp, V, cd, Py)
        Fl = self.flap_factor * self.drag_lift(SA_parallel, V, cl, Py)

        # Horizontal force
        Fx = math.cos(theta) * T - math.sin(theta) * Fl - np.abs(math.cos(gamma) * Fd)

        # Verticle force
        Fy = math.sin(theta) * T + math.cos(theta) * Fl - math.sin(gamma) * Fd - weight

        # Acceleration
        A = (Fx / mass, Fy / mass)

        # Storing values
        # self.cl_list.append(cl)
        # self.cd_list.append(cd)
        # self.gamma_list.append(gamma)
        # self.alpha_list.append(alpha)
        # self.SA_perp_list.append(SA_perp)
        # self.SA_parallel_list.append(SA_parallel)

        # Return current acceleration
        return A

    # Calculate new velocity
    def get_velocity(self, curr_velocity, acceleration):
        return (
            curr_velocity[0] + acceleration[0] * self.dt,
            curr_velocity[1] + acceleration[1] * self.dt,
        )

    # Calculate new position
    def get_position(self, p, v):
        return (p[0] + v[0] * self.dt, p[1] + v[1] * self.dt)

    # Calculate the net thrust experinced
    def percent_thrust_conversion(self, percent_thrust, Py):
        return percent_thrust * self.max_T * self.altitude_factor(Py)

    def dynamics(self, T, theta):
        T_actual = self.percent_thrust_conversion(T, self.P[-1][1])

        curr_mass = self.m_list[-1]

        self.A.append(
            self.get_acceleration(T_actual, theta, self.V[-1], curr_mass, self.P[-1][1])
        )
        self.V.append(self.get_velocity(self.V[-1], self.A[-1]))
        self.P.append(self.get_position(self.P[-1], self.V[-2]))

        fuel_change = self.fuel_consumption(self.P[-1][1], self.V[-1], T_actual)

        self.fuel_m_list.append(self.fuel_m_list[-1] - fuel_change)

        # Storing values
        self.m_list.append(curr_mass - fuel_change)

    def clear_memeory(self, index):
        self.V = self.V[: index - 1]
        self.A = self.A[: index - 1]
        self.P = self.P[: index - 1]
        self.m_list = self.m_list[: index - 1]
        self.fuel_m_list = self.fuel_m_list[: index - 1]

    def update_vectors(self, change_range, theta, thrust_percent):
        self.clear_memeory(change_range[0])

        for i in range(change_range[0], change_range[1]):
            self.T[i] = thrust_percent
            self.theta_list[i] = np.radians(theta)
        for index in range(change_range[0], self.time_range_len):
            curr_T = self.T[index]
            curr_theta = self.theta_list[index]
            self.dynamics(curr_T, curr_theta)

    def main(self, duration, theta, thrust_percent):
        time_range = np.arange(0, duration, self.dt)
        self.time_range_len = len(time_range)
        self.T = [thrust_percent] * len(time_range)
        self.theta_list = [np.radians(theta)] * len(time_range)

        for index in range(self.time_range_len - 1):
            curr_T = self.T[index]
            curr_theta = self.theta_list[index]
            self.dynamics(curr_T, curr_theta)

    def update_vectors2(self, change_index, change_theta, change_thrust):

        new_T = self.T[:]
        new_A = self.A[:]
        new_V = self.V[:]
        new_P = self.P[:]
        new_theta = self.theta_list[:]
        new_fuel_m_list = self.fuel_m_list[:]
        new_m_list = self.m_list[:]

        for i in range(change_index[0], change_index[1]):
            new_T[i] = change_thrust
            new_theta[i] = np.radians(change_theta)

        for i in range(change_index[0], len(new_T)):
            curr_thrust_percentage = new_T[i]
            curr_thrust = self.percent_thrust_conversion(
                curr_thrust_percentage, new_P[i - 1][1]
            )
            curr_theta = new_theta[i]
            new_A[i] = self.get_acceleration(
                curr_thrust,
                curr_theta,
                new_V[i - 1],
                new_m_list[i - 1],
                new_P[i - 1][1],
            )
            new_V[i] = self.get_velocity(new_V[i - 1], new_A[i - 1])
            new_P[i] = self.get_position(new_P[i - 1], new_V[i - 1])
            fuel_change = self.fuel_consumption(new_P[i][1], new_V[i], curr_thrust)
            new_fuel_m_list[i] = new_fuel_m_list[i - 1] - fuel_change
            new_m_list[i] = new_m_list[i - 1] - fuel_change

        update_package = {
            "T": new_T,
            "A": new_A,
            "V": new_V,
            "P": new_P,
            "theta_list": new_theta,
            "fuel_m_list": new_fuel_m_list,
            "m_list": new_m_list,
        }

        return update_package

    def update_with_package(self, update_package):
        self.T = update_package["T"]
        self.A = update_package["A"]
        self.V = update_package["V"]
        self.P = update_package["P"]
        self.theta_list = update_package["theta_list"]
        self.fuel_m_list = update_package["fuel_m_list"]
        self.m_list = update_package["m_list"]


class Simulation:
    def __init__(self, model):
        self.model = model
        self.change_percentage = 0.05

    def random_indicies(self, change_length):
        starting_index = random.choice(
            range(10, self.model.time_range_len - change_length - 1)
        )

        return (starting_index, starting_index + change_length)

    def simulated_annealing(self, max_temp, n_steps):
        most_recent_update = None
        for step in range(n_steps):
            curr_temp = max_temp * np.exp(-step / (n_steps * 0.1))
            change_range = self.random_indicies(
                int(self.model.time_range_len * self.change_percentage)
            )

            new_thurst = random.random()
            new_angle = random.random() * 15

            results = flight.update_vectors2(change_range, new_angle, new_thurst)

            old_cost = self.cost(self.model.fuel_m_list)
            new_cost = self.cost(results["fuel_m_list"])

            change = abs(new_cost - old_cost)
            prob = np.exp(-change / curr_temp)
            if results["P"][-1][1] > self.model.cruise_altitude:
                if new_cost < old_cost or random.random() < prob:
                    self.model.update_with_package(results)
                    most_recent_update = results
            if (step % 50) == 0:
                print(step)
                print(self.cost(self.model.fuel_m_list), self.model.P[-1])
            if step == (n_steps - 1):
                print(step)
                print(self.cost(self.model.fuel_m_list), self.model.P[-1])

        pickle.dump(most_recent_update, open("update_package", "wb"))

    def cost(self, fuel_m_list):
        return fuel_m_list[0] - fuel_m_list[-1]


flight = Flight()
flight.main(600, 5, 1)

print(flight.P[-1])
print(flight.fuel_m_list[0] - flight.fuel_m_list[-1])

s = Simulation(flight)

start_time = time.time()
s.simulated_annealing(100, 20000)
print(time.time() - start_time)


print(s.model.P[-1])
opened = pickle.load(open("update_package", "rb"))
print(opened["P"][-1])
