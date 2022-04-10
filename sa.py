class Simulation:
    def __init__(self, model):
        self.model = model
        self.change_percentage = 0.02
        self.target_altitude = 10000

    def random_indicies(self, change_length):
        starting_index = random.choice(
            range(1, self.model.time_range_len - change_length - 1)
        )

        return (starting_index, starting_index + change_length)

    def random_thrusts(self, curr_thrusts):
        avg_thrusts = sum(curr_thrusts) / len(curr_thrusts)
        results = []
        if avg_thrusts > 0 and (avg_thrusts - 0.25 * avg_thrusts) > 0:
            results.append((avg_thrusts - 0.25 * avg_thrusts))
        if avg_thrusts < 1 and (avg_thrusts + 0.25 * avg_thrusts) < 1:
            results.append((avg_thrusts + 0.25 * avg_thrusts))

        return results

    def random_angles(self, curr_angles):
        curr_angles = np.array(curr_angles)
        avg_angles = np.sum(np.degrees(curr_angles)) / len(curr_angles)
        results = []
        if avg_angles > 0 and (avg_angles - 0.20) > 0:
            results.append(np.radians(avg_angles - 0.20))
        if avg_angles < 25 and (avg_angles + 0.20) < 25:
            results.append(np.radians(avg_angles + 0.20))
        return results

    def simulated_annealing(self, max_temp, n_steps):
        most_recent_update = None
        for step in range(n_steps):
            curr_temp = max_temp * np.exp(-step / (n_steps * 0.1))
            change_range = self.random_indicies(
                int(self.model.time_range_len * self.change_percentage)
            )
            #             curr_thrusts = self.model.T[change_range[0] : change_range[1]]
            #             curr_angles = self.model.theta_list[change_range[0] : change_range[1]]

            #             new_thurst = random.choice(self.random_thrusts(curr_thrusts))
            #             new_angle = random.choice(self.random_angles(curr_angles))

            new_thurst = random.random()
            new_angle = random.random() * 20

            results = flight.update_vectors(change_range, new_angle, new_thurst)

            if results["cruising_index"] is not None:
                old_cost = self.cost(self.model.fuel_m_list, self.model.cruising_index)
                new_cost = self.cost(results["fuel_m_list"], results["cruising_index"])

                change = abs(new_cost - old_cost)
                prob = np.exp(-change / curr_temp)
                if results["P"][-1][1] > self.model.cruise_altitude:
                    if new_cost < old_cost or random.random() < prob:
                        self.model.update_with_package(results)
                        most_recent_update = results
            if (step % 50) == 0:
                print(step)
                print(
                    self.cost(self.model.fuel_m_list, self.model.cruising_index),
                    self.model.P[self.model.cruising_index],
                    self.model.P[-1],
                )
            if step == (n_steps - 1):
                print(step)
                print(
                    self.cost(self.model.fuel_m_list, self.model.cruising_index),
                    self.model.P[self.model.cruising_index],
                    self.model.P[-1],
                )

        pickle.dump(most_recent_update, open("update_package", "wb"))

    def cost(self, fuel_m_list, idx):
        return fuel_m_list[0] - fuel_m_list[idx]
