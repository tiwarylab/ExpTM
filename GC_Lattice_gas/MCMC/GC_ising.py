import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import os

def initialize_spin_lattice(Nx, Ny):
    return np.random.choice([-1, 1], size=(Nx, Ny))

def get_energy(spin, eps, mu):
    Nx, Ny = spin.shape
    energy = 0
    for i in range(Nx):
        for j in range(Ny):
            energy += -(eps/4) * spin[i, j] * (spin[(i+1)%Nx, j] + spin[i, (j+1)%Ny])
            energy += -(mu + 8) * spin[i, j]
    return energy

def metropolis(spin, eps, mu, temperature):
    Nx, Ny = spin.shape
    for _ in range(Nx * Ny):
        i = np.random.randint(Nx)
        j = np.random.randint(Ny)

        # Compute delta energy from flipping spin
        current = spin[i, j]
        nn_sum = spin[(i+1)%Nx, j] + spin[(i-1)%Nx, j] + spin[i, (j+1)%Ny] + spin[i, (j-1)%Ny]
        delta_energy = 2 * current * ((eps/4) * nn_sum + (mu + 8))

        # Metropolis criterion
        if delta_energy <= 0 or np.random.rand() < np.exp(-delta_energy / temperature):
            spin[i, j] *= -1

    return spin

def calculate_num_particles(spin):
    return np.mean((spin + 1) / 2)
def initialize_spin_lattice(Nx, Ny):
    return np.random.choice([-1, 1], size=(Nx, Ny))

def get_energy(spin, eps, mu):
    Nx, Ny = spin.shape
    energy = 0
    for i in range(Nx):
        for j in range(Ny):
            energy += -(eps/4) * spin[i, j] * (spin[(i+1)%Nx, j] + spin[i, (j+1)%Ny])
            energy += -(mu + 8) * spin[i, j]
    return energy

def metropolis(spin, eps, mu, temperature):
    Nx, Ny = spin.shape
    for _ in range(Nx * Ny):
        i = np.random.randint(Nx)
        j = np.random.randint(Ny)

        # Compute delta energy from flipping spin
        current = spin[i, j]
        nn_sum = spin[(i+1)%Nx, j] + spin[(i-1)%Nx, j] + spin[i, (j+1)%Ny] + spin[i, (j-1)%Ny]
        delta_energy = 2 * current * ((eps/4) * nn_sum + (mu + 8))

        # Metropolis criterion
        if delta_energy <= 0 or np.random.rand() < np.exp(-delta_energy / temperature):
            spin[i, j] *= -1

    return spin

def calculate_num_particles(spin):
    return np.mean((spin + 1) / 2)

def simulate_ising_model(Nx, Ny, eps, mu, temperature, num_steps, equilibration_steps):
    spin = initialize_spin_lattice(Nx, Ny)

    # Equilibration
    for _ in range(equilibration_steps):
        spin = metropolis(spin, eps, mu, temperature)

    num_particles_list = []
    energy_list = []

    for _ in range(num_steps):
        spin = metropolis(spin, eps, mu, temperature)
        num_particles_list.append(calculate_num_particles(spin))
        energy_list.append(get_energy(spin, eps, mu))

    return spin, np.array(num_particles_list), np.array(energy_list)


# -------------------------------
# MAIN LOOP — RUN 10 FULL TRIALS
# -------------------------------
Nx = Ny = 8
eps = 4.0
temperature = 7.0
num_steps = 2500
equilibration_steps = 500
external_fields = np.linspace(-12, -4, 2)
n_samples = 2501

for run in range(1, 11):
    print(f"\n========== RUN {run} / 10 ==========")

    # Create directory
    data_dir = f"data_{run}"
    os.makedirs(data_dir, exist_ok=True)

    start_time = time.time()

    save_low_lattice = []
    save_high_lattice = []

    for ns in tqdm(range(n_samples)):
        frames = []
        densities = []

        for mu in external_fields:
            spin, num_particles, energy = simulate_ising_model(
                Nx, Ny, eps, mu, temperature, num_steps, equilibration_steps
            )

            densities.append(num_particles.mean())

            if mu == external_fields[0]:
                save_low_lattice.append(spin.copy())
            else:
                save_high_lattice.append(spin.copy())

        # Save every 500 samples
        if (ns + 1) % 500 == 0:
            for mu in external_fields:
                lattice_name = f"L_size_{Nx}_mu_{int(mu)}_T_{temperature}_N_{ns+1}"
                arr = save_low_lattice if mu == external_fields[0] else save_high_lattice
                np.save(f"{data_dir}/{lattice_name}.npy", np.array(arr))

    end_time = time.time()
    runtime = end_time - start_time

    print(f"Runtime for run {run}: {runtime:.2f} seconds")

    with open(f"{data_dir}/runtime.txt", "w") as f:
        f.write(f"Runtime: {runtime:.4f} seconds\n")

print("\nAll 10 runs complete!")

