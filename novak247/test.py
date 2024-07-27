import numpy as np
import numpy.typing as npt
from scipy import integrate
from matplotlib.animation import FuncAnimation
from scipy.spatial import KDTree
import matplotlib.pyplot as plt



def dPhi_V_of(Phi: npt.ArrayLike, V: npt.ArrayLike) -> npt.ArrayLike:
    """Calculating derivative of VPF according to Phi visual angle array at a given timepoint t"""
    padV = np.pad(V, (1, 1), 'wrap')
    dPhi_V_raw = np.diff(padV)

    if dPhi_V_raw[0] > 0 and dPhi_V_raw[-1] > 0:
        dPhi_V_raw = dPhi_V_raw[0:-1]
    else:
        dPhi_V_raw = dPhi_V_raw[1:, ...]

    dPhi_V = dPhi_V_raw
    return dPhi_V

def compute_state_variables(vel_now: float, Phi: npt.ArrayLike, V_now: npt.ArrayLike,
                            GAM=None, V0=None,
                            ALP0=None, ALP1=None, ALP2=None,
                            BET0=None, BET1=None, BET2=None):
    """Calculating state variables of a given agent according to the main algorithm"""

    dt_V = np.zeros(len(Phi))
    dPhi_V = dPhi_V_of(Phi, V_now)

    G_vel = (-V_now + ALP2 * dt_V)
    G_vel_spike = np.square(dPhi_V)

    G_psi = (-V_now + BET2 * dt_V)
    G_psi_spike = np.square(dPhi_V)

    dpsi = BET0 * integrate.trapz(np.sin(Phi) * G_psi, Phi) + \
            BET0 * BET1 * np.sum(np.sin(Phi) * G_psi_spike)
        
    dvel = GAM * (V0 - vel_now) + \
            ALP0 * integrate.trapz(np.cos(Phi) * G_vel, Phi) + \
            ALP0 * ALP1 * np.sum(np.cos(Phi) * G_vel_spike)
    return dvel, dpsi

def generate_random_position(positions, min_distance, space_size):
    while True:
        new_position = np.random.rand(2) * space_size
        if positions.shape[0] == 0:
            return new_position
        tree = KDTree(positions)
        distances, _ = tree.query(new_position, k=positions.shape[0])
        if np.all(distances >= min_distance):
            return new_position

def initialize_agents(N,space_size,v):

    
    positions = np.zeros((N, 2))
    for i in range(N):
        positions[i] = generate_random_position(positions[:i], 1, space_size)


    # Initialize angles
    #angles = -np.ones(N) * np.pi/2
    angles = (np.random.uniform(-np.pi, np.pi, N))
    
    # Calculate velocities based on angles
    velocities = np.array([v * np.cos(angles), v * np.sin(angles)]).T
    #velocities = np.zeros([N,2])
    
    return positions, velocities, angles

def compute_visual_field(positions, angles, N, R):
    visual_field = np.zeros((N, 16384))
    for i in range(N):
        for j in range(N):
            if i != j:
                xj = positions[j, 0] - positions[i, 0]
                yj = positions[j, 1] - positions[i, 1]
                dij = np.sqrt(xj**2 + yj**2) 
                phij = np.arctan2(yj, xj) - angles[i]
                phij = (phij + np.pi) % (2 * np.pi) - np.pi
                delta_phij = np.arctan(R/dij)
                index_start = int((phij - delta_phij + np.pi) / (2 * np.pi) * 16384)
                index_end = int((phij + delta_phij + np.pi) / (2 * np.pi) * 16384)
                if index_start < index_end:
                    visual_field[i, index_start:index_end] = 1
                else:
                    visual_field[i, index_start:] = 1
                    visual_field[i, :index_end] = 1
                #print(index_end-index_start)
    return visual_field

def simulate_and_animate(N, gamma, alpha0, alpha1, alpha2, 
                         beta0, beta1, beta2, Tend, dt, space_size, v, R):
    positions, velocities, angles = initialize_agents(N, space_size, v)
    #print(angles,velocities)
    phi = np.linspace(-np.pi, np.pi, 16384)

    fig, ax = plt.subplots()
    scat = ax.scatter(positions[:, 0], positions[:, 1], s=50, c='blue')
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)

    def update(frame):
        nonlocal positions, velocities, angles
        visual_field = compute_visual_field(positions, angles, N, R)
        delta_phi = np.pi/8192
        for i in range(N):    
            dvel, dpsi = compute_state_variables(np.linalg.norm(velocities[i]), phi, visual_field[i], v, gamma, 
                                                 alpha0, alpha1, alpha2, beta0, beta1, beta2)
            velocities[i, 0] += dvel * np.cos(dpsi)
            velocities[i, 1] += dvel * np.sin(dpsi)
            velocities[i] = np.clip(velocities[i], [-1, -1], [1, 1])
            angles[i] += dpsi*dt
            angles = (angles + np.pi) % (2 * np.pi) - np.pi
            #print(f"Agent {i}: dvel = {dvel}")
            #print(angles)

        positions += velocities * dt
        scat.set_offsets(positions)
        return scat,
    ani = FuncAnimation(fig, update, frames= int(Tend / dt), interval=1, blit=True, repeat=False) 
    plt.show()



if __name__ == "__main__":
    gamma = 0.1
    beta0 = 10
    beta1 = 0.1
    beta2 = 0
    alpha0 = 10
    alpha1 = 0.1
    alpha2 = 0
    N = 10
    R = 0.5
    Tend = 1000
    dt = 0.1
    space_size = 10
    v = 1
    simulate_and_animate(N, gamma, alpha0, alpha1, alpha2, beta0, beta1, beta2, Tend, dt, space_size, v, R)
