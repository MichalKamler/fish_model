import pygame
import pymunk
import pymunk.pygame_util
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

WIDTH, HEIGHT = 2900, 1500
window = pygame.display.set_mode((WIDTH, HEIGHT))
n_balls = 2
RADIUS = 10
PREFERED_DIR = [0, 0]
CONST_VEL_OF_DIR = 3
RECENTER = True
MAX_VEL = 25
RANDOM_FIRST_HEADING = True

def draw(space, window, draw_options, balls):
    window.fill("white")
    space.debug_draw(draw_options)

    for ball in balls:
        start_pos = ball.body.position
        velocity = ball.body.velocity
        end_pos = start_pos + 10 * velocity
        pygame.draw.line(window, (0, 255, 0), start_pos, end_pos, 2)
     
    ball = balls[0]

    # Display velocity
    velocity = ball.body.velocity
    font = pygame.font.Font(None, 36)
    velocity_text = f"Velocity: ({velocity.x:.2f}, {velocity.y:.2f})"
    text = font.render(velocity_text, True, (0, 0, 0))
    window.blit(text, (10, 10))

    ball2 = balls[1]

    velocity = ball2.body.velocity
    velocity_text = f"Velocity: ({velocity.x:.2f}, {velocity.y:.2f})"
    text = font.render(velocity_text, True, (0, 0, 0))
    window.blit(text, (10, 50))

    # # Display force
    # force = ball.body.force
    # force_text = f"Force: ({force[0]:.2f}, {force[1]:.2f})"
    # text_force = font.render(force_text, True, (0, 0, 0))
    # window.blit(text_force, (10, 50))

    pygame.display.update()

def create_n_balls(n, space, radius, mass):
    balls = []
    center_x = WIDTH // 2
    center_y = HEIGHT // 2
    step_distance = 6*radius  # Distance between each ball
    directions = [(1, 0), (0, -1), (-1, 0), (0, 1)]  # Right, Up, Left, Down
    current_direction = 0  # Start moving to the right
    current_x, current_y = center_x, center_y
    steps_in_current_direction = 1  # Initial steps in the current direction
    steps_taken = 0
    changes_in_direction = 0

    for i in range(n):
        ball = create_ball(space, radius, mass, [current_x, current_y])
        balls.append(ball)

        # Move to the next position in the snail-like pattern
        current_x += step_distance * directions[current_direction][0]#+np.random.randint(3*RADIUS)
        current_y += step_distance * directions[current_direction][1]
        steps_taken += 1

        if steps_taken == steps_in_current_direction:
            steps_taken = 0
            current_direction = (current_direction + 1) % 4
            changes_in_direction += 1
            if changes_in_direction % 2 == 0:
                steps_in_current_direction += 1
    return balls

def create_ball(space, radius, mass, pos):
    body = pymunk.Body()
    body.position = (pos[0], pos[1])
    shape = pymunk.Circle(body, radius)
    shape.mass = mass
    shape.color = (255, 0, 0, 100)
    space.add(body, shape)
    return shape

def plot_intervals(intervals, x_range=(-np.pi, np.pi), num_points=1000):
    """
    Plot a function that is 1 within specified intervals and 0 otherwise.

    Parameters:
    intervals (list of np.ndarray): List of intervals (each interval as a numpy array with 2 elements).
    x_range (tuple): The range of x values for the plot (default is (-pi, pi)).
    num_points (int): Number of points to plot (default is 1000).
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.zeros_like(x)
    
    for interval in intervals:
        start, end = interval
        if start>end:
            y[(x >= start) & (x <= np.pi)] = 1
            y[(x >= -np.pi) & (x <= end)] = 1
        else: 
            y[(x >= start) & (x <= end)] = 1
    
    plt.plot(x, y)
    plt.xlabel('Angle (radians)')
    plt.ylabel('Value')
    plt.title('Interval Plot')
    plt.grid(True)
    plt.show()

def calculate_angle_and_distance(ball1, ball2):
    dx = ball2.body.position.x - ball1.body.position.x
    dy = ball2.body.position.y - ball1.body.position.y
    angle = math.atan2(dy, dx)
    distance = math.sqrt(dx**2 + dy**2)
    return angle, distance

def normalize_angle(angle):
    """
    Normalize an angle to the interval [-pi, pi].

    Parameters:
    angle (float): Angle in radians.

    Returns:
    float: Normalized angle in the interval [-pi, pi].
    """
    normalized_angle = angle % (2 * math.pi)  # Normalize to [0, 2*pi)
    if normalized_angle > math.pi:
        normalized_angle -= 2 * math.pi  # Shift to [-pi, pi)
    return normalized_angle

def create_histograms(balls, radius):
    histograms = []

    for ball1 in balls:
        histogram = []
        for ball2 in balls:
            if ball1 != ball2:
                angle, distance = calculate_angle_and_distance(ball1, ball2)
                angle1 = normalize_angle(angle - math.atan(radius / distance))
                angle2 = normalize_angle(angle + math.atan(radius / distance))
                histogram.append(np.array([angle1, angle2]))
        # Create histogram intervals based on visibility
        histograms.append(histogram)
    
    return histograms

def correct_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

def correct_interval(start_angle, end_angle, heading):
    
    corrected_start = correct_angle(start_angle + heading)
    corrected_end = correct_angle(end_angle + heading)
    
    # Check if the interval crosses the -pi/π boundary
    if corrected_start > corrected_end:
        corrected_end += 2 * np.pi
    
    # Normalize the angles to be within [0, 2π] for positive range
    # corrected_start = (corrected_start + 2 * np.pi) % (2 * np.pi) - np.pi
    # corrected_end = (corrected_end + 2 * np.pi) % (2 * np.pi) - np.pi

    corrected_start = normalize_angle(corrected_start)
    corrected_end = normalize_angle(corrected_end)

    return corrected_start, corrected_end

def dPhi_V_of(Phi, V):
    """Calculating derivative of VPF according to Phi visual angle array at a given timepoint t
        Args:
            Phi: linspace numpy array of visual field axis
            V: binary visual projection field array
        Returns:
            dPhi_V: derivative array of V w.r.t Phi
    """
    # circular padding for edge cases
    padV = np.pad(V, (1, 1), 'wrap')
    dPhi_V_raw = np.diff(padV)

    # we want to include non-zero value if it is on the edge
    if dPhi_V_raw[0] > 0 and dPhi_V_raw[-1] > 0:
        dPhi_V_raw = dPhi_V_raw[0:-1]

    else:
        dPhi_V_raw = dPhi_V_raw[1:, ...]

    dPhi_V = dPhi_V_raw  # / (Phi[-1] - Phi[-2])
    return dPhi_V

def update_velocity_with_histograms(balls, histograms, dt):
    global RANDOM_FIRST_HEADING
    N = 16384
    dphi = np.pi / 8192

    alpha0 = 10
    beta0 = 1
    alpha1 = 0.2
    beta1 = 0.2
    alpha2 = 0
    beta2 = 0

    # Define the angles array for integration
    angles = np.linspace(-np.pi, np.pi, N)
    cos_term = np.cos(angles)
    sin_term = np.sin(angles)
    phi = np.linspace(-np.pi, np.pi, N)

    for i, ball in enumerate(balls):
        V = np.zeros(N)
        if RANDOM_FIRST_HEADING:
            heading = np.random.uniform(-np.pi, np.pi)
            ball.body.velocity = (5*np.cos(heading), -5*np.sin(heading))
        else:
            heading = math.atan2(ball.body.velocity.y, ball.body.velocity.x)
        # Update visual field V from histograms
        for start_angle, end_angle in histograms[i]:
            
            start_angle, end_angle = correct_interval(start_angle, end_angle, heading)
            start_bin = int((start_angle + np.pi) / (2 * np.pi) * N)
            end_bin = int((end_angle + np.pi) / (2 * np.pi) * N)

            if start_bin < end_bin:
                V[start_bin:end_bin] = 1
            else:
                V[start_bin:] = 1
                V[:end_bin] = 1
        dvel, dpsi = compute_state_variables(np.sqrt(ball.body.velocity.x**2 + ball.body.velocity.y**2), phi, V, 0, MAX_VEL, alpha0, alpha1, alpha2, beta0, beta1, beta2)
        # ball.body.velocity += (dt*np.cos(PREFERED_DIR[0])*CONST_VEL_OF_DIR, dt*np.sin(PREFERED_DIR[1])*CONST_VEL_OF_DIR)
        # print(dvel, dpsi)
        psi = math.atan2(-ball.body.velocity.y, ball.body.velocity.x) + dpsi
        ball.body.velocity += (dvel*np.cos(psi), - dvel*np.sin(psi)) # - in y because using pygame
        # ball.body.velocity += (CONST_VEL_OF_DIR*np.cos(PREFERED_DIR[0]), CONST_VEL_OF_DIR*np.sin(PREFERED_DIR[1]))
        if np.sqrt(ball.body.velocity.x**2 + ball.body.velocity.y**2)>MAX_VEL:
            ball.body.velocity = (ball.body.velocity/np.linalg.norm(ball.body.velocity))*MAX_VEL
        # print(np.linalg.norm(ball.body.velocity))
    RANDOM_FIRST_HEADING = False

def plot_values(y):
    # Generate 16384 linearly spaced values between -pi and pi
    x = np.linspace(-np.pi, np.pi, 16384)
    
    # Plot the values
    plt.plot(x, y)
    plt.title('Plot of 16384 Elements Ranging from -π to π')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()

def compute_state_variables(vel_now: float, Phi, V_now, GAM=0.1, V0=1,
                            ALP0=None, ALP1=None, ALP2=None,
                            BET0=None, BET1=None, BET2=None):
    dt_V = np.zeros(len(Phi))

    # Deriving over Phi
    dPhi_V = dPhi_V_of(Phi, V_now)

    # Calculating series expansion of functional G
    G_vel = (-V_now + ALP2 * dt_V)

    # Spikey parts shall be handled separately because of numerical integration
    G_vel_spike = np.square(dPhi_V)

    G_psi = (-V_now + BET2 * dt_V)

    # Spikey parts shall be handled separately because of numerical integration
    G_psi_spike = np.square(dPhi_V)

    # ORIGINAL Algorithm
    dpsi = BET0 * integrate.trapezoid(np.sin(Phi) * G_psi, Phi) + BET0 * BET1 * np.sum(np.sin(Phi) * G_psi_spike)
    dvel = GAM * (V0 - vel_now) + ALP0 * integrate.trapezoid(np.cos(Phi) * G_vel, Phi) + \
            ALP0 * ALP1 * np.sum(np.cos(Phi) * G_vel_spike)
    return dvel, dpsi

def sigmoid(x, s):
    """Sigmoid function with steepness s."""
    return 2 / (1 + np.exp(-s*x)) - 1

def cos_sigmoid(x, s):
    """Composite sigmoid function resembling cos with steepness s."""
    # left part
    left = 2 / (1 + np.exp(-s * (x + (np.pi / 2)))) - 1
    right = -2 / (1 + np.exp(-s * (x - (np.pi / 2)))) + 1
    final = []
    for xid, xi in enumerate(list(x)):
        if xi < 0:
            final.append(left[xid])
        else:
            final.append(right[xid])
    return final

def sin_sigmoid(x, s):
    """Composite sigmoid function resembling sin with steepness s."""
    # left part
    middle = 2 / (1 + np.exp(-s * (x))) - 1
    left = -2 / (1 + np.exp(-s * (x + (np.pi)))) + 1
    right = -2 / (1 + np.exp(-s * (x - (np.pi)))) + 1
    final = []
    for xid, xi in enumerate(list(x)):
        if -np.pi / 2 < xi < np.pi / 2:
            final.append(middle[xid])
        elif xi < -np.pi / 2:
            final.append(left[xid])
        else:
            final.append(right[xid])
    return final

def recenter_balls(balls):
    center_mass_x = 0.0
    center_mass_y = 0.0
    for ball in balls:
        center_mass_x += ball.body.position[0]
        center_mass_y += ball.body.position[1]
    center_mass_x = center_mass_x/n_balls
    center_mass_y = center_mass_y/n_balls
    move_x = WIDTH/2 - center_mass_x
    move_y = HEIGHT/2 - center_mass_y 
    for ball in balls:
        ball.body.position = ball.body.position + (move_x, move_y)

def run(window, width, height, radius):
    pygame.init()
    run = True
    clock = pygame.time.Clock()
    fps = 30
    dt = 1/fps
    vel_phi = np.array([[0, 0]] * n_balls)
    space = pymunk.Space()
    space.gravity = (0, 0)

    balls = create_n_balls(n_balls, space, radius, 10)

    draw_options = pymunk.pygame_util.DrawOptions(window)
    while run: 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break

        draw(space, window, draw_options, balls)
        histograms = create_histograms(balls, radius)
        update_velocity_with_histograms(balls, histograms, dt)
        #vel_phi = set_velocity_for_all_balls(balls, histograms, vel_phi, dt)
        # print(histograms[0][0])
        # histograms[0][0][0], histograms[0][0][1] = correct_interval(histograms[0][0][0], histograms[0][0][1], np.pi/2)
        # print(histograms[0][0])
        # plot_intervals(histograms[0])
        if RECENTER:
            recenter_balls(balls)
        space.step(dt)
        clock.tick(fps)

        
    pygame.quit()
 
if __name__ == "__main__":
    run(window, WIDTH, HEIGHT, RADIUS)
