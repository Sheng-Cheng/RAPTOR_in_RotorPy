from foundation_policy import Raptor
from scipy.spatial.transform import Rotation
import numpy as np
# policy = Raptor()
# policy.reset()
# for simulation_step in range(1000):
#     observation = np.array([[*sim.position, *R(sim.orientation).flatten(), *sim.linear_velocity, *sim.angular_velocity, *sim.action]])
#     action = policy.evaluate_step(observation)[0] # the policy works on batches by default
#     simulation.step(action) # simulation dt=10 ms

class FoudationPolicy(object):
    """
    implementing the RAPTOP policy
    """
    def __init__(self, quad_params):
        """
        Parameters:
            quad_params, dict with keys specified in rotorpy/vehicles
        """

        self.g = 9.81 # m/s^2

        self.policy = Raptor()
        self.policy.reset()

        # initialize the action
        self.action_past = np.zeros(4)

        # load rpm info
        self.rotor_speed_min = quad_params['rotor_speed_min']
        self.rotor_speed_max = quad_params['rotor_speed_max']

    def update(self, t, state, flat_output):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, rad/s
                cmd_motor_thrusts, N
                cmd_thrust, N 
                cmd_moment, N*m
                cmd_q, quaternion [i,j,k,w]
                cmd_w, angular rates in the body frame, rad/s
                cmd_v, velocity in the world frame, m/s
        """

        # fetch the state estimates
        pos  = np.clip(state['x'], -1, 1)
        vel = np.clip(state['v'], -1, 1)
        R = Rotation.from_quat(state['q']).as_matrix()
        omega = state['w']

        # coordinate change (ENU -> FLU/NWU)
        # FLU (x = forward, y = left, z = up)
        def enu_to_nwu_rotation(R_enu):
            """Convert a rotation matrix from ENU to NWU frame."""
            # Transformation matrix from ENU to NWU
            T_enu2nwu = np.array([
                [0, 1, 0],
                [-1, 0, 0],
                [0, 0, 1]
            ])
            return T_enu2nwu @ R_enu
        
        observation = np.hstack((pos, R.flatten(), vel, omega, self.action_past))

        action = np.clip(self.policy.evaluate_step(observation)[0], -1, 1)
        # save current control action
        self.action_past = action

        # put dummy cmd_moment and cmd_q just to pass the sanity checl
        # control wise the cmd_motor_speeds dominates
        cmd_moment_dummy = np.zeros((3,))
        cmd_q_dummy = np.array([0,0,0,1])
        cmd_thrust_dummy = 0
        
        # crazyflie convention is the same as raptor motor location/numbering convetion
        # action = np.array([action[0], action[1], action[2], action[3]])

        # denormalize based on https://github.com/rl-tools/raptor?tab=readme-ov-file#usage
        action_remapped = np.array([action[3], action[0], action[1], action[2]])
        action_rpm = (self.rotor_speed_max - self.rotor_speed_min) * (action_remapped + 1)/2 + self.rotor_speed_min
        # is the unit correct for the converted rpms?
        
        # debug
        # print(np.clip(action_rpm,self.rotor_speed_min,self.rotor_speed_max))
        print(action_rpm)

        # load control
        control_input = {'cmd_motor_speeds':action_rpm,
                         'cmd_thrust':cmd_thrust_dummy,
                         'cmd_moment':cmd_moment_dummy,
                         'cmd_q':cmd_q_dummy}
        
        # 9.24: initial attempt. It couldn't fly the quadrotor. 
        # Things to check:
        # 1. [done] align the reference frame with their requirement
        # 2. [done] align motor numbering with the output command
        # 3. [done] check what's the unit of the raptor policy output (asked the authors on Sept. 27, updated my code based on the instructions from the author)

        # 9.28: I implemented the code after getting further instructions from the author. Still couldn't fly it

        return control_input