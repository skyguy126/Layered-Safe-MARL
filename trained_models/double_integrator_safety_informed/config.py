import numpy as np

class KinematicVehicleConfig():
    V_MIN = 0.5
    V_MAX = 1.5
    V_NOMINAL = 1.0
    ACCEL_MIN = -1  
    ACCEL_MAX = 1
    ANGULAR_RATE_MAX = 1
    MOTION_PRIM_ACCEL_OPTIONS = 5 ## choices are 3 and 5
    MOTION_PRIM_ANGRATE_OPTIONS = 5 ## total motion primitive choices are MOTION_PRIM_ACCEL_OPTIONS * MOTION_PRIM_ANGRATE_OPTIONS
    # Pair of vehicle within this distance might have to start deconfliction maneuver. Pair further than this distance can do whatever they want.
    # This distance is based on the reachability computation (specific to the dynamics), and depends on the SEPARATION_DISTANCE.
    ENGAGEMENT_DISTANCE = 2.0
    # simulation timestep
    DT = 0.1
    # agent within this distance to the landmark is considered to have reached the goal.
    DISTANCE_TO_GOAL_THRESHOLD = 0.3
    GOAL_HEADING_THRESHOLD = np.pi/4
    GOAL_SPEED_THRESHOLD = 0.1 # m/s
    # separation distance between agents for safety
    SEPARATION_DISTANCE = 0.5
    # communication distance (entities within this distance are considered in each agent's observations)
    COORDINATION_RANGE = 5

    # Safety value function for safety filter
    VALUE_FUNCTION_FILE_NAME = 'data/kinematic_vehicle_value_function.pkl'

class JobyS4VehicleConfig():
    V_MIN = 60 * 0.514444 * 0.001 # knots to km/s
    V_MAX = 175 * 0.514444  * 0.001 # knots to km/s
    V_NOMINAL = 110 * 0.514444  * 0.001 # knots to km/s
    # V_MIN = 0.03 # knots to km/s
    # V_MAX = 0.09 # knots to km/s
    # V_NOMINAL = 0.06 # knots to km/s
    ACCEL_MIN = -0.001 # km/s^2
    ACCEL_MAX = 0.002 # km/s^2
    ANGULAR_RATE_MAX = 0.1 # rad/s
    MOTION_PRIM_ACCEL_OPTIONS = 5
    MOTION_PRIM_ANGRATE_OPTIONS = 5
    # Note that this has to change based on separation distance.
    # 5 When separation distance is 2200 ft.
    # TO FIX WITH NEW VALUE FUNCTION.
    ENGAGEMENT_DISTANCE = 2.3
    # separation distance used to evaluate the engagement distance.
    ENGAGEMENT_DISTANCE_REFERENCE_SEPARATION_DISTANCE = 2200 * 0.0003048
    
    DT = 1.0
    DISTANCE_TO_GOAL_THRESHOLD = 750 * 0.0003048 # ft to km
    GOAL_HEADING_THRESHOLD = np.pi/4
    GOAL_SPEED_THRESHOLD = 20 * 0.514444 * 0.001 # knots to km/s
    # GOAL_SPEED_THRESHOLD = 0.02 # knots to km/s
    
    # Ref: Preliminary Analysis of Separation Standards for Urban Air Mobility Using Unmitigated Fast-Time
    # Test params: 1500, 1800, 2200, 5000 ft
    SEPARATION_DISTANCE = 2200 * 0.0003048 # (first parameter in ft, converted to km)
    # COORDINATION_RANGE = 5 # 3 miles to km
    COORDINATION_RANGE = 3 * 1.60934 # 3 miles to km
    VALUE_FUNCTION_FILE_NAME = 'data/joby_value_function_new_config.pkl'
    # VALUE_FUNCTION_FILE_NAME = 'data/joby_value_function.pkl'
    TTR_FILE_NAME = 'data/joby_reach_ttr_new_t500_2.pkl'
    # TTR_FILE_NAME = 'data/joby_reach_ttr.pkl'


class DoubleIntegratorConfig():
    VX_MIN = -0.5
    VX_MAX = 0.5
    VY_MIN = -0.5   
    VY_MAX = 0.5
    # Only used for goal point target speed.
    V_MIN = 0.1
    V_NOMINAL = 0.5
    V_MAX = np.sqrt(VX_MAX**2 + VY_MAX**2)
    ACCELX_MIN = -0.5
    ACCELX_MAX = 0.5
    ACCELY_MIN = -0.5
    ACCELY_MAX = 0.5
    ACCELX_OPTIONS = 5 # double check appropriate value
    ACCELY_OPTIONS = 5 # double check appropriate value
    ENGAGEMENT_DISTANCE = 1.0
    # separation distance used to evaluate the engagement distance.
    ENGAGEMENT_DISTANCE_REFERENCE_SEPARATION_DISTANCE = 0.5

    DT = 0.1
    DISTANCE_TO_GOAL_THRESHOLD = 0.2 # m
    GOAL_HEADING_THRESHOLD = np.pi/4
    GOAL_SPEED_THRESHOLD = 0.15 # m/s
    SEPARATION_DISTANCE = 0.5 # m
    COORDINATION_RANGE = 4 # m
    
    VALUE_FUNCTION_FILE_NAME = 'data/crazyflies_value_function.pkl'
    
class TwoVehicleDeconflictionNmpcConfig():
    """ TODO: change class name to generalize to safety filter config, and specify to kinematic vehicle.
    """
    CBF_RATE = 3.0
    # this might be different from common config dt. This is MPC's discretization timestep.
    DT = 0.1
    # Horizon in time
    T_HORIZON = 4

class RewardWeightConfig():
    # min and max reward at each timestep.
    MIN_REWARD = -40
    MAX_REWARD = 50
    
    GOAL_REACH = 50
    CONFLICT = -20 # if agent is within separation distance of another agent
    CONFLICT_VALUE = -2 # safety value function is given as penalty. this is a multiplication weight.
    MULTIPLE_ENGAGEMENT = -1 # if multiple agents are within the engagement distance of the agent.
    DIFF_FROM_FILTERED_ACTION = -1 # penalty if unfiltered MARL action is different from filtered action.

class RewardBinaryConfig():
    # fyi, goal_reach_reward is always true since it is the main reward term form performance.     
    # Usually, only one of CONFLICT or CONFLICT_VALUE is used.
    CONFLICT = False
    CONFLICT_VALUE = False
    MULTIPLE_ENGAGEMENT = False
    SEPARATION_DISTANCE_CURRICULUM = False
    INITIAL_PHASE_USE_SAFETY_FILTER = False
    # Don't try this before RSS submission..
    DIFF_FROM_FILTERED_ACTION = False
    
## TODO: think about better way
# supported types: "circular_config", "left_to_right_merge", "left_to_right_cross", "bottom_to_top_merge"
eval_scenario_type = "left_to_right_merge_and_land"
