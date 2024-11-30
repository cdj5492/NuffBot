import time
from inputs import get_gamepad
import rlviser_py as vis
import RocketSim as rs
import random
import keyboard
import torch
from rlgym_sim.utils import common_values
import math
import threading

class XboxController(object):
    MAX_TRIG_VAL = math.pow(2, 8)
    MAX_JOY_VAL = math.pow(2, 15)

    def __init__(self):

        self.LeftJoystickY = 0
        self.LeftJoystickX = 0
        self.RightJoystickY = 0
        self.RightJoystickX = 0
        self.LeftTrigger = 0
        self.RightTrigger = 0
        self.LeftBumper = 0
        self.RightBumper = 0
        self.A = 0
        self.X = 0
        self.Y = 0
        self.B = 0
        self.LeftThumb = 0
        self.RightThumb = 0
        self.Back = 0
        self.Start = 0
        self.LeftDPad = 0
        self.RightDPad = 0
        self.UpDPad = 0
        self.DownDPad = 0

        self._monitor_thread = threading.Thread(target=self._monitor_controller, args=())
        self._monitor_thread.daemon = True
        self._monitor_thread.start()


    def read(self): # return the buttons/triggers that you care about in this methode
        x = self.LeftJoystickX
        y = self.LeftJoystickY
        a = self.A
        b = self.X # b=1, x=2
        rb = self.RightBumper
        return [x, y, a, b, rb]


    def _monitor_controller(self):
        while True:
            events = get_gamepad()
            for event in events:
                if event.code == 'ABS_Y':
                    self.LeftJoystickY = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_X':
                    self.LeftJoystickX = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_RY':
                    self.RightJoystickY = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_RX':
                    self.RightJoystickX = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_Z':
                    self.LeftTrigger = event.state / XboxController.MAX_TRIG_VAL # normalize between 0 and 1
                elif event.code == 'ABS_RZ':
                    self.RightTrigger = event.state / XboxController.MAX_TRIG_VAL # normalize between 0 and 1
                elif event.code == 'BTN_TL':
                    self.LeftBumper = event.state
                elif event.code == 'BTN_TR':
                    self.RightBumper = event.state
                elif event.code == 'BTN_SOUTH':
                    self.A = event.state
                elif event.code == 'BTN_NORTH':
                    self.Y = event.state #previously switched with X
                elif event.code == 'BTN_WEST':
                    self.X = event.state #previously switched with Y
                elif event.code == 'BTN_EAST':
                    self.B = event.state
                elif event.code == 'BTN_THUMBL':
                    self.LeftThumb = event.state
                elif event.code == 'BTN_THUMBR':
                    self.RightThumb = event.state
                elif event.code == 'BTN_SELECT':
                    self.Back = event.state
                elif event.code == 'BTN_START':
                    self.Start = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY1':
                    self.LeftDPad = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY2':
                    self.RightDPad = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY3':
                    self.UpDPad = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY4':
                    self.DownDPad = event.state



game_mode = rs.GameMode.SOCCAR

# Create example arena
arena = rs.Arena(game_mode)

# Set boost pad locations
vis.set_boost_pad_locations([pad.get_pos().as_tuple() for pad in arena.get_boost_pads()])


# Setup example arena
car = arena.add_car(rs.Team.BLUE)
car2 = arena.add_car(rs.Team.ORANGE)
car.set_state(rs.CarState(pos=rs.Vec(z=17), vel=rs.Vec(x=50), boost=100))
car2.set_state(rs.CarState(pos=rs.Vec(z=17), vel=rs.Vec(x=50), boost=100))
arena.ball.set_state(rs.BallState(pos=rs.Vec(y=400, z=100), ang_vel=rs.Vec(x=2)))
car2.set_controls(rs.CarControls(throttle=1, steer=.5, boost=True))

# buncha cars
#cars = []
#for i in range(200):
    #cars.append(arena.add_car(rs.Team.ORANGE))
    #cars[i].set_state(rs.CarState(pos=rs.Vec(z=17), vel=rs.Vec(x=50), boost=100))
    
controller = XboxController()

# Run for 60 seconds
TIME = 60

steps = 0
start_time = time.time()
# for i in range(round(TIME * arena.tick_rate)):
while True:
    throttle = 0
    steer = 0
    jump = 0
    boost = 0
    roll = 0
    handbrake = 0
    # if keyboard.is_pressed('w'):
    #     throttle = 1
    # if keyboard.is_pressed('a'):
    #     steer = -1
    # if keyboard.is_pressed('s'):
    #     throttle = -1
    # if keyboard.is_pressed('d'):
    #     steer = 1
    # if keyboard.is_pressed('q'):
    #     roll = -1
    # if keyboard.is_pressed('e'):
    #     roll = 1
    # if keyboard.is_pressed('k'):
    #     jump = 1
    # if keyboard.is_pressed('j'):
    #     boost = 1
    # if keyboard.is_pressed('shift'):
    #     handbrake = 1
    throttle = controller.RightTrigger - controller.LeftTrigger
    steer = controller.LeftJoystickX
    jump = controller.A
    boost = 1 if controller.RightBumper else 0
    if controller.X:
        roll += -1
    if controller.B:
        roll += 1
    pitch = -controller.LeftJoystickY
    handbrake = 1 if controller.LeftBumper else 0


    
    
    car.set_controls(rs.CarControls(throttle=throttle, steer=steer, boost=boost, jump=jump, pitch=pitch, yaw=steer, roll=roll, handbrake=handbrake))

    # car2 gets controlled by the model
    # create observation
    # obs_builder = DefaultObs(
    #   pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
    #   ang_coef=1 / np.pi,
    #   lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
    #   ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL)

    controls = rs.CarControls(throttle=random.randint(0, 100)/100, steer=random.randrange(-1, 1), boost=random.randint(0, 1) == 1, jump=random.randint(0, 100) == 1)
    car2.set_controls(controls)
    #for c in cars:
        #c.set_controls(controls)


    arena.step(1)

    # Render the current game state
    pad_states = [pad.get_state().is_active for pad in arena.get_boost_pads()]
    ball = arena.ball.get_state()
    car_data = [
        (car.id, car.team, car.get_config(), car.get_state())
        for car in arena.get_cars()
    ]

    vis.render(steps, arena.tick_rate, game_mode, pad_states, ball, car_data)

    # sleep to simulate running real time (it will run a LOT after otherwise)
    time.sleep(max(0, start_time + steps / arena.tick_rate - time.time()))
    steps += 1

# Tell RLViser to exit
print("Exiting...")
vis.quit()