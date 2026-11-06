import sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "sim")]
from sim.runner import GladosSim

POINTS = [(-0.3, 0.1, -0.55), (-0.2, -0.15, -0.6)]  # head_mount xyz in m


def run(robot, rounds=None):
    #n = 0
    #while robot._running() and (rounds is None or n < rounds):
        #for point in POINTS:
            #robot.move(position=point, unit="cart_pos")
            #robot.run_action("nod")
        #n += 1
    
    robot.move(main_swivel=0, upper_arm=-0.27, lower_arm = 1.28, tilt=0.24, nod=0.27, unit="rad")

    while robot._running():
        #robot.run_action("ending_demo")
        robot.run_action(nod=30, lower_arm=80, duration=2)
        robot.run_action(main_swivel=-30, tilt=-30, duration=2)
        robot.run_action(nod=15.47, lower_arm=83.03, tilt=-15, duration=1)
        robot.run_action(nod=-10, lower_arm=5, duration=2, additive=True)
        robot.run_action(nod=20, lower_arm=-10, duration=2, additive=True)

        robot.move(main_swivel=0, upper_arm=-0.27, lower_arm = 1.28, tilt=0.24, nod=0.27, unit="rad")



if __name__ == "__main__":
    sim = GladosSim()
    with sim.launch():
        run(sim.robot)