import time
import pybullet as p
import pybullet_data

import EnvCreator

file = "./maps/empty.png"

#create urdf of environment
env_c = EnvCreator.envCreator(file=file,resolution=0.1,height=2,density=1,flip=False) ##see code for options
env_urdf = env_c.get_urdf_fast(output_dir="./maps")
print("creating ",env_urdf," ...")

#init pybullet env
client = p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
p.resetSimulation()

#create ground plane
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")

#load urdf files
env_asset = p.loadURDF(env_urdf,useFixedBase=True,flags=p.URDF_MERGE_FIXED_LINKS)

for t in range(10000):
    p.stepSimulation()
    time.sleep(1./100.)