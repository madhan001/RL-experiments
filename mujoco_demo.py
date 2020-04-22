from mujoco_py import load_model_from_xml, MjSim, MjViewer
import glfw
import time

MODEL_XML = """
<?xml version="1.0" ?>
<mujoco>
    <option timestep="0.005" gravity="0 0 -9.81"/>
    
    <worldbody>
        <body name="ball" pos="0 -0.9 0.11">
            <joint axis="1 0 0" name="ball-x" type="slide"/> <!--x axis-->
            <joint axis="0 1 0" name="ball-y" type="slide"/> <!--y axis-->
            <joint axis="0 0 1" name="ball-z" type="slide"/>   <!--z axis-->
            <geom mass="1.0" pos="0 0 0" rgba="1 0 0 1" size="0.1" type="sphere"/>  <!--geometric object type-->
        </body>
        <body name="box" pos="0 0.7 0.11">
            <joint axis="1 0 0" name="box-x" type="slide"/>
            <joint axis="0 1 0" name="box-y" type="slide"/>
            <joint axis="0 0 1" name="box-z" type="slide"/>
            <geom mass="1.0" pos="0 0 0" rgba="0 0 1 1" size="0.1 0.1 0.1" type="box"/>
        </body>
        <body name="floor" pos="0 0 0">
            <geom size="1.0 1.0 0.01" rgba="0 1 0 1" type="box"/>
        </body>
    </worldbody>
    <actuator>
        <general joint="ball-y"/> <!-- move ball along y axis -->
    </actuator>
</mujoco>
"""

model = load_model_from_xml(MODEL_XML)
sim = MjSim(model)
viewer = MjViewer(sim)
t = 0
for t in range(350):
    #sim.data.ctrl[0] = 10 #control signal to actuator
    if t==20: #simulates impulsive initial velocity
        sim.data.qvel[1] = 7.5 #got this value by trial and error
    #print(sim.data.qvel[1]) #prints y velocity for debugging
    time.sleep(0.01) #slows down animation
    sim.step()
    viewer.render()

glfw.destroy_window(viewer.window)