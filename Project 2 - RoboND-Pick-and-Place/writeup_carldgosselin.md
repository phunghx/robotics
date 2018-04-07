## Project: Kinematics Pick & Place
### The Writeup
In this project, a simulated Kuka KR210 arm targets, picks up, and disposes a can from a shelf to a dropbox.

<div align=center>
	<img src="misc_images/req-challenge.gif">
</div>
<br>

[//]: # (Image References)

[image1]: ./misc_images/misc1.png
[image2]: ./misc_images/misc3.png
[image3]: ./misc_images/misc2.png

### Kinematic Analysis
#### 1. Run the forward_kinematics demo and evaluate the kr210.urdf.xacro file to perform kinematic analysis of Kuka KR210 robot and derive its DH parameters.

Below are a few snippets of code from the kr210.urdf.xacro file:

```
 <!--Links-->
 <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="${mass0}"/>
      <inertia ixx="60" ixy="0" ixz="0" iyy="70" iyz="0" izz="100"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="${-pi/2} 0 0"/>
      <geometry>
        <mesh filename="package://kuka_arm/meshes/kr210l150/visual/base_link.dae"/>
      </geometry>
      <material name="">
        <color rgba="0.75294 0.75294 0.75294 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="${-pi/2} 0 0"/>
      <geometry>
        <mesh filename="package://kuka_arm/meshes/kr210l150/collision/base_link.stl"/>
      </geometry>
    </collision>
  </link>
  ...

 <!-- joints -->
  <joint name="fixed_base_joint" type="fixed">
    <parent link="base_footprint"/>
    <child link="base_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>
  ...
  
  <!--Transmission and actuators-->
  <transmission name="tran1">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="joint_1">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="motor1">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>
  ...
```
The data from the kr210.urdf.xacro file (above) is used to derive the DH parameters.


Below is a summary of the DH parameter assignment process:
<br>
<div align=center>
	<img src="misc_images/DH_assignment.png">
</div>
<br> 
Result...
<div align=center>
	<img src="misc_images/DHparamAssignment.JPG" height="600">	
</div>
<br>

Below are the DH parameters:
<br><br>
<div align=center>
	<img src="misc_images/DHparameters2.png" height="600">
</div>

#### 2. Using the DH parameter table you derived earlier, create individual transformation matrices about each joint. 

Links | alpha(i-1) | a(i-1) | d(i-1) | theta(i)
--- | --- | --- | --- | ---
0->1 | 0 | 0 | L1 | qi
1->2 | - pi/2 | L2 | 0 | -pi/2 + q2
2->3 | 0 | 0 | 0 | 0
3->4 |  0 | 0 | 0 | 0
4->5 | 0 | 0 | 0 | 0
5->6 | 0 | 0 | 0 | 0
6->EE | 0 | 0 | 0 | 0


In addition, also generate a generalized homogeneous transform between base_link and gripper_link using only end-effector(gripper) pose.
<br>
[placeholder]
<br>


#### 3. Decouple Inverse Kinematics problem into Inverse Position Kinematics and inverse Orientation Kinematics; doing so derive the equations to calculate all individual joint angles.

And here's where you can draw out and show your math for the derivation of your theta angles. 

![alt text][image2]

### Project Implementation

#### 1. Fill in the `IK_server.py` file with properly commented python code for calculating Inverse Kinematics based on previously performed Kinematic Analysis. Your code must guide the robot to successfully complete 8/10 pick and place cycles. Briefly discuss the code you implemented and your results. 


Here I'll talk about the code, what techniques I used, what worked and why, where the implementation might fail and how I might improve it if I were going to pursue this project further.  

