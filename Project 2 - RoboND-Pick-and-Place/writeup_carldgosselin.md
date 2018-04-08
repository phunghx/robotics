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

The Forward Kinematics (FK) approach calculates the final position and rotation of the end-effector with the parameters of each joint in a series of conjoined links.  The Kuka arm in this project has 6 joints (or 6 degrees of freedom).

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
	<img src="misc_images/DHparamAssignment2.JPG" height="600">	
</div>
<br>

A few notes about the kr210.urdf.xacro file... <br>

a1 is the distance between joint 1 and joint 2 -> 0.35. <br>
This is indicated in the kr210.urdf.xacro file: <br>
```
<joint name="joint_2" type="revolute">
 <origin xyz="0.35 0 0.42" rpy="0 0 0"/>
 ```
a2 is the distance between joint 2 and joint 3 along the z axis -> 1.25.
```
<joint name="joint_3" type="revolute">
 <origin xyz="0 0 1.25" rpy="0 0 0"/>
``` 
a3 is the distance between joint 3 and joint 4 along the z axis -> -0.054.
```
<joint name="joint_4" type="revolute">
 <origin xyz="0.96 0 -0.054" rpy="0 0 0"/>
```
d1 is the distane between joint 0 and joint 2 along the z axis.  Therefore the distance between joint 0 to joint 1 and joint 1 to joint 2 is to be added. <br>
0.33 + 0.42 -> 0.75.
```
<joint name="joint_1" type="revolute">
 <origin xyz="0 0 0.33" rpy="0 0 0"/>
 ...
<joint name="joint_2" type="revolute">
 <origin xyz="0.35 0 0.42" rpy="0 0 0"/>
```
d4 is the distance between joint 3 and joint 5 along the x axis. <br>
0.96 + 0.54 -> 1.5.
```
<joint name="joint_4" type="revolute">
 <origin xyz="0.96 0 -0.054" rpy="0 0 0"/>
 ...
<joint name="joint_5" type="revolute">
 <origin xyz="0.54 0 0" rpy="0 0 0"/>
```
dg is the distance between joint 6 and the end effector along the x axis. <br>
0.193 + 0.11 -> 0.303
```
<joint name="joint_6" type="revolute">
 <origin xyz="0.193 0 0" rpy="0 0 0"/>
 ...
<joint name="gripper_joint" type="fixed">
 <parent link="link_6"/>
 <child link="gripper_link"/>
 <origin xyz="0.11 0 0" rpy="0 0 0"/><!--0.087-->
```
note:  Theta, the revolute axis, has **theta 2** corresponding to a turn of -90 degrees.
<br><br>
Behold, the resulting DH parameters table:
<br><br>
<div align=center>
	<img src="misc_images/DHparameters2.JPG" height="600">
</div>
<br>

Here is the DH table in python code:
```
 # DH parameters
    s = {alpha0: 0,     a0:   0,    d1: 0.75,    q1: q1,
         alpha1: -pi/2, a1: 0.35,   d2: 0,       q2: q2 -pi/2,  
         alpha2: 0,     a2: 1.25,   d3: 0,       q3: q3,
         alpha3: -pi/2, a3: -0.054, d4: 1.5,     q4: q4,
         alpha4: pi/2,  a4:   0,    d5: 0,       q5: q5,
         alpha5: -pi/2, a5:   0,    d6: 0,       q6: q6,
         alpha6: 0,     a6:   0,    d7: 0.303,   q7: 0}
```

#### 2. Using the DH parameter table you derived earlier, create individual transformation matrices about each joint. 


Here is the theoretical equation to create individual transformation matrices.  As you can see, the DH convention uses four individual transforms:

<div align=center>
	<img src="misc_images/fourtransforms.png">
</div>
<br>
The following matrix will return a homogeneous transformation matrix:

<div align=center>
	<img src="misc_images/matrix2.png">
</div>
<br>

Here is the modified DH Transformation matrix in code:
```
# Modified DH Transformation matrix

def DH_T_Matrix (q, alpha, d, a):
    DH_Matrix = Matrix([[             cos(q),          -sin(q),           0,             a],
                        [ sin(q)*cos(alpha), cos(q)*cos(alpha), -sin(alpha), -sin(alpha)*d],
                        [ sin(q)*sin(alpha), cos(q)*sin(alpha),  cos(alpha),  cos(alpha)*d],
                        [                 0,                 0,           0,             1]])
    return DH_Matrix


# Create individual transformation matrices

    T0_1 = DH_T_Matrix(q1, alpha0, d1, a0).subs(s)
    T1_2 = DH_T_Matrix(q2, alpha1, d2, a1).subs(s)
    T2_3 = DH_T_Matrix(q3, alpha2, d3, a2).subs(s)
    T3_4 = DH_T_Matrix(q4, alpha3, d4, a3).subs(s)
    T4_5 = DH_T_Matrix(q5, alpha4, d5, a4).subs(s)
    T5_6 = DH_T_Matrix(q6, alpha5, d6, a5).subs(s)
    T6_G = DH_T_Matrix(q7, alpha6, d7, a6).subs(s)
```


In addition, also generate a generalized homogeneous transform between base_link and gripper_link using only end-effector(gripper) pose.
<br><br>
The following code is applied to generate a generalized homogeneous transform using only the end-effector pose:
```
T0_G = DH_T_Matrix(T0_1*T1_2*T2_3*T3_4*T4_5*T5_6*T6_G)
```
<br>

#### 3. Decouple Inverse Kinematics problem into Inverse Position Kinematics and Inverse Orientation Kinematics; doing so derive the equations to calculate all individual joint angles.

Inverse Kinematics (IK) is the opposite of Forward Kinematics (FK).  IK calculates the parameters of each joint in a series of conjoined links based on the end-effectors coordinate position and rotation. 
The IK problem can be decoupled into **Inverse Position** and **Inverse Orientation** because of a common intersection point.  The intersecting point is joint_5 and is called the wrist center (WC).  <br>
<br>
Below are the steps to solving the inverse kinematics problem: <br>
<br>
**Step 1**: is to complete the DH parameter table for the manipulator. Hint: place the origin of frames 4, 5, and 6 coincident with the WC. <br>

**Step 2**: is to find the location of the WC relative to the base frame. Recall that the overall homogeneous transform between the base and end effector has the form,
<div align=center>
	<img src="misc_images/IK_step2a.png">
</div>
</br>
If, for example, you choose z4 parallel to z6 and pointing from the WC to the EE, then this displacement is a simple translation along z6. The magnitude of this displacement, let’s call it d, would depend on the dimensions of the manipulator and are defined in the URDF file. Further, since r13, r23, and r33 define the Z-axis of the EE relative to the base frame, the Cartesian coordinates of the WC is, <br>
<br>
<div align=center>
	<img src="misc_images/IK_step2b.png">
</div>
</br>

**Step 3**: find joint variables, q1, q2 and q3, such that the WC has coordinates equal to equation (3). This is the hard step. One way to attack the problem is by repeatedly projecting links onto planes and using trigonometry to solve for joint angles. Unfortunately, there is no generic recipe that works for all manipulators so you will have to experiment. The example in the next section will give you some useful guidance. <br>
<br>

**Step 4**: once the first three joint variables are known, perform the calculations via the application of homogeneous transforms up to the WC. <br>
<br>

**Step 5**: find a set of Euler angles corresponding to the rotation matrix, <br>
<br>
<div align=center>
	<img src="misc_images/IK_step5.png">
</div>
<br>

**Step 6**: choose the correct solution among the set of possible solutions<br>
<br>
Here is the code for resolving the **Inverse Kinematics** problem: <br>
<br>
```
 # Compensate for rotation discrepancy between DH parameters and Gazebo
	    # More inforation can be found in KR210 forward kinematics section
        ROT_Error = ROT_z.subs(y, radians(180)) * ROT_y.subs(p, radians(-90))
    
        ROT_EE = ROT_EE * ROT_Error
        ROT_EE = ROT_EE.subs({'r': roll, 'p': pitch, 'y': yaw})

        EE = Matrix ([[px],
                      [py],
                      [pz]])

        WC = EE - (0.303) * ROT_EE[:,2]
	    
	    
	    # Calculate joint angles using Geometric IK method
	    theta1 = atan2(WC[1], WC[0])

        # SSS triangle for theta 2 and theta 3
        side_a = 1.501
        side_b = sqrt(pow((sqrt(WC[0] * WC[0] + WC[1] * WC[1]) - 0.35), 2) + pow((WC[2] - 0.75), 2))
        side_c = 1.25

        angle_a = acos((side_b * side_b + side_c * side_c - side_a * side_a) / (2 * side_b * side_c))
        angle_b = acos((side_a * side_a + side_c * side_c - side_b * side_b) / (2 * side_a * side_c))
        angle_c = acos((side_a * side_a + side_b * side_b - side_c * side_c) / (2 * side_a * side_b))

        theta2 = pi/2 - angle_a - atan2(WC[2] - 0.75, sqrt(WC[0] * WC[0] + WC[1] * WC[1]) - 0.35)
        theta3 = pi/2 - (angle_b + 0.036)  # 0.036 accounts for sag on link4 of -0.054m

        R0_3 = T0_1[0:3,0:3] * T1_2[0:3,0:3] * T2_3[0:3,0:3]
        R0_3 = R0_3.evalf(subs={q1 : theta1, q2 : theta2, q3 : theta3})
    
        R3_6 = R0_3.inv("LU")*ROT_EE

        # Euler angles from rotation matrix
        theta4 = atan2(R3_6[2,2], -R3_6[0,2])
        theta5 = atan2(sqrt(R3_6[0,2] * R3_6[0,2] + R3_6[2,2] * R3_6[2,2]), R3_6[1,2])
        theta6 = atan2(-R3_6[1,1], R3_6[1,0])
```

### Project Implementation

#### 1. Fill in the `IK_server.py` file with properly commented python code for calculating Inverse Kinematics based on previously performed Kinematic Analysis. Your code must guide the robot to successfully complete 8/10 pick and place cycles. Briefly discuss the code you implemented and your results. 

Hello, I need help here.  Not able to get the kuka arm to grip the can.  Do you mind checking my code in IK_server.py?