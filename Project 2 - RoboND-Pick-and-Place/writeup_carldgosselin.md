## Project: Kinematics Pick & Place
### The Writeup
---

<div align=center>
	<img src="misc_images/req-challenge.gif">
</div>

**Steps to complete the project:**  


1. Set up your ROS Workspace.
2. Download or clone the [project repository](https://github.com/udacity/RoboND-Kinematics-Project) into the ***src*** directory of your ROS Workspace.  
3. Experiment with the forward_kinematics environment and get familiar with the robot.
4. Launch in [demo mode](https://classroom.udacity.com/nanodegrees/nd209/parts/7b2fd2d7-e181-401e-977a-6158c77bf816/modules/8855de3f-2897-46c3-a805-628b5ecf045b/lessons/91d017b1-4493-4522-ad52-04a74a01094c/concepts/ae64bb91-e8c4-44c9-adbe-798e8f688193).
5. Perform Kinematic Analysis for the robot following the [project rubric](https://review.udacity.com/#!/rubrics/972/view).
6. Fill in the `IK_server.py` with your Inverse Kinematics code. 


[//]: # (Image References)

[image1]: ./misc_images/misc1.png
[image2]: ./misc_images/misc3.png
[image3]: ./misc_images/misc2.png

## [Rubric](https://review.udacity.com/#!/rubrics/972/view) Points

---
### The Writeup

### Kinematic Analysis
#### 1. Run the forward_kinematics demo and evaluate the kr210.urdf.xacro file to perform kinematic analysis of Kuka KR210 robot and derive its DH parameters.

Below is a summary of the DH parameter assignment process:

1. Label all joints from {1, 2, … , n}.
2. Label all links from {0, 1, …, n} starting with the fixed base link as 0.
3. Draw lines through all joints, defining the joint axes.
4. Assign the Z-axis of each frame to point along its joint axis.
5. Identify the common normal between each frame \hat{Z}_{i-1} 
Z
^
  
i−1
​	  and frame \hat{Z}_{i} 
Z
^
  
i
​	  .

The endpoints of "intermediate links" (i.e., not the base link or the end effector) are associated with two joint axes, {i} and {i+1}. For i from 1 to n-1, assign the \hat{X}_{i} 
X
^
  
i
​	  to be …

For skew axes, along the normal between \hat{Z}_{i} 
Z
^
  
i
​	  and \hat{Z}_{i+1} 
Z
^
  
i+1
​	  and pointing from {i} to {i+1}.
For intersecting axes, normal to the plane containing \hat{Z}_{i} 
Z
^
  
i
​	  and \hat{Z}_{i+1} 
Z
^
  
i+1
​	 .

For parallel or coincident axes, the assignment is arbitrary; look for ways to make other DH parameters equal to zero.

For the base link, always choose frame {0} to be coincident with frame {1} when the first joint variable ( {\theta}_{1}θ 
1
​	  or {d}_{1}d 
1
​	 ) is equal to zero. This will guarantee that {\alpha}_{0}α 
0
​	  = {a}_{0}a 
0
​	  = 0, and, if joint 1 is a revolute, {d}_{1}d 
1
​	  = 0. If joint 1 is prismatic, then {\theta}_{1}θ 
1
​	 = 0.

For the end effector frame, if joint n is revolute, choose {X}_{n}X 
n
​	  to be in the direction of {X}_{n-1}X 
n−1
​	  when {\theta}_{n}θ 
n
​	  = 0 and the origin of frame {n} such that {d}_{n}d 
n
​	  = 0.

#### 2. Using the DH parameter table you derived earlier, create individual transformation matrices about each joint. In addition, also generate a generalized homogeneous transform between base_link and gripper_link using only end-effector(gripper) pose.

Links | alpha(i-1) | a(i-1) | d(i-1) | theta(i)
--- | --- | --- | --- | ---
0->1 | 0 | 0 | L1 | qi
1->2 | - pi/2 | L2 | 0 | -pi/2 + q2
2->3 | 0 | 0 | 0 | 0
3->4 |  0 | 0 | 0 | 0
4->5 | 0 | 0 | 0 | 0
5->6 | 0 | 0 | 0 | 0
6->EE | 0 | 0 | 0 | 0


#### 3. Decouple Inverse Kinematics problem into Inverse Position Kinematics and inverse Orientation Kinematics; doing so derive the equations to calculate all individual joint angles.

And here's where you can draw out and show your math for the derivation of your theta angles. 

![alt text][image2]

### Project Implementation

#### 1. Fill in the `IK_server.py` file with properly commented python code for calculating Inverse Kinematics based on previously performed Kinematic Analysis. Your code must guide the robot to successfully complete 8/10 pick and place cycles. Briefly discuss the code you implemented and your results. 


Here I'll talk about the code, what techniques I used, what worked and why, where the implementation might fail and how I might improve it if I were going to pursue this project further.  


And just for fun, another example image:
![alt text][image3]


