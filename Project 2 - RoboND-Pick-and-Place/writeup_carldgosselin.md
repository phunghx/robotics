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

<div align=center>
	<img src="misc_images/DH_assignment.png">
</div>



Below are the DH parameters for this project:

[//]: # (Image References)
[start]: ./readme_images/start.jpg
[dh]: ./readme_images/dh.png
[alpha]: ./readme_images/alpha.png
[alpha_i-1]: ./readme_images/alpha_i-1.png
[a]: ./readme_images/a_i-1.png
[d]: ./readme_images/d_i.png
[theta]: ./readme_images/theta_i.png
[pi2]: ./readme_images/pi2.png
[-pi2]: ./readme_images/-pi2.png
[theta1]: ./readme_images/theta_1.png
[theta2]: ./readme_images/theta_2.png
[theta2-90]: ./readme_images/theta_2-90.png
[theta3]: ./readme_images/theta_3.png
[theta4]: ./readme_images/theta_4.png
[theta5]: ./readme_images/theta_5.png
[theta6]: ./readme_images/theta_6.png
[transform-single]: ./readme_images/transform-single.png
[transform-simple]: ./readme_images/transform-simple.png
[transform-composition1]: ./readme_images/transform-composition1.png
[transform-composition2]: ./readme_images/transform-composition2.png
[A_r_P_A_0]: ./readme_images/A_r_P_A_0.png
[A]: ./readme_images/A.png
[P]: ./readme_images/P.png
[A_0]: ./readme_images/A_0.png
[R_small]: ./readme_images/r.png
[r_11]: ./readme_images/r_11.png
[A_B_R]: ./readme_images/A_B_R.png
[rotation-single]: ./readme_images/rotation-single.png
[transform-comb]: ./readme_images/transform-comb.png
[diag-clean]: ./readme_images/diag-clean.png
[diag-detailed]: ./readme_images/diag-detailed.png
[O_0]: ./readme_images/O_0.png
[O_1]: ./readme_images/O_1.png
[O_2]: ./readme_images/O_2.png
[O_2_1]: ./readme_images/O_2_1.png
[O_EE]: ./readme_images/O_EE.png
[Z_1]: ./readme_images/Z_1.png
[theta_2-calc]: ./readme_images/theta_2-calc.png
[theta_2-zoom]: ./readme_images/theta_2-zoom.png
[delta]: ./readme_images/delta.png
[delta-calc]: ./readme_images/delta-calc.png
[WC]: ./readme_images/WC.png
[WC^1]: ./readme_images/WC^1.png
[theta_3-zoom]: ./readme_images/theta_3-zoom.png
[theta_3-calc]: ./readme_images/theta_3-calc.png
[epsilon]: ./readme_images/epsilon.png
[epsilon-calc]: ./readme_images/epsilon-calc.png
[T]: ./readme_images/T.png
[R]: ./readme_images/R.png
[R-calc]: ./readme_images/R-calc.png
[R_0_6]: ./readme_images/R_0_6.png
[R_3_6]: ./readme_images/R_3_6.png
[R_rpy-calc]: ./readme_images/R_rpy-calc.png
[R_3_6-calc-LHS-1]: ./readme_images/R_3_6-calc-LHS-1.png
[R_3_6-calc-LHS-2]: ./readme_images/R_3_6-calc-LHS-2.png
[y]: ./readme_images/y.png
[P_small]: ./readme_images/p.png

![dh][dh]

|ID   |![alpha][alpha_i-1] |![a][a] |![d][d] |![theta][theta]    |
|:---:|:------------------:|:------:|:------:|:-----------------:| 
|    1|                  0 |      0 |   0.75 |     ![q1][theta1] |
|    2|      ![-pi2][-pi2] |   0.35 |      0 |  ![q2][theta2-90] |
|    3|                  0 |   1.25 |      0 |     ![q3][theta3] |
|    4|      ![-pi2][-pi2] | -0.054 |   1.50 |     ![q4][theta4] |
|    5|        ![pi2][pi2] |      0 |      0 |     ![q5][theta5] |
|    6|      ![-pi2][-pi2] |      0 |      0 |     ![q6][theta6] |
|   EE|                  0 |      0 |  0.303 |                 0 |


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


