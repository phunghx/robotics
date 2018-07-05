## Project: Perception Pick & Place

# Required Steps for a Passing Submission:
1. Extract features and train an SVM model on new objects (see `pick_list_*.yaml` in `/pr2_robot/config/` for the list of models you'll be trying to identify). 
2. Write a ROS node and subscribe to `/pr2/world/points` topic. This topic contains noisy point cloud data that you must work with.
3. Use filtering and RANSAC plane fitting to isolate the objects of interest from the rest of the scene.
4. Apply Euclidean clustering to create separate clusters for individual items.
5. Perform object recognition on these objects and assign them labels (markers in RViz).
6. Calculate the centroid (average in x, y and z) of the set of points belonging to that each object.
7. Create ROS messages containing the details of each object (name, pick_pose, etc.) and write these messages out to `.yaml` files, one for each of the 3 scenarios (`test1-3.world` in `/pr2_robot/worlds/`).  [See the example `output.yaml` for details on what the output should look like.](https://github.com/udacity/RoboND-Perception-Project/blob/master/pr2_robot/config/output.yaml)  
8. Submit a link to your GitHub repo for the project or the Python code for your perception pipeline and your output `.yaml` files (3 `.yaml` files, one for each test world).  You must have correctly identified 100% of objects from `pick_list_1.yaml` for `test1.world`, 80% of items from `pick_list_2.yaml` for `test2.world` and 75% of items from `pick_list_3.yaml` in `test3.world`.
9. Congratulations!  Your Done!

# Extra Challenges: Complete the Pick & Place
7. To create a collision map, publish a point cloud to the `/pr2/3d_map/points` topic and make sure you change the `point_cloud_topic` to `/pr2/3d_map/points` in `sensors.yaml` in the `/pr2_robot/config/` directory. This topic is read by Moveit!, which uses this point cloud input to generate a collision map, allowing the robot to plan its trajectory.  Keep in mind that later when you go to pick up an object, you must first remove it from this point cloud so it is removed from the collision map!
8. Rotate the robot to generate collision map of table sides. This can be accomplished by publishing joint angle value(in radians) to `/pr2/world_joint_controller/command`
9. Rotate the robot back to its original state.
10. Create a ROS Client for the “pick_place_routine” rosservice.  In the required steps above, you already created the messages you need to use this service. Checkout the [PickPlace.srv](https://github.com/udacity/RoboND-Perception-Project/tree/master/pr2_robot/srv) file to find out what arguments you must pass to this service.
11. If everything was done correctly, when you pass the appropriate messages to the `pick_place_routine` service, the selected arm will perform pick and place operation and display trajectory in the RViz window
12. Place all the objects from your pick list in their respective dropoff box and you have completed the challenge!
13. Looking for a bigger challenge?  Load up the `challenge.world` scenario and see if you can get your perception pipeline working there!

## [Rubric](https://review.udacity.com/#!/rubrics/1067/view) Points

---
### Writeup / README

### Exercise 1, 2 and 3 pipeline implemented
#### 1. Complete Exercise 1 steps. Pipeline for filtering and RANSAC plane fitting implemented.

Please click <a href="https://github.com/carldgosselin/robotics/blob/master/Project%203%20-%20RoboND-3D-Perception/Exercises%201%202%203/Exercise1%20-%20tabletop%20segmentation%20code%20and%20pics.md">here</a> for link to Exercise 1 documentation.

#### 2. Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.  

Please click <a href="https://github.com/carldgosselin/robotics/blob/master/Project%203%20-%20RoboND-3D-Perception/Exercises%201%202%203/Exercise2%20-%20Euclidean%20Clustering%20with%20ROS%20and%20PCL%20-%20code%20and%20pics.md">here</a> for link to Exercise 2 documentation.

#### 3. Complete Exercise 3 Steps.  Features extracted and SVM trained.  Object recognition implemented.

Please click <a href="https://github.com/carldgosselin/robotics/blob/master/Project%203%20-%20RoboND-3D-Perception/Exercises%201%202%203/Exercise3%20-%20object%20recognition%20code%20and%20pics.md">here</a> for link to Exercise 3 documentation.

### Pick and Place Setup

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.

# test1.world

Confusion matrix for test1 world
<div align=center>
	<img src="misc_images/test_world_1.PNG">	
</div>
<br>

SVM score for test1 world
<div align=center>
	<img src="misc_images/test_world_1_svm.PNG">	
</div>
<br>

Object recognition for test1 world
<div align=center>
	<img src="misc_images/3D Perception Object Recognition 1.PNG"> <br>
	3 out of 3	
</div>
<br>

# test2.world

Confusion matrix for test2 world
<div align=center>
	<img src="misc_images/test_world_2.PNG">	
</div>
<br>

SVM score for test2 world
<div align=center>
	<img src="misc_images/test_world_2_svm.PNG">	
</div>
<br>

Object recognition for test2 world
<div align=center>
	<img src="misc_images/3D Perception Object Recognition 2.PNG"> <br>
	5 out of 5	
</div>
<br>

# test3.world

Confusion matrix for test3 world
<div align=center>
	<img src="misc_images/test_world_3.PNG">	
</div>
<br>

SVM score for test3 world
<div align=center>
	<img src="misc_images/test_world_3_svm.PNG">	
</div>
<br>

Object recognition for test3 world
<div align=center>
	<img src="misc_images/3D Perception Object Recognition 3.PNG"> <br>
	5 out of 7	
</div>
<br>
<br>
<a href="https://github.com/carldgosselin/robotics/blob/master/Project%203%20-%20RoboND-3D-Perception/output_1.yaml">output_1.yaml</a>
<br>
<a href="https://github.com/carldgosselin/robotics/blob/master/Project%203%20-%20RoboND-3D-Perception/output_2.yaml">output_2.yaml</a>
<br>
<a href="https://github.com/carldgosselin/robotics/blob/master/Project%203%20-%20RoboND-3D-Perception/output_3.yaml">output_3.yaml</a>
<br>


Spend some time at the end to discuss your code, what techniques you used, what worked and why, where the implementation might fail and how you might improve it if you were going to pursue this project further. 

I liked how the course broke down the project into 3 incremental exercises.  I was able to gain a better understanding of the many moving parts to this project thanks to this division of labour.  

The implementation of this project might fail in the following ways:<br>
- The robot won't be able to detect the objects if they are completely obscured from it's vision.<br>
- The robot may not be able to recognize objects if the noise level goes above a certain threshold. <br>
- The robot will not recognize objects it hasn't been trained on. <br>

If I were to pursue this project further, I would look to improve my SVM scores as well as complete the additional challenge section of the project.





