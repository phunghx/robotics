# UDACITY - Robotics Software Engineer Nanodegree

**Term** 1 <br>
1 Introduction​ ​to​ ​Robotics <br>
2 ROS​ ​Essentials <br>
3 Kinematics <br>
4 Perception <br>
5 Controls <br>
6 Deep Learning <br>

**Term** 2 <br>
7 Introduction <br>
8 Robotic​Systems​Deployment <br>
9 Localization <br>
10 SLAM <br>
11 Reinforcement​Learning​for​Robotics <br>
12 Path​Planning​and​Navigation

## Project:  Search and Sample Return
In this project, you will write code to autonomously map a simulated environment and search for samples of interests.

## Install
Installation links: <br>
Link 1 <br>
Link 2 <br>
Link 3 <br>
...

## Code
Open file.py

## Run
Make sure you are in the top-level project directory smartcab/ (that contains this README). Then run: <br>
python folder/file.py <br>

## Project Rubric

**Writeup** <br>
__Criteria__: Provide a writeup (aka this README file) <br>
**Meets Specifications:**  The README should include a statement and supporting figures / images that explain how each rubric item was addressed, and specifically where in the code each step was handled.

**Notebook Analysis**<br>
**Criteria:** Run the functions provided in the notebook on test images (first with test data provided, next on data you have recorded).  Add/modify functional to allow for color selection of obstacles and rock samples.<br>
**Meets Specifications:** Describe in your writeup (and identify where in your code) how you modified or added functions to add obstacle and rock sample identification.

**Criteria:** Populate the `process_image ()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test ddata using the `moviepy` functions provided to create video output of your result.<br>
**Meets Specifications:** Describe in your writeup how you modified the `process_image()` to demonstrate your analysis and how you created a worldmap.  Include your video output with your submission.

**Criteria:** Fill in the `perception_step()` (at the bottom of the `perception.py` script) in (`decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.<br>
**Meets Specifications:** `perception_step()` and `decision_step()` functions have been filled in and their functionality explained in the writeup.

**Criteria:** Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.<br>
**Meets Specifications:**  By running `drive.py` and launching the simulator in autonomous mode, your rover does a reasonably good job at mapping the environment.

The rover must map at least 40% of the environment with 60% fidelity (accuracy) against the ground truth.  You must also find (map) the location of at least one rock sample.  They don't need to pick up any rocks up, just have them appear in the map (should happen automatically if their map pixels in `Rover.worldmap[:,:,1]` overlap with sample locations.)

Note: running the simulator with different choices of resolution and graphics may produce different results, particularly on different machines!  Make note of your simulator settings (resolution and graphics quality set on launch) and frames per second (FPS output to erminal by `drive_rover.py`) in your writeup when you submit the project so your reviewer can reproduce your results. 

**Suggestions to Make Your Project Stand Out!:** To have a standout submission on this project you should not only successfully navigate and map the environment in "Autonomous Mode", but also figure out how to collect the samples and return them to the starting point (middle of the map) when you have found them all!







