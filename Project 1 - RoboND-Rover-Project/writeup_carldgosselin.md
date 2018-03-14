# Project 1: Search and Sample Return

## [Project Rubric](https://review.udacity.com/#!/rubrics/916/view)

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

### Notebook Analysis
#### 2. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded).
####Add/modify functions to allow for color selection of obstacles and rock samples.

- My first modification to the code was to redirect the path variable to my own dataset of pictures that I captured with the rover <br>
[add code here .../carl_dataset path] <br>
- I kept the code in the Calibration Data section intact.  I could have created a 'grid' picture of my own via the Rover app but decided to keep what was there already. <br>
- The grid image is required for the perspective transform function.  You could say that this is the initial step for the perspective transform step. <br>
[add snippet of code just for the grid pic and then add the grid pic just below it] <br>
- Also, the first step in color thresholding for rock identification is getting a screenshot of the actual rocks we will be searching for <br>
[ add snippet of code for the rock image and then add the picture of the rock image]


- Modified the perpsective transform function <br>
the function transforms the picture, from the ground, to a top-down view of the world. <br>
[Show pic of a ground picture then show pic of the same picture from a top-down view]



- Modified the Color thresholding function <br>
The purpose of the color thresholding function, for the purpose of this project, is two-fold.  First, it is to identify navigable terrain, the second is to identify obstacles.  In the second instance, obstacles will need to split between obstacles to avoid and obstacles that are rocks. <br>
[add pic showing normal image, then add pic showing results of color thresholding]

- Coordinate Transformations code stays intact <br>
image coordinates -> to rover coordinates -> and eventually to world coordinates







#### 3. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result.





### Autonomous Navigation and Mapping

#### 4. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.






#### 5. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  

**Note: running the simulator with different choices of resolution and graphics quality may produce different results, particularly on different machines!  Make a note of your simulator settings (resolution and graphics quality set on launch) and frames per second (FPS output to terminal by `drive_rover.py`) in your writeup when you submit the project so your reviewer can reproduce your results.**

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

1. Explore the `to_polar_coords` function to direct the rover to the most navigable terrain
