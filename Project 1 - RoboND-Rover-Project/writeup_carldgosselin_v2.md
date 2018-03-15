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


- Modified the perspective transform function <br>
the function transforms the picture, from the ground, to a top-down view of the world. <br>
[Show pic of a ground picture then show pic of the same picture from a top-down view]



- Modified the Color thresholding function <br>
The purpose of the color thresholding function, for the purpose of this project, is two-fold.  First, it is to identify navigable terrain, the second is to identify obstacles.  In the second instance, obstacles will need to split between obstacles to avoid and obstacles that are rocks. <br>
[add pic showing normal image, then add pic showing results of color thresholding]

- Coordinate Transformations code stays intact <br>
image coordinates -> to rover coordinates -> and eventually to world coordinates


- In the second time around, I added the lines of code in the perspective transform section to exclude any processing of areas of image that was simply out of view for the camera. <br>
- In other words, we don't want to map pixels that are zeros simply because they were outside the field of view of the camera. We just want to map pixels that are inside the field of view of the camera.<br>
[show code]

- On the second pass of the Jupyter Notebook, I also added a function to find rocks in the environment. <br>
- To identify the rocks, the RGB values were set to 110, 110, 50 <br>
- A rock would be identified when the red channel was greater than 110, the green channel greater than 110 and the blue channel less than 50 <br>
[show snippet of code]









#### 3. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result.

So to start off, at this point, we have all images ia a saved dataset call data. And now we want to process them <br>
and once the image has been process, a library called moviepy will process the images to make a video.  "Making a video of our processed image output" <br>

Applied perspective transforms and color threshold to each image <br>
[show code for this] <br>
- The above code now returns both warped and mask

then, created the threshold map, this includes the mask to exclude the pixels outside of the field of view of the rover.
[snippet of code]
- the code will output '1' wherever the map shows a '0'(because of the -1 and absolute value) and then multiply by mask (which has '0' for pixels outside the view of the camera)<br>
- so it basically gives you a map of where the obstacle pixels are, excluding the pixels that are outside the field of view.

then, from the thresholded images, convert navigable terrain to rover-centric coordinates <br>
[snippet of code] <br>


then, updating the world map with obstacles, rock location, and navigable terrain. <br>
[snippet of code] <br>

In the worldmap, we want to identify 3 components with 3 different colors <br>
- We want to identify obstacles and add them to the red channel 
[show snippet of code] 

- We want to identify rocks and add them to the green channel <br>
[show snippet of code]

- We want to identify navigable terrain and add them to the blue channel <br>
[show snippet of code]


- Next, we want to update the rock_map to capture when we've found rocks. <br>
[show snippet of code]


- Then, i used the existing code to run moviepy to view the processed images in sequence for a movie-like view of the results. <br>
[show pic of video - and link to real video] <br>
- Same code expcet the overlay code will actually have something to show when the video runs.




### Autonomous Navigation and Mapping

#### 4. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.


talk about the databucket class.  Data is the global object will all of the data... how we use the robot log? video 5:10 <br>
also reading in map of the world or ground truth map_bw.png <br>

At this point, I'm able to take the code from the process_image() steps and place it (with a few changes) to the perception_step() in the `perception.py` script <br>

Existing code... <br>
drive_rover.py  - script that calls all other scripts <br>
- will need to talk about nav_angles and nav_dists <br>

- line 11 executes the perception step.  This triggers all of the image processing functionality <br>
- line 112 executes the decision step.  This triggers the movement of the rover (and also has starting code for picking up rocks)

supportingFunction.py
- this updates all of the values of the Rover object
- It also updates all of the writing at every step

Now, let's look at perecption.py first... <br>

- Perception comes prepolutated with all of the functions we found in the notebook <br>
- The processImage function in the notebook is called perceptionStep in the 

Video 27:16

Different as the perception step takes in the Rover object (as opposed to the Data object in the notebook) <br>
- The Rover object gives us access to all things Rover, such at the images the rover is seeing, the position, the yaw, etc... <br>
- I'm going to use the image in particular to do some analysis on things like driveable terrain versus avoiding obstacles <br>
- Then store the output of the perception into various fields inside the rover such as color-thresholded images for obstacles and color thresholded images for navigable terrain. <br>
- Then, feed the steering variables such as nav_angles to steer the rover.  If this is not updated, the rover will drive into a straight line until it bumps into an object. <br>

modification 1 <br>
- similar to the changes in the notebook, I added the perspect_tranform function the mask variable to identify the pixels in the image that are out of view of the camera <rb>
- also, similar to the changes in the notebook, I added the find_rocks function to find and tally the rocks that are found on the captured images <br>

Within the 'perception_step' <br>
- execute all of the same functions that we're added to the notebook in 'process_image' <br>
- so, defined the 'perspective_transform' points (source and destination) <br>
[snippet of code] <br>
- performed a perspective transform on the image <br>
[snippet of code] <br>
- then, threshold the warped image<br>
- then, create the obstacle image map based on the image inputs <br>
- addd a few lines of code to show to the viewer what the 'perspect_tranform' looks like.   <br>
[snippet of code rover.vision_image]... multiply by 255 to see the colors <br>
- then, convert map image to rover coordinates <br>
[snippet of code] xpix, ypix <br>
- then convert to work pixels for navigable terrain and avoiding obstacles <br>

note:  I did not update the decision.py script

#### 5. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  

**Note: running the simulator with different choices of resolution and graphics quality may produce different results, particularly on different machines!  Make a note of your simulator settings (resolution and graphics quality set on launch) and frames per second (FPS output to terminal by `drive_rover.py`) in your writeup when you submit the project so your reviewer can reproduce your results.**

**Settings**
fps = 25 in simulator
fps = 60 in moviepy

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

1. Explore the `to_polar_coords` function to direct the rover to the most navigable terrain
2. Improve the overlap between navigable terrain and obstacles (data.worldmap[..., ...])






