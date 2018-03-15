# Project 1: Search and Sample Return

## [Project Rubric](https://review.udacity.com/#!/rubrics/916/view)

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

### Notebook Analysis
#### 2. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded).

####Add/modify functions to allow for color selection of obstacles and rock samples.

My first modification to the code was to redirect the path variable to my own dataset of pictures that I captured with the rover <br>
`path = '../carl_dataset/IMG/*'` <br>

I kept the code in the Calibration Data section intact.  I could have created a 'grid' picture of my own via the Rover app but decided to keep what was there already.
The grid image is required for the perspective transform function.  You could say that this is the initial step for the perspective transform step. <br>
[add snippet of code just for the grid pic and then add the grid pic just below it] <br>
```
example_grid = '../calibration_images/example_grid1.jpg'
example_rock = '../calibration_images/example_rock1.jpg'
grid_img = mpimg.imread(example_grid)
rock_img = mpimg.imread(example_rock)

fig = plt.figure(figsize=(12,3))
plt.subplot(121)
plt.imshow(grid_img)
plt.subplot(122)
plt.imshow(rock_img)
```
<br>
<div align=center>
	<img src="./calibration_images/example_grid1.jpg"> <br>
</div>
</br>

Also, the first step in color thresholding for rock identification is getting a screenshot of the actual rock we will be searching for <br>
<br>
<div align=center>
	<img src="./calibration_images/example_rock1.jpg "> <br>
</div>
</br>

**Modifications to the `perspect_transform` function** <br>
Purpose: This function transforms the picture, from the ground, to a top-down view of the world. <br>
<div align=center>
	<img src="./misc/field_of_view.png">
</div>
<br>

Lines of code were added in the **Perspective Transform** section to exclude the processing of pixels that were outside the field of view of the camera. <br>
```
def perspect_transform(img, src, dst):
	...
	mask = cv2.warpPerspective(np.ones_like(img[:,:,0]), M, (img.shape[1], img.shape[0])) #New
    return warped, mask #new
    

warped, mask = perspect_transform(grid_img, source, destination)
fig = plt.figure(figsize=(12,3)) #new
plt.subplot(121) #new
plt.imshow(warped)
plt.subplot(122) #new
plt.imshow(mask, cmap='gray') #new
```
<div align=center>
	<img src="./misc/rover_img.png"><img src="./misc/persp_img.png">
</div>
<br>


**Modifications to the `color_thresh` (color thresholding function)** <br>
Purpose:  The purpose of this function is to help identify navigable terrain versus obstacles (or non-navigable terrain). <br>
<div align=center>
	<img src="./misc/field_of_view.png"> <br>
	<img src="./misc/color_thresholding.png"> <br>
</div>
</br>
note:  A separate function will identify rocks.

Another note:  the entire **Coordinate Transformations** section in the notebook remains the same. <br>
This is the section of the notebook where the code converts image coordinates to rover coordinates and then to world coordinates.

**Added new section in notebook to find rocks** <br>
A new section was added in the notebook to identify rocks in the environment called `find_rocks`. <br>
To identify the rocks, the RGB values were set to 110, 110, 50 <br>
A rock would be identified when the red channel was greater than 110, the green channel greater than 110 and the blue channel less than 50 <br>
```
def find_rocks(img, levels=(110, 110, 50)):
    rockpix = ((img[:,:,0] > levels[0]) \
              & (img[:,:,1] > levels [1]) \
              & (img[:,:,2] < levels[2]))
    
    color_select = np.zeros_like(img[:,:,0])
    color_select[rockpix] = 1
    
    return color_select

rock_map = find_rocks(rock_img)
fig = plt.figure(figsize=(12,3))
plt.subplot(121)
plt.imshow(rock_img)
plt.subplot(122)
plt.imshow(rock_map, cmap='gray')

```
<div align=center>
	<img src="./misc/find_rocks.png">
</div>
</br>

#### 3. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result.

So to start off, at this point, we have all images ia a saved dataset called `data`. And now we want to process them. <br>
Once the images havs been processed, a library called `moviepy` will create a video out of the processed images.

Each image is processed by the `perspec_transform` and `color_thresh` functions <br>
```
warped, mask = perspect_transform(img, source, destination)
threshed = color_thresh(warped)
```
note:  `mask` was added to exclude pixels outside the camera view <br>

next, the threshold map is created <br>
`  obs_map = np.absolute(np.float32(threshed) - 1) * mask` <br>
the code will output '1' wherever the map shows a '0'(because of the -1 and absolute value) and then multiply by mask (which has '0' for pixels outside the view of the camera)<br>
The above gives you a map of where the obstacle pixels are, excluding the pixels that are outside the field of view. <br>

then, from the thresholded images, the navigable terrain identified from the images are converted to rover-centric coordinates <br>
`xpix, ypix = rover_coords(threshed)` <br>

aftwards, the world map is updated with obstacles, rock location, and navigable terrain. <br>
```
data.worldmap[y_world, x_world, 2] = 255
data.worldmap[obs_y_world, obs_x_world, 0] = 255

nav_pix = data.worldmap[:,:,2] > 0
    
data.worldmap[nav_pix, 0] = 0

rock_map = find_rocks(warped, levels=(110, 110, 50))
if rock_map.any():
    rock_x, rock_y = rover_coords(rock_map)
    rock_x_world, rock_y_world = pix_to_world(rock_x, rock_y, xpos,
                                                 ypos, yaw, world_size, scale)
    data.worldmap[rock_y_world, rock_x_world, :] = 255
```

<br>

In the worldmap, we want to identify 3 components with 3 different colors. <br>
We want to identify navigable terrain and add them to the blue channel <br>
`data.worldmap[y_world, x_world, 2] = 255`

We also want to identify obstacles and add them to the red channel <br>
`data.worldmap[obs_y_world, obs_x_world, 0] = 255` <br>

Next, the rock_map is updated when rocks are found. <br>

 ``` 
 rock_map = find_rocks(warped, levels=(110, 110, 50))
    if rock_map.any():
        rock_x, rock_y = rover_coords(rock_map)
        rock_x_world, rock_y_world = pix_to_world(rock_x, rock_y, xpos,
                                                 ypos, yaw, world_size, scale)
        data.worldmap[rock_y_world, rock_x_world, :] = 255
```
In the above code, `data.worldmap[rock_y_world, rock_x_world, :] = 255` adds the rock detection in green color.

The last section in `process_image` is to run `moviepy`to view the processed images in sequence for a movie-like view of the results. <br>

<div align=center>
	<a href="output/test_mapping.mp4">
		<img src="./misc/video_output.png">
	</a> <br>
	Click <a href="output/test_mapping.mp4">here</a> to view video output
</div>









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






