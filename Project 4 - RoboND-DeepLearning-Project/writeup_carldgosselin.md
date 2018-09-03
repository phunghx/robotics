## Project: Follow-me drone

I'll start this report with the final grade score:
```
In [63]: # And the final grade score is 
final_score = final_IoU * weight
print(final_score)
```
Result: 0.41784247025954463

For this project, achieving a final grade score over 0.40 (40%) was made possible with the following Neural Network parameters:
```
learning_rate = 0.0005
batch_size = 64
num_epochs = 50
steps_per_epoch = 72
validation_steps = 50
workers = 2
``` 

### Neural Network parameters - explained:<br>
**1. Learning Rate - The rate at which 'old beliefs' are abandoned for new ones.**<br>
At the start of the project, the learning rate was set to 0.  During my experimentation with the code, I set the learning rate to 0.001.  I eventually reduced the learning rate to 0.0007 and then finally to 0.0005.  This was the last variable changed in order to achieve the final grade score above 0.40.<br>

**2. Batch Size - number of training samples/images that get propagated through the network in a single pass.**<br>
The batch size was also set to 0 at the start of the project.  I set this number to 64.<br>

**3. Number of Epoch - number of times the entire training dataset gets propagated through the network.**<br>
This variable was also set to 0 at the beginning of the project.  I went as I up as 100 for this variable and eventually settled on 50.  I reduced this number to prevent overfitting the data and focused my attention a little more on the learning rate to achieve the final grade score above 0.40.<br>

**4. Steps per Epoch - number of batches of training images that go through the network in 1 epoch.**<br>
This variable was also set to 200.  Similar to the "Number of Epoch" variable, I decided to reduce the number of steps to 72 to prevent overfitting the data and focused my attention a little more on the learning rate to achieve the final grade score above 0.40.<br>

**5. Validation Steps - number of batches of validation images that go through the network in 1 epoch.**<br>
This variable was initially set to 50.  I left this variable intact.<br>

**6. Workers - maximum number of processes to spin up. This can affect your training speed and is dependent on your hardware.**<br>
This variable was initially set to 2.  I left this variable intact as my hardware seemed to run the exercise without any major delays.<br>

# Network Architecture - Explained

Encoder blocks, a 1x1 convolution, and decoder blocks are used for the Fully Convolutional Network (FCN).

The encoder block takes in the image data for processing within two functions to produce the output layer: separable_conv2d_batchnorm and separable_conv2d_batchnorm.

## Encoder block

```
def  encoder_blockencoder (input_layer, filters, strides):
    
    # TODO Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    
    return output_layer
```
...
```
def separable_conv2d_batchnorm(input_layer, filters, strides=1):
    output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides,
                             padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer
```

After the encoding block and 1x1 convolution block are executed, the decoder block is used to extract/decode the encoded data with additional spatial information.

## Decoder block

```
def decoder_block(small_ip_layer, large_ip_layer, filters):
    
    # TODO Upsample the small input layer using the bilinear_upsample() function.
    upsample = bilinear_upsample(small_ip_layer)
    
    # TODO Concatenate the upsampled and large input layers using layers.concatenate
    concat_layer = layers.concatenate([upsample, large_ip_layer])

    # TODO Add some number of separable convolution layers
    output_layer = separable_conv2d_batchnorm(concat_layer, filters)
    
    return output_layer
```
...
```
def bilinear_upsample(input_layer):
    output_layer = BilinearUpSampling2D((2,2))(input_layer)
    return output_layer
```
...
```
def separable_conv2d_batchnorm(input_layer, filters, strides=1):
    output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides,
                             padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer
```

## Conclusion

As with any machine learning algorithm, data is key.  To improve the scores and predictability of this exercise, the quality of images for both training and validation sets is key.  Capturing quality images of the Hero and the absence of the Hero will improve the scores of this exercise.  In addition, getting a variation of images that are close-up and far-away would also improve the scores of this exercise.