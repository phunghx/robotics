## Project: Follow-me drone

I'll start this report with the final score:
```
In [63]: # And the final grade score is 
final_score = final_IoU * weight
print(final_score)
```
Result: 0.41784247025954463

# Network Architecture - Explained
[convey an understanding of the network architecture]
- Explain each layer of the network architecture
- Explain the role that it plays in the overall network
- Demonstrate the benefit/drawback of different network architectures pertaining to this project
- Justify the currentnetowrk with fatual data
- Provide graph, table, illustration or figure to serve as reference

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

# Neural Network Parameters - Explained
[convey an understanding of the parameters chosen]

```
learning_rate = 0.0005
batch_size = 64
num_epochs = 50
steps_per_epoch = 72
validation_steps = 50
workers = 2
```

# 1 by 1 convolutions - Explained


# Fully Connected Layer - Explained


# Image Manipulation - Explained
- Encoding images
- Decoding images


# Limitations to the neural network given the data


# Model
- Score greater or equal to 40% (0.40)