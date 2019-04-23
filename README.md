# Nueral Style Transfer
## Objectives
* Implement neural style transfer to change the style of art paintings.

## Packages Used
* Python 3.6
* [OpenCV](https://docs.opencv.org/3.4.4/) 4.0.0
* [keras](https://keras.io/) 2.1.0
* [Tensorflow](https://www.tensorflow.org/install/) 1.13.0
* [cuda toolkit](https://developer.nvidia.com/cuda-toolkit) 10.0
* [cuDNN](https://developer.nvidia.com/cudnn) 7.4.2
* [NumPy](http://www.numpy.org/)
* [SciPy](https://www.scipy.org/scipylib/index.html)

## Approaches
The `style_transfer.py` ([check here](https://github.com/meng1994412/Neural_style_transfer)) is used to store any parameters, including input image path, content and style layers, weight values, and etc and to proceed style transfer.

The `neuralstyle.py` ([check here](https://github.com/meng1994412/Neural_style_transfer/blob/master/pipeline/nn/conv/neuralstyle.py)) under `pipeline/nn/conv/` defines a class that encapsulates all logic required to apply neural style transfer to a content image and style image.

`VGG19` pre-trained on ImageNet is used to apply neural style transfer in this project. The loss function used is a three-component loss function including content loss, style loss and total-variation loss.

### Content Loss
For building the content loss function, we can select just one high-level layer of the network as content loss, which is `block4_conv2` in `VGG19`. `L2-norm` is used between content image and output image. We can force the output image to have similar structural content (but not necessarily similar style) as output image by minimizing the `L2-norm`.

### Style Loss
Unlike content loss, the style loss uses multiple layers in order to construct a multi-scale representation of style and texture. Obtain multi-scale representations can help to capture the style at low-level layers, mid-level layers, and high-level layers. During the training process, the goal is to minimize the loss (`L2-norm` again) between the style of output image and the style of style image in order to force the style of the output image to correlate with the style of the style image. And Gram matrix is used to compute the correlations between activations of layers.

### Total-Variation Loss
The total-variation loss operates only on output image. It encourages spatial smoothness through out the output image.

## Results
Here are three sample results of neural style transfer. In Figure 1, the content image (top left) is a photo of London Bridge, the style image (top right) is The Starry Night, by Van Gogh, and the resulting image (bottom) London Bridge in The Starry Night style.

<img align="left" src="https://github.com/meng1994412/Neural_style_transfer/blob/master/inputs/london_bridge.jpg" height="250"> <img align="right" src="https://github.com/meng1994412/Neural_style_transfer/blob/master/inputs/starry_night.jpg" height="250">


<p align="center">
  <img src="https://github.com/meng1994412/Neural_style_transfer/blob/master/results/london_bridge_in_starry_night.png" height="250">
</p>

Figure 1: Cotent image (top left), style image (top right), and resulting image (bottom).

In Figure 2, the content image (top left) is a painting of evening in Annecy, the style image (top right) is pencil sketch of Westmuir, and resulting image (bottom) is Annecy in pencil sketch style.

<img align="left" src="https://github.com/meng1994412/Neural_style_transfer/blob/master/inputs/evening_in_annecy.jpg" height="250"> <img align="right" src="https://github.com/meng1994412/Neural_style_transfer/blob/master/inputs/westmuir_sketch.jpg" height="250">


<p align="center">
  <img src="https://github.com/meng1994412/Neural_style_transfer/blob/master/results/annecy_sketch.png" height="250">
</p>

Figure 2: Cotent image (top left), style image (top right), and resulting image (bottom).

In Figure 3, the content image (top left) is a photo of Eiffel Tower, the style image (top right) is Haystacks, End of Summer, by Claude Monet, and the result image (bottom) is Eiffel Tower in Monet style.

<img align="left" src="https://github.com/meng1994412/Neural_style_transfer/blob/master/inputs/eiffel_tower2.jpg" height="240"> <img align="right" src="https://github.com/meng1994412/Neural_style_transfer/blob/master/inputs/monet_haystacks.jpg" height="240">


<p align="center">
  <img src="https://github.com/meng1994412/Neural_style_transfer/blob/master/results/eiffel_tower_in_monet.png" height="240">
</p>

Figure 3: Cotent image (top left), style image (top right), and resulting image (bottom).
