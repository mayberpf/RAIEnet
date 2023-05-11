# Two_Pass_Algorithm
Connected-component labeling is used in computer vision to detect connected regions in binary digital images, although color images and data with higher dimensionality can also be processed.[1][2] When integrated into an image recognition system or human-computer interaction interface, connected component labeling can operate on a variety of information.[3][4] Blob extraction is generally performed on the resulting binary image from a thresholding step, but it can be applicable to gray-scale and color images as well. Blobs may be counted, filtered, and tracked.

## Algorithm

### One component at a time
This is a fast and very simple method to implement and understand. It is based on graph traversal methods in graph theory. In short, once the first pixel of a connected component is found, all the connected pixels of that connected component are labelled before going onto the next pixel in the image. This algorithm is part of Vincent and Soille's watershed segmentation algorithm, other implementations also exist.

In order to do that a linked list is formed that will keep the indexes of the pixels that are connected to each other, steps (2) and (3) below. The method of defining the linked list specifies the use of a depth or a breadth first search. For this particular application, there is no difference which strategy to use. The simplest kind of a last in first out queue implemented as a singly linked list will result in a depth first search strategy.

It is assumed that the input image is a binary image, with pixels being either background or foreground and that the connected components in the foreground pixels are desired. The algorithm steps can be written as:

1. Start from the first pixel in the image. Set current label to 1. Go to (2).
2. If this pixel is a foreground pixel and it is not already labelled, give it the current label and add it as the first element in a queue, then go to (3). If it is a background pixel or it was already labelled, then repeat (2) for the next pixel in the image.
3. Pop out an element from the queue, and look at its neighbours (based on any type of connectivity). If a neighbour is a foreground pixel and is not already labelled, give it the current label and add it to the queue. Repeat (3) until there are no more elements in the queue.
4. Go to (2) for the next pixel in the image and increment current label by 1.

Note that the pixels are labelled before being put into the queue. The queue will only keep a pixel to check its neighbours and add them to the queue if necessary. This algorithm only needs to check the neighbours of each foreground pixel once and doesn't check the neighbours of background pixels.

### Two Pass

