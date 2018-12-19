# Multi-Process-Fusion
Multi-Process Fusion: Visual Place Recognition Using Multiple Image Processing Methods
Paper under review for IEEE RAL. 
Copyright: Stephen Hausler

ChangeLog:
Currently at Revision 1.1.

191218: Added additional option to set the image resizing value for the CNN input. Set to 227 by 227 for HybridNet and 224 by 224 for Vgg-16.
131218: Made a collection of changes in response to reviewer feedback. In particular, the worstID algorithm has changed.
190918: fixed bug that caused error at end of dataset immediately prior to printing the precision-recall curves.



Requirements:

1) MATLAB 2017 or later.
2) MATLAB Neural Network Toolbox and Image Processing Toolbox.
3) Computer with stand-alone graphics hardware.
4) A downloaded CNN caffemodel and prototxt (not included in this repository).


Getting Started:

1) Obtain CNN model files. For HybridNet, permission must be attained from the original author. Other networks will also work, such as VGG-16 trained on Places365.
2) To begin, launch MATLAB and open Multi_Process_Fusion.m.
3) Edit the adjustable settings for your particular dataset. Requires a collection of individual images.
4) Run the file. Note: the reference traverse will take several minutes with no intermediate feedback. Once the query traverse begins, a figure will display the recognition process.


Detailed instructions for testing on the St Lucia dataset:

1) Download the caffemodel and prototxt for the CNN model you wish to use (recommend HybridNet or VGG-16 trained on Places365).
2) The GPS .mat file for the St Lucia dataset is included in this repository. 
3) To download the St Lucia dataset, please go to https://wiki.qut.edu.au/display/cyphy/St+Lucia+Multiple+Times+of+Day and download "180809_1545" and "190809_0845". 
4) Extract individual frames out of the downloaded videos, for example, using Avconv on Ubuntu (https://libav.org/avconv.html). 
5) For "180809_1545", extract frames out of the video at 15FPS and limit to the first 4000 frames. Place these extracted images into a new folder containing just these images - this is the query dataset.
6) For "190809_0845", extract frames out of the video at 15FPS and limit to the first 3945 frames. Place these extracted images into a new folder containing just these images - this is the reference dataset. The number of images is set such that there is only one query image per location and no double-ups. The code will still work with double-ups, however the performance will drop as the matching scoring algorithm will find these double-ups and assume that severe perceptual aliasing is present. 
5) Then edit "Multi_Process_Fusion.m" and rename "Ref_folder" and "Query_folder" to point to the file locations where you saved the reference and query dataset images. Also edit "GT_file" to point to the save location of the GPS .mat file.
6) Edit "datafile" and "protofile" to point to the file locations where you saved your caffemodel and prototxt files. Then edit actLayer for the layer you wish to extract features from. Recommend setting to 15 for HybridNet and 24 for VGG-16.
7) Other settings can be left as-is, however experimentation can be made by varying different settings, such as the minimum and maximum sequence length, the quality rate-of-change threshold, and the Rwindow value. The chosen CNN layer can also be changed for more experimentation.


Acknowledgements:

MATLAB Libaries: MATLAB
patchNormalizeHMM: Niko Sunderhauf copyright 2013
sort_nat: Douglas M. Schwarz copyright 2008
Hybrid Net (not included in this release): Zetao Chen 2017

