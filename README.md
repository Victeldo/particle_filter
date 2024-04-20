# Particle Filter Implementation
## By Prithvi Shahani

This is the README for the Particle Filter


## Requisites + SETUP

Python version 3.9

The following packages must be pre-installed:
numpy
cv2
matplotlib
math
skimage


## SETUP

Run the particle_filter.py to run the simulation.

Switching the image:
Uncomment the desired image and leave the other images commented out under the chosen images section

Switching the resampling:

In the particle filter functions, uncomment the desired resample() function and comment out the other resample function()

Switching the posterior function:

In the particle filter functions, uncomment the desired determine_ParticleWeights() function and comment out the other determine_ParticleWeights() function

## Files

particle_filter.py

README.md

cs141_hw2.pdf (writeup)

demo.mp4 (extra credit demonstration of particle filter in action)

MarioMap.png
CityMap.png
BayMap.png

figures folder has the following files:
Figure 2: avg 100 trials, bay map, color histogram + systematic resampling
Figure 3: avg 100 trials, bay map, color histogram + roulette
Figure 4: avg 100 trials, bay map, ssim + roulette
Figure 5: avg 100 trials, bay map, ssim + systematic resampling

Figure 7: avg 100 trials, city map, ssim + systematic resampling 
Figure 8: avg 100 trials, city map, color histogram + systematic resampling 
Figure 9: avg 100 trials, city map, color histogram + roulette resampling 
Figure 10: avg 100 trials, city map, ssim + roulette resampling

Figure 11: avg 100 trials, Mario map, ssim + roulette resampling
Figure 12: avg 100 trials, Mario map, ssim + systematic resampling
Figure 13: avg 100 trials, Mario map, color histogram + systematic resampling
Figure 14: avg 100 trials, Mario map, color histogram + roulette resampling

figures/bonus folder has the following files:
Fig20 noise is at 3. Its uniform noise color histogram
Fig21 noise is at 20. Its uniform noise color histogram
Fig22 noise is at 3. Its uniform noise ssim
Fig24 noise is at 20. Its uniform noise ssim

