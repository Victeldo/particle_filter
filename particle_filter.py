import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

########### hardcoded variables ###########
DIST = 50 # 1 unit of dist is 50 pixels
IM_DIA = 50 # image_size of drone (pixel)
GAUSS_NOISE = 0.5 # uncertaintly in movement
GAUSS_NOISE_P = 0.5 # resamplign noise
DRONE_NOISE = 3 # Hard coded gaussian noise of the drone reading pixels (if using noisy readings)
BINS_PER_CHANNEL = 32 # arbitrary bins per channel chosen for histogram comparison (more bins, longer runtime)


########### chosen image ###########
# im = cv2.imread('CityMap.png')
# im = cv2.imread('BayMap.png')
im = cv2.imread('MarioMap.png')

########### simulation environment ###########

class environment():
  def __init__(self, image):
    self.image = image
    self.margin = IM_DIA // 2
    self.height, self.width = im.shape[:2]
    self.x_range, self.y_range = self.width / DIST, self.height / DIST
    self.center_x, self.center_y = self.x_range // 2, self.y_range // 2
  
  # mainly for initializating the drone position
  def get_map_info(self):
    return self.margin, self.center_x, self.center_y

  def get_range(self):
    return self.x_range

  def loc(self, image_at_step, drone_pos, circ_size, particle=False, weight=1):

    # scale position to pixel coords
    drone_pos_x, drone_pos_y = int((drone_pos[0] + self.center_x) * DIST), int((drone_pos[1] + self.center_y) * DIST) 

    if particle:
      cv2.circle(image_at_step, (drone_pos_x, drone_pos_y), radius=round(circ_size*weight), color=(255, 255, 0), thickness=2)
      # cv2.circle(image, (drone_pos_x, drone_pos_y), radius=round(x_range*1000*weight), color=(255, 255, 0), thickness=2)
    else:
      cv2.circle(image_at_step, (drone_pos_x, drone_pos_y), radius=circ_size, color=(255, 16, 240), thickness=-1)
      
    return image_at_step

  def is_valid_pos(self, pos):
    margin_adj = self.margin / DIST
    pos_x, pos_y = pos
    # return self.margin <= pos_x <= (self.center_x - self.margin) and self.margin <= pos_y <= (self.height - self.margin)
    return (margin_adj-self.center_x) <= pos_x <= (self.center_x-margin_adj) and (margin_adj-self.center_y) <= pos_y <= (self.center_y-margin_adj)

  def move_pos(self, movement_vec, pos, drone=False):
    new_pos = np.add(pos, movement_vec)
    if self.is_valid_pos(new_pos):

      noisy_pos = np.add(new_pos, np.random.normal(0, math.sqrt(GAUSS_NOISE)))

      if drone and self.is_valid_pos(noisy_pos):
        return noisy_pos
      
      else:
        return new_pos
      
    else:
      return pos

########### drone functions ###########
  
class drone():
  def __init__(self, margin, center_x, center_y):
    drone.pos = [np.random.uniform((margin/DIST)-(center_x), center_x -(margin/DIST)), 
                 np.random.uniform((margin/DIST)-(center_y), center_y -(margin/DIST))] # accounting for edge of image
    
  def set_pos(self, pos):
    self.pos = pos

  def get_pos(self):
    return self.pos
  
  def generate_move(self):
    angle = np.random.uniform(0, 2* np.pi)
    dx = np.cos(angle)
    dy = np.sin(angle)
    return [dx, dy]
  
  def obs(self, margin, center_x, center_y):
    observation_image = np.zeros((IM_DIA,IM_DIA,3))

    # scale position to pixel coords
    drone_pos_x, drone_pos_y = int((self.pos[0] + center_x) * DIST), int((self.pos[1] + center_y) * DIST)

    # calculate top left of image
    top_left_x = drone_pos_x - margin
    top_left_y = drone_pos_y - margin

    # calculate bottom right of image
    bottom_right_x = top_left_x + 2*margin
    bottom_right_y = top_left_y + 2*margin

    # extract observed image from the image
    observation_image = im[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    relative_drone_pos_x = drone_pos_x - top_left_x
    relative_drone_pos_y = drone_pos_y - top_left_y

    if observation_image.shape != np.zeros((IM_DIA,IM_DIA,3)).shape:
      print(observation_image.shape)
      print(drone_pos_x, drone_pos_y)

    return observation_image


########### particle functions ###########
  
class Particle():
  def __init__(self, location):
    self.location = location
    self.weight = 1    # all particles initialized to weight 1
  
  def set_weight(self, weight):
    self.weight = weight
  
  def get_weight(self):
    return self.weight
  
  def get_location(self):
    return self.location
  
  def set_location(self, location):
    self.location = location
  
  def is_valid_pos(self, margin, center_x, center_y):
    margin_adj = margin / DIST
    pos_x, pos_y = self.location
    return (margin_adj-center_x) <= pos_x <= (center_x-margin_adj) and (margin_adj-center_y) <= pos_y <= (center_y-margin_adj)

  
  def obs(self, margin, center_x, center_y):
    observation_image = np.zeros((IM_DIA,IM_DIA,3))

    # scale position to pixel coords
    drone_pos_x, drone_pos_y = int((self.location[0] + center_x) * DIST), int((self.location[1] + center_y) * DIST)

    # calculate top left of image
    top_left_x = drone_pos_x - margin
    top_left_y = drone_pos_y - margin

    # calculate bottom right of image
    bottom_right_x = top_left_x + 2*margin
    bottom_right_y = top_left_y + 2*margin

    # extract observed image from the image
    observation_image = im[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    relative_drone_pos_x = drone_pos_x - top_left_x
    relative_drone_pos_y = drone_pos_y - top_left_y

    return observation_image
    

########### particle filter functions ###########

class ParticleFilter():
  def __init__(self, margin, center_x, center_y, size=1000):
    self.ParticleList = []
    for i in range(size):
      init_loc = [np.random.uniform((margin/DIST)-(center_x), center_x + 1-(margin/DIST)), 
                 np.random.uniform((margin/DIST)-(center_y), center_y + 1-(margin/DIST))]
      self.ParticleList.append(Particle(init_loc))

  def get_ParticleList(self):
    return self.ParticleList

  def normalize_ParticleWeights(self):
    totalweight = 0
    for particle in self.ParticleList:
      totalweight += particle.get_weight()
    if totalweight != 0:
      for particle in self.ParticleList:
        normalweight = (particle.get_weight()/totalweight)
        particle.set_weight(normalweight)
    else:
        uniform_weight = 1.0 / len(self.ParticleList)
        for particle in self.ParticleList:
            particle.set_weight(uniform_weight)
  
  # color histogram comparison approach
  def determine_ParticleWeights(self, obs_image, margin, center_x, center_y):
    for particle in self.ParticleList:
      if particle.is_valid_pos(margin, center_x, center_y):
        p_image = particle.obs(margin, center_x, center_y)

        
        obs_rgb       = cv2.cvtColor(obs_image, cv2.COLOR_BGR2RGB)
        particle_rgb  = cv2.cvtColor(p_image, cv2.COLOR_BGR2RGB)

        obs_hist      = cv2.calcHist([obs_rgb], [0,1,2], None, [BINS_PER_CHANNEL, BINS_PER_CHANNEL, BINS_PER_CHANNEL], [0,256,0,256,0,256])
        particle_hist = cv2.calcHist([particle_rgb], [0,1,2], None, [BINS_PER_CHANNEL, BINS_PER_CHANNEL, BINS_PER_CHANNEL], [0,256,0,256,0,256])

        # normalize the color histograms
        obs_hist      = cv2.normalize(obs_hist, obs_hist).flatten()
        particle_hist = cv2.normalize(particle_hist, particle_hist).flatten()

        relation      = cv2.compareHist(obs_hist, particle_hist, cv2.HISTCMP_CORREL)

        # discard negative correlation as it doesn't imply close colors (and squaring will cause resampling to take the higher weight)
        weight = max(0, relation)
        weight = weight**2

        particle.set_weight(weight)
      else:
        particle.set_weight(0)
    self.normalize_ParticleWeights()
    
  # structural similarity approach (if particle image is out of bounds, give it a weight of 0)

  # def determine_ParticleWeights(self, obs_image, margin, center_x, center_y):
  #   for particle in self.ParticleList:
  #     if particle.is_valid_pos(margin, center_x, center_y):

  #       p_image = particle.obs(margin, center_x, center_y)

  #       obs_gray       = cv2.cvtColor(obs_image, cv2.COLOR_BGR2GRAY)
  #       particle_gray  = cv2.cvtColor(p_image, cv2.COLOR_BGR2GRAY)

  #       (score, diff) = ssim(obs_gray, particle_gray, full=True)
  #       diff = (diff * 255).astype("uint8")

  #       # discard negative correlation as it doesn't imply close colors (and squaring will cause resampling to take the higher weight)
  #       weight = max(0, score)
  #       weight = weight**2
  #       particle.set_weight(weight)
  #     else:
  #       particle.set_weight(0)

  #   self.normalize_ParticleWeights()
  
  # roulette wheel approach + add resampling noise to avoid sample degeneracy
  def resample(self):
    new_particles = []

    # using cumulative sum
    total_weight = [particle.get_weight() for particle in self.ParticleList]
    # getting particle at index i (for actual resampling)
    particle_locations = [p.get_location() for p in self.ParticleList]

    for i in range(len(self.ParticleList)):
      # indexed roulette selection
      r = np.random.uniform()
      # summing up until find the index of the particle's weight at the cumulative sum
      weight_sum = 0
      j = 0

      while weight_sum < r:
        weight_sum += self.ParticleList[j].get_weight()
        j += 1
      j -= 1

      # add noise to particle
      pos_x, pos_y = particle_locations[j]
      pos_x += np.random.normal(0, math.sqrt(GAUSS_NOISE_P))
      pos_y += np.random.normal(0, math.sqrt(GAUSS_NOISE_P))
      new_particles.append(Particle([pos_x, pos_y]))

    self.ParticleList = new_particles
  
  # systematic resampling approach
  # def resample(self):
  #   new_particles = []
  #   total_weight = sum(p.get_weight() for p in self.ParticleList)
  #   particle_weights = [p.get_weight() for p in self.ParticleList]

  #   # Systematic resampling with equal spacing
  #   step = total_weight / len(self.ParticleList)
  #   u = np.random.uniform(0, step)

  #   i = 0
  #   j = 0
  #   current_weight = 0
  #   while j < len(self.ParticleList):
  #     current_weight += particle_weights[i]
  #     if current_weight >= u + j * step:
  #       pos_x, pos_y = self.ParticleList[i].get_location()
  #       pos_x += np.random.normal(0, math.sqrt(GAUSS_NOISE_P))
  #       pos_y += np.random.normal(0, math.sqrt(GAUSS_NOISE_P))
  #       new_particles.append(Particle([pos_x, pos_y]))
  #       j += 1
  #       if j < len(self.ParticleList):
  #         u = current_weight - particle_weights[i]
  #     i = (i + 1) % len(self.ParticleList)  # Wrap around the list

  #   self.ParticleList = new_particles

  def move_particles(self, movement_vec):
    for particle in self.ParticleList:
      pos = particle.get_location()
      new_pos = np.add(pos, movement_vec)
      # particles that go out of bounds have 0 weight
      particle.set_location(new_pos)
  
  # returns average location of all particles at a time step for experimental purposes
  def avg_location(self):
    avgLocation = 0
    for particle in self.ParticleList:
      avgLocation += particle.get_location()
    
    avgLocation = avgLocation / len(self.ParticleList)

########### simulation ###########
# env = environment(im)
# m, x, y = env.get_map_info()
# agent = drone(m, x, y)
# filter = ParticleFilter(m, x, y, size=100)

# trials
n = 100
timesteps = 100
# arrays for the plots
all_drone_positions = []
all_particle_positions = []


for j in range(n):
  env = environment(im)
  m, x, y = env.get_map_info()
  agent = drone(m, x, y)
  filter = ParticleFilter(m, x, y, size=100)

  drone_positions = []
  particle_positions = []

  # for n timesteps
  for i in range(timesteps):
    obs_im = agent.obs(m, x, y)
    image_at_step = im.copy()
    ParticleList = filter.get_ParticleList()
    filter.determine_ParticleWeights(obs_im, m, x, y)
    for particle in ParticleList:
      temp_pos = particle.get_location()
      image_at_step = env.loc(image_at_step, temp_pos, 10, particle=True, weight=int(particle.get_weight()*env.get_range()))
    drone_pos = agent.get_pos()
    image_at_step = env.loc(image_at_step, drone_pos, 10)
    filter.resample()

    # to visualize the particle filter in action
    # cv2.imshow('Map image',image_at_step)
    # cv2.waitKey(0) # draw circle for position at each time step after pressing enter

    movement_vec = agent.generate_move()
    agent.set_pos(env.move_pos(movement_vec, drone_pos, drone=True))
    filter.move_particles(movement_vec)

    drone_positions.append(agent.get_pos())
    particle_positions.append([p.get_location() for p in filter.get_ParticleList()])
  
  all_drone_positions.append(drone_positions)
  all_particle_positions.append(particle_positions)

avg_drone_positions = np.mean(all_drone_positions, axis=0)
avg_particle_positions = np.mean(all_particle_positions, axis=0)

# cv2.destroyAllWindows() 

# Plotting functions
  
mse_values = []
for drone_pos, particle_pos in zip(avg_drone_positions, avg_particle_positions):
    # Compute MSE between drone position and particle positions
    mse = np.mean([mean_squared_error(drone_pos, particle_pos_single) for particle_pos_single in particle_pos])
    mse_values.append(mse)

# Plotting the MSE values
plt.plot(range(1, len(mse_values) + 1), mse_values, label='Mean Squared Error')
plt.xlabel('Time Step')
plt.ylabel('MSE')
plt.title('Mean Squared Error between Drone Position and Particle Positions')
plt.legend()
plt.grid(True)
plt.show()
