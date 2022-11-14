import numpy as np 
import matplotlib.pyplot as plt
import random


def get_closest_position_in_list(position_list : list, pos : tuple) -> tuple:
  """
  Finds the position which have the shortest position from a current position

  Input parameters:
    position_list : list of positions to calculate distance of
    pos           : position to find the distance with respect to 
  """
  assert position_list, "List must be non-empty"

  shortest_distance_pos : tuple = position_list[0]
  shortest_distance : float = np.linalg.norm(np.array(shortest_distance_pos) - np.array(pos))

  for position in position_list:
    distance : float = np.linalg.norm(np.array(position) - np.array(pos)) # Will calculate the first position twice
    if distance < shortest_distance:
      shortest_distance = distance
      shortest_distance_pos = position

  return shortest_distance_pos


def obstacle_free_RRT(xy_0 : tuple, xy_target : tuple, distance_target_hit : float, world_dimensions : tuple) -> tuple:
  """
  Perform the RRT-algorithm without any obstacles
  
  Input parameters:
    xy_0                : initial position
    xy_target           : target position
    distance_target_hit : distance to assume the target has been achieved
    world_dimensions    : size of the world in (max_height, max_width)

  Output:
    list of coordinates tested before achieving the target 
  """

  max_height, max_width = world_dimensions
  assert (xy_0[0] <= max_width and xy_0[1] <= max_height) and (xy_0[0] >= 0 and xy_0[1] >= 0), "Dimensions must fit"
  assert (xy_target[0] <= max_width and xy_target[1] <= max_height) and (xy_target[0] >= 0 and xy_target[1] >= 0), "Dimensions must fit"

  positions_list : list = [xy_0]
  edges_list : list = []

  target_achieved : bool = False 

  while not target_achieved:
    # Create random position
    random_x : int = random.randint(0, max_width)
    random_y : int = random.randint(0, max_height)
    random_pos : tuple = (random_x, random_y) 

    if random_pos in positions_list:
      continue

    # Get the closest position and create an edge between these
    closest_pos : tuple = get_closest_position_in_list(positions_list, (random_x, random_y))
    edges_list.append((closest_pos, random_pos))
    positions_list.append(random_pos)
    
    # Check if target reached
    distance_to_target : float = np.linalg.norm(np.array(random_pos) - np.array(xy_target))
    if distance_to_target <= distance_target_hit:
      break

  return positions_list, edges_list


if __name__ == '__main__':
  # Parameters

  xy_0 : tuple = (100, 100)
  xy_target : tuple = (666, 420)

  world_width = 1000
  world_height = 1000 

  distance_target_hit = 50

  # Run RRT without any obstacles
  searched_positions, edges = obstacle_free_RRT(xy_0, xy_target, distance_target_hit, (world_width, world_height))

  # Plot the positions
  x_positions : list = [pos[0] for pos in searched_positions]
  y_positions : list = [pos[1] for pos in searched_positions]

  plt.scatter(x_positions, y_positions, marker="o", color="k")
  plt.scatter([xy_target[0]], [xy_target[1]], marker="o", color="green")
  plt.scatter([xy_0[0]], [xy_0[1]], marker="o", color="red")

  # Plot the edges
  for edge in edges:
    start_pos : tuple = edge[0]
    end_pos : tuple = edge[1]
    plt.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 'k-')

  # Plotting the last edge to the target
  plt.plot([x_positions[-1], xy_target[0]], [y_positions[-1], xy_target[1]], 'g-')
 
  plt.show()

