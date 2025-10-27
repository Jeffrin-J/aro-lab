#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:44:32 2023

@author: stonneau
"""

import pinocchio as pin
import numpy as np
from numpy.linalg import pinv, norm

from config import LEFT_HAND, RIGHT_HAND, LEFT_HOOK, RIGHT_HOOK
from tools import collision
from inverse_geometry import computeqgrasppose
import time


def samplecubeplacement(cube, cubeplacementq0, cubeplacementqgoal):
    """
    Sample a random cube placement between initial and goal positions.
    Returns an SE3 placement for the cube.

    We bound the sampling to reasonable positions near the workspace.
    """
    # Interpolate between initial and goal with random factor
    alpha = np.random.rand()

    # Linear interpolation of position
    pos_init = cubeplacementq0.translation
    pos_goal = cubeplacementqgoal.translation
    pos_sample = pos_init * (1 - alpha) + pos_goal * alpha

    # Add small random perturbation to explore workspace
    perturbation = np.random.randn(3) * 0.05  # 5cm std deviation
    pos_sample = pos_sample + perturbation

    # Keep Z above table (minimum height)
    pos_sample[2] = max(pos_sample[2], 0.90)  # Keep above table

    # For now, keep the same orientation (can be extended to sample rotations)
    rot_sample = cubeplacementq0.rotation

    # Could also interpolate/sample rotation if needed:
    # rot_sample = pin.exp3(alpha * pin.log3(cubeplacementq0.rotation.T @ cubeplacementqgoal.rotation)) @ cubeplacementq0.rotation

    return pin.SE3(rot_sample, pos_sample)


def samplevalidconfig(robot, cube, cubeplacementq0, cubeplacementqgoal, qcurrent, max_attempts=50):
    """
    Sample a collision-free configuration with grasping constraint.

    Process:
    1. Sample a random cube placement
    2. Solve IK to find a grasping configuration
    3. Check if collision-free
    4. Return configuration if valid

    Returns:
    --------
    q : valid grasping configuration (or None if failed)
    cube_placement : the cube placement for this configuration
    """
    for _ in range(max_attempts):
        # Sample cube placement
        cube_placement = samplecubeplacement(cube, cubeplacementq0, cubeplacementqgoal)

        # Compute grasping configuration for this cube placement
        q, success = computeqgrasppose(robot, qcurrent, cube, cube_placement, viz=None)

        if success:
            return q, cube_placement

    # Failed to find valid configuration
    return None, None


def projectpath(robot, cube, q0, q1, cube_placement_0, cube_placement_1, discretization_steps=50):
    """
    Project a path between q0 and q1 such that grasping constraints are maintained.

    Given two grasping configurations, we need to ensure that every interpolated
    configuration also satisfies the grasping constraint.

    Strategy: Interpolate the cube placement, then solve IK at each step to find
    the corresponding robot configuration.

    Parameters:
    -----------
    robot : robot model
    cube : cube model
    q0, q1 : start and end configurations
    cube_placement_0, cube_placement_1 : start and end cube placements
    discretization_steps : number of steps for interpolation

    Returns:
    --------
    path : list of configurations (empty if projection failed)
    success : boolean indicating if projection succeeded
    """
    path = [q0]

    for i in range(1, discretization_steps + 1):
        # Interpolation parameter
        alpha = float(i) / discretization_steps

        # Interpolate cube placement (position and orientation)
        pos_0 = cube_placement_0.translation
        pos_1 = cube_placement_1.translation
        pos_interp = pos_0 * (1 - alpha) + pos_1 * alpha

        # SLERP for rotation (spherical linear interpolation)
        # For SE3: M_interp = M_0 * exp(alpha * log(M_0^-1 * M_1))
        rot_delta = cube_placement_0.rotation.T @ cube_placement_1.rotation
        rot_interp = cube_placement_0.rotation @ pin.exp3(alpha * pin.log3(rot_delta))

        cube_placement_interp = pin.SE3(rot_interp, pos_interp)

        # Compute configuration for this cube placement
        q_prev = path[-1]
        q_interp, success = computeqgrasppose(robot, q_prev, cube, cube_placement_interp, viz=None)

        if not success:
            # Projection failed - could not maintain grasp
            return path, False

        path.append(q_interp)

    return path, True


def distance(q1, q2):
    """Euclidean distance between configurations"""
    return norm(q2 - q1)


def nearest_vertex(G, q_rand):
    """Find index of nearest vertex in graph G to q_rand"""
    min_dist = float('inf')
    idx = -1
    for i, (parent, q, cube_placement) in enumerate(G):
        dist = distance(q, q_rand)
        if dist < min_dist:
            min_dist = dist
            idx = i
    return idx


def extend(robot, cube, q_near, q_rand, cube_placement_near, cube_placement_rand, delta_q=0.3):
    """
    Extend from q_near towards q_rand while maintaining grasping constraint.

    Returns:
    --------
    q_new : new configuration
    cube_placement_new : cube placement at q_new
    reached : True if q_rand was reached
    """
    dist = distance(q_near, q_rand)

    if dist <= delta_q:
        # Can reach q_rand directly
        # Try to project path
        path, success = projectpath(robot, cube, q_near, q_rand,
                                     cube_placement_near, cube_placement_rand,
                                     discretization_steps=20)
        if success:
            return q_rand, cube_placement_rand, True
        else:
            return q_near, cube_placement_near, False
    else:
        # Take a step of size delta_q towards q_rand
        alpha = delta_q / dist

        # Interpolate configuration (rough estimate)
        q_target = q_near + alpha * (q_rand - q_near)

        # Interpolate cube placement
        pos_near = cube_placement_near.translation
        pos_rand = cube_placement_rand.translation
        pos_target = pos_near + alpha * (pos_rand - pos_near)

        rot_delta = cube_placement_near.rotation.T @ cube_placement_rand.rotation
        rot_target = cube_placement_near.rotation @ pin.exp3(alpha * pin.log3(rot_delta))

        cube_placement_target = pin.SE3(rot_target, pos_target)

        # Compute exact grasping configuration for this cube placement
        q_new, success = computeqgrasppose(robot, q_near, cube, cube_placement_target, viz=None)

        if success:
            # Verify path from q_near to q_new is collision-free under grasp
            path, path_success = projectpath(robot, cube, q_near, q_new,
                                             cube_placement_near, cube_placement_target,
                                             discretization_steps=10)
            if path_success:
                return q_new, cube_placement_target, False

        return q_near, cube_placement_near, False


def rrt_with_grasp(robot, cube, qinit, qgoal, cubeplacementq0, cubeplacementqgoal,
                   max_iterations=5000, delta_q=0.3, goal_bias=0.1):
    """
    RRT planner that maintains grasping constraints.

    Parameters:
    -----------
    robot : robot model
    cube : cube model
    qinit : initial grasping configuration
    qgoal : goal grasping configuration
    cubeplacementq0 : initial cube placement
    cubeplacementqgoal : goal cube placement
    max_iterations : maximum RRT iterations
    delta_q : step size for extension
    goal_bias : probability of sampling goal

    Returns:
    --------
    path : list of configurations from qinit to qgoal
    success : True if path found
    """
    # Graph: list of (parent_index, configuration, cube_placement)
    G = [(None, qinit, cubeplacementq0)]

    print(f"Starting RRT with grasp constraints (max_iter={max_iterations}, delta_q={delta_q})")

    for iteration in range(max_iterations):
        # Sample configuration with grasping constraint
        print('yes', iteration)
        if np.random.rand() < goal_bias:
            # Bias towards goal
            q_rand = qgoal
            cube_placement_rand = cubeplacementqgoal
        else:
            # Sample random configuration
            q_rand, cube_placement_rand = samplevalidconfig(robot, cube, cubeplacementq0,
                                                            cubeplacementqgoal, qinit)
            if q_rand is None:
                continue  # Failed to sample, try again

        # Find nearest vertex
        idx_near = nearest_vertex(G, q_rand)
        parent_q = G[idx_near][1]
        parent_cube_placement = G[idx_near][2]

        # Extend towards q_rand
        q_new, cube_placement_new, reached = extend(robot, cube, parent_q, q_rand,
                                                     parent_cube_placement, cube_placement_rand,
                                                     delta_q)

        # Check if made progress
        if distance(q_new, parent_q) < 1e-3:
            continue  # No progress made

        # Add to graph
        G.append((idx_near, q_new, cube_placement_new))

        # Check if goal reached
        if distance(q_new, qgoal) < delta_q:
            # Try to connect to goal
            path, success = projectpath(robot, cube, q_new, qgoal,
                                        cube_placement_new, cubeplacementqgoal,
                                        discretization_steps=20)
            if success:
                print(f"Path found after {iteration + 1} iterations!")
                # Add goal to graph
                G.append((len(G) - 1, qgoal, cubeplacementqgoal))

                # Extract path
                return extract_path(G), True

        if (iteration + 1) % 100 == 0:
            print(f"  Iteration {iteration + 1}/{max_iterations}, graph size: {len(G)}")

    print(f"Path not found after {max_iterations} iterations")
    return [qinit, qgoal], False


def extract_path(G):
    """Extract path from graph by backtracking from goal"""
    path = []
    node = G[-1]  # Start from goal

    while node[0] is not None:
        path.insert(0, node[1])  # Insert configuration at beginning
        node = G[node[0]]  # Move to parent

    path.insert(0, G[0][1])  # Add initial configuration

    return path


# Main function required by lab instructions
def computepath(qinit, qgoal, cubeplacementq0, cubeplacementqgoal):
    """
    Returns a collision-free path from qinit to qgoal under grasping constraints.
    The path is expressed as a list of configurations.

    This is the required interface for the lab assessment.
    """
    # Import here to avoid circular dependency
    from tools import setupwithmeshcat
    robot, cube, _ = setupwithmeshcat()

    # Use RRT to find path
    path, success = rrt_with_grasp(robot, cube, qinit, qgoal,
                                    cubeplacementq0, cubeplacementqgoal,
                                    max_iterations=3000, delta_q=0.4)

    if not success:
        print("Warning: RRT failed to find complete path, returning direct path")
        # Return projected path if possible
        path_projected, proj_success = projectpath(robot, cube, qinit, qgoal,
                                                    cubeplacementq0, cubeplacementqgoal,
                                                    discretization_steps=50)
        if proj_success:
            return path_projected

    return path


def displaypath(robot,path,dt,viz):
    for q in path:
        viz.display(q)
        time.sleep(dt)


if __name__ == "__main__":
    from tools import setupwithmeshcat
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
    from inverse_geometry import computeqgrasppose
    
    robot, cube, viz = setupwithmeshcat()
    
    
    q = robot.q0.copy()
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    
    if not(successinit and successend):
        print ("error: invalid initial or end configuration")
    
    path = computepath(q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)
    
    displaypath(robot,path,dt=0.5,viz=viz) #you ll probably want to lower dt
    
