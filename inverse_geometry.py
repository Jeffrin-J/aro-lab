#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import pinocchio as pin 
import numpy as np
from numpy.linalg import pinv,inv,norm,svd,eig
from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, EPSILON
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET

from tools import setcubeplacement

def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    '''
    Return a collision free configuration grasping a cube at a specific location.

    The function computes a robot configuration where:
    - LEFT_HAND is aligned with LEFT_HOOK
    - RIGHT_HAND is aligned with RIGHT_HOOK
    - The configuration is collision-free
    - Joint limits are respected

    Parameters:
    -----------
    robot : robot model
    qcurrent : current/initial configuration to start from
    cube : cube model
    cubetarget : target SE3 placement for the cube
    viz : optional visualizer

    Returns:
    --------
    q : configuration grasping the cube (collision-free)
    success : boolean indicating if a valid grasp pose was found
    '''
    # Set cube to target placement
    setcubeplacement(robot, cube, cubetarget)

    # Update cube geometry to get hook positions
    pin.framesForwardKinematics(cube.model, cube.data, cube.q0)
    pin.updateGeometryPlacements(cube.model, cube.data,
                                  cube.collision_model, cube.collision_data, cube.q0)

    # Get hook frame IDs
    left_hook_id = cube.model.getFrameId(LEFT_HOOK)
    right_hook_id = cube.model.getFrameId(RIGHT_HOOK)

    # Hook poses in world frame
    oMlhook = cubetarget * cube.data.oMf[left_hook_id]
    oMrhook = cubetarget * cube.data.oMf[right_hook_id]

    # Get end-effector frame IDs
    left_hand_id = robot.model.getFrameId(LEFT_HAND)
    right_hand_id = robot.model.getFrameId(RIGHT_HAND)

    # Initialize from current configuration
    q = qcurrent.copy()

    # IK parameters
    max_iterations = 1000
    epsilon = 1e-4
    step_size = 0.5

    # Inverse kinematics using iterative Jacobian-based approach
    for i in range(max_iterations):
        # Update robot kinematics
        pin.framesForwardKinematics(robot.model, robot.data, q)

        # Current end-effector poses
        oMlhand = robot.data.oMf[left_hand_id]
        oMrhand = robot.data.oMf[right_hand_id]

        # Compute errors (in local frame for numerical stability)
        left_error = pin.log(oMlhand.inverse() * oMlhook).vector
        right_error = pin.log(oMrhand.inverse() * oMrhook).vector

        # Check convergence
        if norm(left_error) < epsilon and norm(right_error) < epsilon:
            # Check collision
            if not collision(robot, q):
                if viz:
                    viz.display(q)
                return q, True
            else:
                # Converged but in collision, need to escape
                break

        # Compute Jacobians
        pin.computeJointJacobians(robot.model, robot.data, q)
        J_left = pin.getFrameJacobian(robot.model, robot.data, left_hand_id, pin.ReferenceFrame.LOCAL)
        J_right = pin.getFrameJacobian(robot.model, robot.data, right_hand_id, pin.ReferenceFrame.LOCAL)

        # Stack errors and Jacobians (dual-arm task)
        error = np.concatenate([left_error, right_error])
        J = np.vstack([J_left, J_right])

        # Compute damped pseudoinverse
        damping = 1e-6
        J_pinv = J.T @ inv(J @ J.T + damping * np.eye(J.shape[0]))

        # Compute update with postural bias (to keep natural-looking poses)
        dq_task = J_pinv @ error

        # Add postural task: minimize deviation from current posture
        # This creates a "postural bias" toward natural configurations
        postural_weight = 0.1
        P = np.eye(len(q)) - J_pinv @ J  # Null space projector
        dq_posture = postural_weight * (qcurrent - q)

        dq = dq_task + P @ dq_posture

        # Update configuration
        q = q + step_size * dq

        # Project to joint limits
        q = projecttojointlimits(robot, q)

    # If IK converged but in collision, try to escape collision
    # Use gradient-based collision avoidance
    max_escape_iter = 200
    for i in range(max_escape_iter):
        if not collision(robot, q):
            if viz:
                viz.display(q)
            return q, True

        # Compute numerical gradient of collision
        gradient = np.zeros(len(q))
        delta = 0.01

        for j in range(len(q)):
            q_plus = q.copy()
            q_minus = q.copy()
            q_plus[j] += delta
            q_minus[j] -= delta

            # Simple collision score
            score_plus = 1.0 if collision(robot, q_plus) else 0.0
            score_minus = 1.0 if collision(robot, q_minus) else 0.0

            gradient[j] = (score_plus - score_minus) / (2 * delta)

        # Move away from collision
        if norm(gradient) > 1e-6:
            q = q - 0.05 * gradient / norm(gradient)
        else:
            # Add random perturbation if stuck
            q = q + np.random.randn(len(q)) * 0.02

        # Project to joint limits
        q = projecttojointlimits(robot, q)

    # Failed to find collision-free grasp
    if viz:
        viz.display(q)
    return q, False
            
if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals
    robot, cube, viz = setupwithmeshcat()
    
    q = robot.q0.copy()
    
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    
    updatevisuals(viz, robot, cube, q0)
    
    
    