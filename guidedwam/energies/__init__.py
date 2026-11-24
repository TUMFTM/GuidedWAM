"""Differentiable energies on the metric ego-frame trajectory [B, T, 3] -> [B].

navhard: camera freespace, lane centre, waypoint drivable area, box clearance.
navhard_fp: footprint / corner drivable area (oracle and learned field), off-route.
kinematic: speed cap, target speed, comfort.
"""
