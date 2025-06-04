import numpy as np
from math import sqrt, atan2, pi, inf

class SafeDistanceInter:
    def __init__(self, neighbor_region=5, neighbor_num=10, vxmax=1.5, vymax=1.5, acceler=0.5, env_train=True, safe_radius=0.2):
        self.nr = neighbor_region
        self.nm = neighbor_num
        self.vxmax = vxmax
        self.vymax = vymax
        self.acceler = acceler
        self.env_train = env_train
        self.safe_radius = safe_radius

    def config_safe_distance_inf(self, robot_state, nei_state_list, obs_cir_list, obs_line_list, action=np.zeros((2,)), **kwargs):
        # Preprocess to get components in region
        robot_state, ns_list, oc_list, ol_list = self.preprocess(robot_state, nei_state_list, obs_cir_list, obs_line_list)
        
        action = np.squeeze(action)
        
        # Calculate safe distance information for each neighbor
        safe_info_list = []
        collision_flag = False
        min_dis = inf
        
        # Process moving neighbors
        for nei in ns_list:
            safe_info = self.calculate_safe_info(robot_state, nei, action, is_moving=True)
            safe_info_list.append(safe_info)
            if safe_info['collision']:
                collision_flag = True
            if safe_info['distance'] < min_dis:
                min_dis = safe_info['distance']
                
        # Process static obstacles
        for obs in oc_list:
            safe_info = self.calculate_safe_info(robot_state, obs, action, is_moving=False)
            safe_info_list.append(safe_info)
            if safe_info['collision']:
                collision_flag = True
            if safe_info['distance'] < min_dis:
                min_dis = safe_info['distance']
                
        # Process line obstacles
        for line in ol_list:
            safe_info = self.calculate_line_safe_info(robot_state, line, action)
            safe_info_list.append(safe_info)
            if safe_info['collision']:
                collision_flag = True
            if safe_info['distance'] < min_dis:
                min_dis = safe_info['distance']
        
        # Sort and limit number of observations
        safe_info_list.sort(key=lambda x: x['distance'])
        if len(safe_info_list) > self.nm:
            safe_info_list = safe_info_list[:self.nm]
            
        # Convert to observation format
        obs_list = []
        for info in safe_info_list:
            obs = [
                info['rel_vel_x'],
                info['rel_vel_y'],
                info['cos_angle'],
                info['sin_angle'],
                info['distance'],
                1.0 / (info['time_to_collision'] + 0.2)  # Similar to RVO's input_exp_time
            ]
            obs_list.append(obs)
            
        return obs_list, collision_flag, min_dis

    def calculate_safe_info(self, robot_state, other_state, action, is_moving=True):
        x, y, vx, vy, r = robot_state[0:5]
        mx, my, mvx, mvy, mr = other_state[0:5]
        
        # Calculate relative position and velocity
        rel_x = x - mx
        rel_y = y - my
        
        if is_moving:
            rel_vx = 2*action[0] - mvx - vx
            rel_vy = 2*action[1] - mvy - vy
        else:
            rel_vx = action[0]
            rel_vy = action[1]
            
        # Calculate distance and angle
        distance = sqrt(rel_x**2 + rel_y**2)
        angle = atan2(rel_y, rel_x)
        
        # Calculate time to collision
        time_to_collision = self.calculate_time_to_collision(rel_x, rel_y, rel_vx, rel_vy, r + mr)
        
        # Check for collision
        collision = distance <= (r + mr)
        
        return {
            'rel_vel_x': rel_vx,
            'rel_vel_y': rel_vy,
            'cos_angle': np.cos(angle),
            'sin_angle': np.sin(angle),
            'distance': distance - mr,  # Distance to obstacle surface
            'time_to_collision': time_to_collision,
            'collision': collision
        }

    def calculate_line_safe_info(self, robot_state, line, action):
        x, y, vx, vy, r = robot_state[0:5]
        
        # Calculate distance to line segment
        point = np.array([x, y])
        sp = np.array(line[0])
        ep = np.array(line[1])
        
        l2 = (ep - sp) @ (ep - sp)
        if l2 == 0.0:
            distance = np.linalg.norm(point - sp)
            projection = sp
        else:
            t = max(0, min(1, ((point-sp) @ (ep-sp)) / l2))
            projection = sp + t * (ep-sp)
            distance = np.linalg.norm(projection - point)
            
        # Calculate angle to line
        rel_vector = projection - point
        angle = atan2(rel_vector[1], rel_vector[0])
        
        # Calculate time to collision
        time_to_collision = self.calculate_time_to_collision(
            point[0] - projection[0],
            point[1] - projection[1],
            action[0],
            action[1],
            r
        )
        
        # Check for collision
        collision = distance <= r
        
        return {
            'rel_vel_x': action[0],
            'rel_vel_y': action[1],
            'cos_angle': np.cos(angle),
            'sin_angle': np.sin(angle),
            'distance': distance,
            'time_to_collision': time_to_collision,
            'collision': collision
        }

    def calculate_time_to_collision(self, rel_x, rel_y, rel_vx, rel_vy, min_dist):
        a = rel_vx**2 + rel_vy**2
        b = 2*rel_x*rel_vx + 2*rel_y*rel_vy
        c = rel_x**2 + rel_y**2 - min_dist**2
        
        if c <= 0:
            return 0
            
        temp = b**2 - 4*a*c
        if temp <= 0:
            return inf
            
        t1 = (-b + sqrt(temp)) / (2*a)
        t2 = (-b - sqrt(temp)) / (2*a)
        
        t3 = t1 if t1 >= 0 else inf
        t4 = t2 if t2 >= 0 else inf
        
        return min(t3, t4)

    def preprocess(self, robot_state, nei_state_list, obs_cir_list, obs_line_list):
        robot_state = np.squeeze(robot_state)
        ns_list = list(filter(lambda x: 0 < self.distance(robot_state, x) <= self.nr, nei_state_list))
        oc_list = list(filter(lambda y: 0 < self.distance(robot_state, y) <= self.nr, obs_cir_list))
        ol_list = list(map(lambda z: self.segment_in_circle(robot_state[0], robot_state[1], self.nr, z), obs_line_list))
        ol_list = [x for x in ol_list if x is not None]
        
        return robot_state, ns_list, oc_list, ol_list

    @staticmethod
    def distance(point1, point2):
        return sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

    @staticmethod
    def segment_in_circle(x, y, r, line):
        start_point = np.array(line[0:2])
        d = np.array([line[2] - line[0], line[3] - line[1]])
        f = np.array([line[0] - x, line[1] - y])
        
        a = d@d
        b = 2*f@d
        c = f@f - r**2
        
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            return None
            
        t1 = (-b - sqrt(discriminant)) / (2*a)
        t2 = (-b + sqrt(discriminant)) / (2*a)
        
        if t1 >= 0 and t1 <= 1 and t2 >= 0 and t2 <= 1:
            segment_point1 = start_point + t1 * d
            segment_point2 = start_point + t2 * d
        elif t1 >= 0 and t1 <= 1 and t2 > 1:
            segment_point1 = start_point + t1 * d
            segment_point2 = np.array(line[2:4])
        elif t1 < 0 and t2 >= 0 and t2 <= 1:
            segment_point1 = np.array(line[0:2])
            segment_point2 = start_point + t2 * d
        elif t1 < 0 and t2 > 1:
            segment_point1 = np.array(line[0:2])
            segment_point2 = np.array(line[2:4])
        else:
            return None
            
        return [segment_point1, segment_point2]