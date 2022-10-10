from __future__ import print_function
from enum import Enum
import cv2
import numpy as np
import math

class WallClass(Enum):
    kUNKNOWN    = 0
    kISOLATED   = 1
    kNEAR_OBJ   = 2
    kON_OBJ     = 3


# Pseudocode for API
class WallDetector:
    def __init__(self, debug=False):
        ## Private Vars
        self._debug             = debug
        self._plate_radius_cm   = 8.5   # radius of plate in cm. 8.5=blue, 9.0=white
        self._near_wall_cm      = 1     # distace to detect if near wall
        self._food_set          = {}
        self._food_class_list   = {}
        self._other_food_thresh = 6.0 #cm. distance threshold to see if the food is near another food item
        self._food_history      = 50

        ## Table Plane Params
        self._hough_accum   = 1.5
        self._hough_min_dist= 100
        self._hough_param1  = 100
        self._hough_param2  = 70 # Larger is more selective
        self._hough_min     = 100
        self._hough_max     = 150
        self._table_buffer  = 50 # Extra radius around plate to use for table

        ## Segmentation Params
        self._seg_color_px  = 25    # Number of pixels around u, v to use for average color
        self._color_range   = 100   # range in rgb values to use around centerpoint for segmentation by color
        self._sub_width     = 100   # width of subregion in pixels to use
        self._sub_height    = 100   # height of subregion in pixels to use

        ## Subregion Params
        self._num_regions = 15 # number of regions to discretize region around fruit

        ## Decision Params
        self._exclude_region_cm = 0.2 # distance to detect if subregion is clsoe to wall and shouldn't be counted
        self._percent_for_case_1 = 0.15 #percent of subregions with less than threshold
        self._percent_for_case_3 = 0.5 #percent of subregions with more than threshold
        self._percent_for_near_food = 0.1 #percent of subregions ... when food is near other food
        self._percent_for_case_3_near_wall = 0.3 #percent of subregions with more than threshold when near wall
        self._region_threshold = 17 #mm. use to compare region height to table height, should be plate_height + buffer
        self._extention_dist = 1.5 #cm. distance from food edge for subregions

    def _px_close_to_plate(self, y, x, plate_uv, plate_r, dist_cm = None):
        """
            Detects if pixel is close to plate edge
            x,y : x and y point in pixel space of point
            plate([x,y,r]): circle in pixel space of plate
            to_rim_thresh(float): threshold to determine if a given point is close to rim
        """
        if dist_cm is None:
            dist_cm = self._near_wall_cm
        dist_ratio = plate_r / self._plate_radius_cm
        threshold = dist_ratio * dist_cm
        length = math.sqrt((y - plate_uv[1])**2 + (x - plate_uv[0])**2)
        if length < (plate_r - threshold):
            return False
        return True

    def _near_rim(self, mask, plate_uv, plate_r):
        """
            Detect if food is near the plate wall
            mask(image matrix): food = 1, otherwise 0
            plate_uv, plate_r: plate circle with u,v centerpoint and radius r
        """
        near_rim = False
        pixels = np.where(mask > 0)
        for i in range(len(pixels[0])):
            if self._px_close_to_plate(pixels[0][i], pixels[1][i], plate_uv, plate_r):
                near_rim = True
                break

        if self._debug:
            print("Near Rim? " + str(near_rim))

        return near_rim

    def _near_other_food(self, point, r):
        """
            Detects if food center is within a threshold distance of another food item
            point(x,y): position of center of food item
            r (float): radius of plate in pixels
        """

        # cm -> pixels. compostable plate
        near_other_food = False
        dist_ratio = float(r)/self._plate_radius_cm
        for _, bite in self._food_set.iteritems():
            if point[0] != bite[0] or point[1] != bite[1]:
                if ((float(point[0])-float(bite[0]))**2 + (float(point[1])-float(bite[1]))**2 )**(0.5) < dist_ratio * self._other_food_thresh:
                    near_other_food = True
                    break

        if self._debug:
            print("Near Food? " + str(near_other_food))

        return near_other_food

    def _fit_table(self, gray, depth):
        """
            Find table plane.
            gray(image matrix): grayscale image of plate
            depth(image matrix): depth image of plate

            Returns:
            plate_uv: (u, v) of plate center
            plate_r: radius of plate in px
            height: new image matrix, height of pixel above table
        """

        # Detect Largest Circle (Plate)
        # raida-- get plate locaton (center and radius) using houghcircles which finds circles in a grayscale image using the Hough transform
        # from open cv package
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, self._hough_accum, self._hough_min_dist,
            param1=self._hough_param1, param2=self._hough_param2, minRadius=self._hough_min, maxRadius=self._hough_max)
        if circles is None:
            return None, None, None
        circles = np.round(circles[0, :]).astype("int")
        plate_uv = (0, 0)
        plate_r = 0
        for (x,y,r) in circles:
            print("Radius: " + str(r))
            if r > plate_r:
                plate_uv = (x, y)
                plate_r = r

        # Create Mask for Depth Image
        plate_mask = np.zeros(depth.shape)
        cv2.circle(plate_mask, plate_uv, plate_r + self._table_buffer, 1.0, -1)
        cv2.circle(plate_mask, plate_uv, plate_r, 0.0, -1)
        depth_mask = (depth * (plate_mask).astype("uint16")).astype(float)

        # Noise removal
        kernel = np.ones((6,6), np.uint8)
        depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_OPEN, kernel, iterations = 3)
        depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_CLOSE, kernel, iterations = 3)

        # Remove Outliers
        depth_var = np.abs(depth_mask - np.mean(depth_mask[depth_mask > 0]))
        depth_std = np.std(depth_mask[depth_mask > 0])
        depth_mask[depth_var > 2.0*depth_std] = 0

        # Fit Plane: depth = a*u + b*v + c
        d_idx = np.where(depth_mask > 0)
        d = depth_mask[d_idx].astype(float)
        coeffs = np.hstack((np.vstack(d_idx).T, np.ones((len(d_idx[0]), 1)))).astype(float)
        b, a, c = np.linalg.lstsq(coeffs, d)[0]

        # Create Table Depth Image
        u = np.linspace(0, depth.shape[1], depth.shape[1], False)
        v = np.linspace(0, depth.shape[0], depth.shape[0], False)
        U, V = np.meshgrid(u, v)
        table = a*U + b*V + c
        table = table.astype("uint16")

        # New Height
        height = table - np.clip(depth, None, table)
        height[np.where(depth == 0)] = 0

        # Noise removal
        kernel = np.ones((2,2), np.uint8)
        height = cv2.morphologyEx(height, cv2.MORPH_OPEN, kernel, iterations = 2)
        height = cv2.morphologyEx(height, cv2.MORPH_CLOSE, kernel, iterations = 2)

        # Debugging
        if self._debug:
            gray_mask = (gray * (plate_mask).astype("uint8")).astype("uint8")
            cv2.imshow("gray", gray)
            cv2.imshow("gray_mask", gray_mask)
            cv2.imshow("depth", depth * 255)
            cv2.imshow("depth_mask", 255 * depth_mask.astype("uint16"))
            cv2.imshow("table", 255 * table)
            cv2.imshow("height_above_table", 300 * height)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # raida-- here to get the diameter, we can compute and return plate_r * 2 = plate_d
        # if we add plate_d as a parameter to return from this function, we can then perhaps save it in  a variable
        # with that saved variable, say d, we can say while(current_plate_len != d), keep the arm move the plate 
        # to bring the full plate into the camera view

        # perhaps we need a new code file like ada_visual_servoing.py that can call wall_detector._fit_table function and
        # have other information like  arm movement and plate holding commands relative to the table

        # question:how to connect the new file to a runnable file on the robot, so all these hypothetical 
        # codes can be evaluated if works or not?

        # comment: I think this file' code(when/if the above mentioned code is written) fulfils the task of detecting plate length 
        # in camera view suitable from partial to entirety. So, answers my Q1 and Q2 from the milestone #1 doc. The remaining has
        # to do with moving the arm holding the plate, and creatng a file for these functionalities without harming this original file

        #question: is my intuition going in the right direction?
        return plate_uv, plate_r, height

    def _segment(self, img, uv):
        """
            Segment food item into a mask and masked depth
            img(image matrix): color image of plate
            uv([x,y]): centerpoint of food item in pixel space

            Returns:
            mask: 1 = food; 0 = not food
            bbox: food bounding box
        """

        # Color region for segmentation
        diameter = 25
        color_begin = img[int(uv[1]-diameter/2):int(uv[1]+diameter/2), int(uv[0]-diameter/2):int(uv[0]+diameter/2)]
        b,g,r =  cv2.split(color_begin)
        b = np.mean(b)
        g = np.mean(g)
        r = np.mean(r)
        color_begin = (b,g,r)
        color_end = []

        # Prevent going over 255
        for i in range(0,len(color_begin)):
            if color_begin[i] + self._color_range/2 > 255:
                color_end.append(255)
            else:
                color_end.append(color_begin[i] + int(self._color_range/2))
        color_end = np.array(color_end,np.uint8)
        color_begin = np.array(color_begin,np.uint8)

        # Prevent going under 255
        for i in range(0,len(color_begin)):
            if color_begin[i] - self._color_range/2 < 0:
                color_begin[i] = 0
            else:
                color_begin[i] = color_begin[i] - int(self._color_range/2)

        # Get subregion of image
        rect = (uv[0]-self._sub_width/2,
            uv[1]-self._sub_width/2,
            self._sub_width,
            self._sub_height)
        rect2 = [int(uv[0]-self._sub_width/2),
            int(uv[0]+self._sub_width/2),
            int(uv[1]-self._sub_height/2),
            int(uv[1]+self._sub_height/2)]

        for point in range(0,len(rect2)):
            if rect2[point] < 0:
                rect2[point] = 0
        rgb = img[rect2[2]:rect2[3],rect2[0]:rect2[1]]
        gray = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)

        # Threshold subregion
        thresh = cv2.inRange(rgb,color_begin,color_end)

        # Noise removal
        kernel = np.ones((5,5), np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations = 2)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations = 2)

        # Isolate just food item over other segmented items
        ret, markers = cv2.connectedComponents(opening)
        center = markers[int(self._sub_height/2), int(self._sub_width/2)]
        for y in range(0,len(markers)):
            for x in range(0,len(markers[0])):
                if markers[y,x] != center:
                    markers[y,x] = 1
                else:
                    markers[y,x] = 0
        markers = np.array(markers, np.uint8)


        # Make fullsize mask
        mask_inv = np.ones((480,640), np.uint16) # mask has food = 0, otherwise 1
        mask_inv[rect2[2]:rect2[3],rect2[0]:rect2[1]] = markers
        mask = (1 - mask_inv).astype(np.uint8)

        # Apply mask
        bbox = rect2

        if self._debug:
            imgCopy = img.copy()
            imgCopy = cv2.bitwise_and(imgCopy, imgCopy, mask=mask)
            cv2.rectangle(imgCopy, (bbox[0], bbox[2]), (bbox[1], bbox[3]), (0, 0, 255), 3)
            cv2.imshow("segmentation", imgCopy)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return mask, bbox

    def _subregions(self, mask, height, plate_r):
        """
            Make subregions around food item
            mask(image matrix): food = 1, otherwise 0
            height(image matrix): height image
            plate_r: radius of plate in pixels

            Returns:
            regions: Each region is an array of points around food
            regions_median: median height of each region
        """
        try:
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        except ValueError:
            _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(contours[0])

        # calculate x,y coordinate of center fruit
        M = cv2.moments(hull)
        if M["m00"] == 0:
            # No proper moments
            return None, None
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        extended_region = []
        dist_ratio = float(plate_r) / self._plate_radius_cm
        dist = self._extention_dist * dist_ratio
        for h_point in hull:
            internal_dist = math.sqrt((cX-h_point[0][0])**2 + (cY - h_point[0][1])**2)
            direction = h_point[0][0] > cX
            if h_point[0][0] - cX != 0:
                slope = float(h_point[0][1]-cY)/float(h_point[0][0] - cX)
            else:
                slope = 0
            xn = int((dist+internal_dist)/math.sqrt((slope**2)+1))
            if not direction:
                xn = xn * -1
            yn = int(xn*slope)+ cY
            xn += cX
            extended_region.append([xn,yn])

        ext_mask = np.zeros((480,640),np.uint8)
        for point in range(0,len(extended_region)):
            if point != len(extended_region)-1:
                cv2.line(ext_mask,tuple(extended_region[point]),tuple(extended_region[point+1]),1,2)
            else:
                cv2.line(ext_mask,tuple(extended_region[point]),tuple(extended_region[0]),1,2)

        # Mask of region around fruit
        try:
            contours,hierarchy = cv2.findContours(ext_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        except ValueError:
            _, contours,hierarchy = cv2.findContours(ext_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        hull_ext = cv2.convexHull(contours[0])
        M_ext = cv2.moments(hull_ext)
        eX = int(M_ext["m10"] / M_ext["m00"])
        eY = int(M_ext["m01"] / M_ext["m00"])
        for y in range(0,len(ext_mask)):
            for x in range(0,len(ext_mask[0])):
                if cv2.pointPolygonTest(contours[0],(x,y),False)>0:
                    ext_mask[y,x] = 1
                if mask[y,x] == 1:
                    ext_mask[y,x] = 0

        # Discretize region around fruit
        # TODO: (Gilwoo) this errors with division_by_zero sometimes.
        # Since the current mode Can we make wall detector safe
        # (not crash) when it's actually not above plate? (We run spanet all the time)
        angle = int(360/self._num_regions)
        regions = []
        for x in range(self._num_regions):
            regions.append([])
        for y in range(0,len(ext_mask)):
            for x in range(0,len(ext_mask[0])):
                if ext_mask[y,x] == 1:
                    p_angle = math.degrees(math.atan2(float(y)-eY,(float(x)-eX)))
                    if p_angle < 0:
                        p_angle = p_angle +360
                    reg = int(p_angle/angle)
                    if reg >= len(regions):
                        reg = 0
                    regions[reg].append((x,y))

        height_regions = []
        region_median = []
        for region in regions:
            h_reg = []
            for reg in region:
                h_reg.append(height[reg[1],reg[0]])
            height_regions.append(h_reg)
            region_median.append(np.median(h_reg))
        return regions, region_median

    def _decide(self, near_rim, near_other_food, regions, regions_median, plate_uv, plate_r):
        """
            Decide what position the food is in
            near_rim(bool): whether fruit is near wall or not
            near_other_food(bool): whether fruit is near other food or not
            regions(list of lists): lists of height values for each region around food
            regions_median(list of floats): list of median height values per region
            plate_uv([x,y]): center of plate in pixel space
            plate_r: radius of plate in pixels
        """
        food_class = WallClass.kUNKNOWN

        # Wall Case (2 or 3)
        if near_rim:
            # Exclude Regions Near Plate Wall
            excluded_regions = []
            for region in range(0,len(regions)):
                for point in regions[region]:
                    if self._px_close_to_plate(point[1], point[0], plate_uv, plate_r, self._exclude_region_cm):
                        excluded_regions.append(region)
                        break

            # Majority Vote of Regions
            num_with_stuff = 0
            considered_regions = self._num_regions - len(excluded_regions)
            if considered_regions > 0: # To prevent an error, but this should never happen if cirlces are good
                for region in range(0,len(regions)):
                    if region not in excluded_regions:
                        if self._debug:
                            print("Region Number: " + str(region))
                            print("Region Height: " + str(regions_median[region]))
                        if regions_median[region] > self._region_threshold:
                            num_with_stuff += 1
                if self._debug:
                    print("Percent of regions above threshold: " + str(float(num_with_stuff)/float(considered_regions)))
                if float(num_with_stuff)/float(considered_regions) >= self._percent_for_case_3_near_wall:
                    food_class = WallClass.kON_OBJ
                else:
                    food_class = WallClass.kNEAR_OBJ
            else:
                print("WARNING: No region included")
                food_class = WallClass.kNEAR_OBJ

        # Not near a wall
        else:
            case_1_percent = self._percent_for_case_1
            if near_other_food:
                 case_1_percent = self._percent_for_near_food

            # Majority Vote of Regions
            num_with_stuff = 0
            for region in range(0,len(regions)):
                if self._debug:
                            print("Region Number: " + str(region))
                            print("Region Height: " + str(regions_median[region]))
                if regions_median[region] > self._region_threshold:
                    num_with_stuff+=1
            if self._debug:
                    print("Percent of regions above threshold: " + str(float(num_with_stuff)/float(self._num_regions)))
            if float(num_with_stuff)/float(self._num_regions) <= case_1_percent:
                food_class = WallClass.kISOLATED
            else:
                # differentiate from 2 / 3
                if float(num_with_stuff)/float(self._num_regions) >= self._percent_for_case_3:
                    food_class = WallClass.kON_OBJ
                else:
                    food_class = WallClass.kNEAR_OBJ

        # if self._debug:
        print("FOOD CLASS " + str(food_class))

        return food_class

    def classify(self, box, img, depth):
        uv = box['uv']
        item_id = box['id']
        # Create Grayscale Image
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Fit Table Plane
        plate_uv, plate_r, height = self._fit_table(img_gray, depth)
        if plate_uv is None:
            if self._debug:
                print("Warning, no Hough Circle!")
            return WallClass.kUNKNOWN

        # See if near other food item
        near_food_item =  self._near_other_food(uv, plate_r)

        # Segment Food Item
        mask, bbox = self._segment(img, uv)

        # See if near plate rim
        near_rim = self._near_rim(mask, plate_uv, plate_r)

        # Create Sub-regions
        region, region_median = self._subregions(mask, height, plate_r)
        if region is None:
            if self._debug:
                print("Warning, no proper subregions!")
            return WallClass.kUNKNOWN

        food_class = self._decide(near_rim, near_food_item, region, region_median, plate_uv, plate_r)

        # Add to list
        self._food_class_list[item_id].append(food_class)
        if len(self._food_class_list[item_id]) > self._food_history:
            self._food_class_list[item_id].pop(0)

        # Return mode of list
        return max(set(self._food_class_list[item_id]), key=self._food_class_list[item_id].count)

    def register_items(self, boxes):
        for item in boxes:
            self._food_set[item['id']] = item['uv']
            if item['id'] not in self._food_class_list:
                self._food_class_list[item['id']] = []
        if self._debug:
            print("Registered UV Point: " + str(item['uv']))


