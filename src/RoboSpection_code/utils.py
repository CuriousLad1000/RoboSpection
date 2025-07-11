import rospy
import PyKDL
import math
import time
import random
import tf2_ros
import numpy as np
import open3d as o3d
import subprocess
import sys
import glob
import moveit_msgs
from scipy.spatial.transform import Rotation as R
from scipy.spatial import distance
from scipy import spatial
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped, Pose, PoseStamped
from realsense2_camera.msg import Extrinsics
from cv_bridge import CvBridge
from ast import literal_eval

import copy
from copy import deepcopy
from PIL import Image as PIL_img
from io import BytesIO
import threading


class CameraProcessor:
    """
    A class for providing visual inspection pipeline.
    It does processing and analyzing of 3D point clouds using Open3D and ROS.
    Provides functions for Poin cloud generation from depth and color frames, filtering, clustering, transformation,
    coordinate generation, and publishing frames.
    """
    
    def __init__(self, samples=1, offset_y=0.13, offset_z=0.0, trim_base=0.05, manual_offset=0.0, 
                 cluster_discard=0, spacing=0.01, eps=0.05, min_points=10, cluster_trim=0.01, tgt_coord_samples=3, 
                 tgt_final_trim=0.0, tgt_reverse=True, tgt_preview=True, z_offset=0.3, coord_skip=0, tgt_motion_delay=0.1, 
                 tgt_save=True, dbug=False, robo=True):
        """
        Initializes the CameraProcessor with a bridge for image conversions and a TF buffer for transformations.
        
        :param robo: default True - initializes ROS controls, transforms, broadcasts, buffers, CvBridge, rospy
        :param bridge: cv_bridge for converting ROS images to OpenCV format.
        :param tfbuffer: TF2 buffer for handling transformations.
        """
        self.samples = samples
        self.offset_y = offset_y
        self.offset_z = offset_z
        self.trim_base = trim_base
        self.manual_offset = manual_offset
        self.cluster_discard = cluster_discard
        self.spacing = spacing
        self.eps = eps
        self.min_points = min_points
        self.cluster_trim = cluster_trim
        self.tgt_coord_samples = tgt_coord_samples
        self.tgt_final_trim = tgt_final_trim
        self.tgt_reverse = tgt_reverse
        self.tgt_preview = tgt_preview
        self.z_offset = z_offset
        self.coord_skip = coord_skip
        self.tgt_motion_delay = tgt_motion_delay
        self.tgt_save = tgt_save
        self.dbug = dbug
        
        if robo==True:
            self.bridge = CvBridge() #bridge
            self.tfbuffer = tf2_ros.Buffer() #tfbuffer
            self.br = tf2_ros.TransformBroadcaster()
            self.static_br = tf2_ros.StaticTransformBroadcaster()
            self.listener = tf2_ros.TransformListener(self.tfbuffer)
            self.loop_rate = rospy.Rate(0.5) # Node cycle rate (in Hz).
        
        self.mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1,origin=[0, 0, 0])
        
        self.front_def = [ 0.66268635354679917, -0.64917448455669569, -0.37338891979194477 ]
        self.lookat_def = [ 0.16328584993371112, 0.10901604638723184, 0.39735867391549773 ]
        self.up_def = [ -0.27349211301491638, 0.25436583492870624, -0.92763143874043985 ]
        self.zoom_def = 1.02

    def grab_frame(self):
        """
        Captures color and depth frames from the camera.

        :return: A tuple containing (color_frame, depth_frame) as OpenCV images.
        """
        frame_color = rospy.wait_for_message('/camera/color/image_raw', Image, timeout=None)
        cv_image_color = self.bridge.imgmsg_to_cv2(frame_color, desired_encoding='rgb8')

        frame_depth = rospy.wait_for_message('/camera/depth/image_raw', Image, timeout=None)
        cv_image_depth = self.bridge.imgmsg_to_cv2(frame_depth)

        return cv_image_color, cv_image_depth

    def get_cam_param(self, depth_to_color=False):
        """
        Retrieves camera intrinsic and extrinsic parameters.

        :param depth_to_color: Boolean flag indicating whether depth-to-color extrinsics should be returned.
        :return: Tuple (width, height, fx, fy, cx, cy, translation, rotation).
        """
        frame_depth_info = rospy.wait_for_message('/camera/depth/camera_info', CameraInfo, timeout=None)
        w, h = frame_depth_info.width, frame_depth_info.height
        fx, fy, cx, cy = frame_depth_info.K[0], frame_depth_info.K[4], frame_depth_info.K[2], frame_depth_info.K[5]

        if depth_to_color:
            frame_depth_extrinsic = rospy.wait_for_message('/camera/extrinsics/depth_to_color', Extrinsics, timeout=None)
            return w, h, fx, fy, cx, cy, frame_depth_extrinsic.translation, frame_depth_extrinsic.rotation
        else:
            rot = [1, 0, 0, 0, 1, 0, 0, 0, 1]
            trans = [0, 0, 0]
            return w, h, fx, fy, cx, cy, trans, rot

    def fetch_transform(self, frame1, frame2, quat=0):
        """
        Fetches the transformation between two frames.

        :param frame1: Source frame name.
        :param frame2: Target frame name.
        :param quat: If 1, returns quaternion instead of roll-pitch-yaw angles.
        :return: Translation (x, y, z) and either rotation as (rpy) or (qx, qy, qz, qw).
        """
        while True:
            try:
                trans = self.tfbuffer.lookup_transform(frame1, frame2, rospy.Time(), rospy.Duration(8.0))
                trans = trans.transform

                rot = R.from_quat([trans.rotation.x, trans.rotation.y, trans.rotation.z, trans.rotation.w])
                rpy = rot.as_euler('XYZ', degrees=True)
                break
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                continue
        if quat == 0:
            return trans.translation.x, trans.translation.y, trans.translation.z, rpy[0], rpy[1], rpy[2]
        else:
            return trans.translation.x, trans.translation.y, trans.translation.z, \
                   trans.rotation.x, trans.rotation.y, trans.rotation.z, trans.rotation.w

    def generate_point_cloud(self, color_frame, depth_frame, trim_base, from_depth=False, depth_scale=1000.0, depth_trunc=1.0, align=False, Dbug=False, zoom_def=0.72):
        """
        Generates a point cloud from color and depth frames.

        :param color_frame: Color frame from the camera.
        :param depth_frame: Depth frame from the camera.
        :param trim_base: Trim value for filtering the ground.
        :param from_depth: Boolean flag to generate point cloud using depth-only.
        :param depth_scale: Scaling factor for depth values.
        :param depth_trunc: Maximum depth threshold.
        :param align: Align depth to color space.
        :param Dbug: Debug mode for visualization.
        :param zoom_def: Default zoom value.
        :return: Open3D point cloud object.
        """
        w, h, fx, fy, cx, cy, trans, rot = self.get_cam_param(depth_to_color=align)

        cam = o3d.camera.PinholeCameraParameters()
        cam.intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
        cam.extrinsic = np.array([
            [rot[0], rot[1], rot[2], trans[0]],
            [rot[3], rot[4], rot[5], trans[1]],
            [rot[6], rot[7], rot[8], trans[2]],
            [0., 0., 0., 1.]
        ])

        color_raw = o3d.geometry.Image(np.asarray(color_frame))
        depth_raw = o3d.geometry.Image(np.asarray(depth_frame.astype(np.uint16)))

        if not from_depth:
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, convert_rgb_to_intensity=False)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, cam.intrinsic, cam.extrinsic)
        
            #CROP TO FILTER OUT ROBOT'S SHADOW ADJUST OFFSET ACCORDINGLY
            PC_BBOX = pcd.get_axis_aligned_bounding_box()
            minB_X = PC_BBOX.min_bound[0]
            maxB_X = PC_BBOX.max_bound[0]
            minB_Y = PC_BBOX.min_bound[1]
            maxB_Y = PC_BBOX.max_bound[1]
            minB_Z = PC_BBOX.min_bound[2]
            maxB_Z = PC_BBOX.max_bound[2]      
        
           #RAW Point cloud from depth
            if Dbug==True:
                mesh = self.mesh
                o3d.visualization.draw_geometries([pcd,mesh], window_name='01_RAW Pointcloud: from Simulated Camera', point_show_normal=True, zoom=zoom_def, front=self.front_def, lookat=self.lookat_def, up=self.up_def)
        
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(minB_X, minB_Y, minB_Z), 
                                                   max_bound=(maxB_X, maxB_Y, maxB_Z-trim_base)) #filter ground part 
            pcd = pcd.crop(bbox)
           
        else:
            pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_raw, cam.intrinsic, cam.extrinsic,
                                                                  depth_scale, depth_trunc,
                                                                  stride=1, project_valid_depth_only=True)
        
            #RAW Point cloud from depth
            if Dbug==True:
                mesh = self.mesh
                o3d.visualization.draw_geometries([pcd,mesh], window_name='01_RAW Pointcloud: from Real Camera', point_show_normal=True, zoom=zoom_def, front=self.front_def, lookat=self.lookat_def, up=self.up_def)
        
        return pcd

    def get_average_point_cloud(self, offset_y, offset_z, manual_offset, trim_base, Dbug, eval_tag=False):
        """
        Computes an averaged point cloud from multiple frames.

        :param offset_y: Offset in Y-axis.
        :param offset_z: Offset in Z-axis.
        :param manual_offset: Manual depth threshold adjustment.
        :param trim_base: Trim value for ground filtering.
        :param Dbug: Debug mode for visualization.
        :param eval_tag: Boolean flag to transform to world coordinates.
        :return: Filtered Open3D point cloud.
        """
        zoom_def = 1.34
        cam_robo_depth = abs(self.fetch_transform('camera_depth_optical_frame', 'panda_link0', quat=0)[2])
        color_frame, depth_frame = self.grab_frame()

        downpcd = self.generate_point_cloud(color_frame, depth_frame, trim_base, from_depth=False,
                                            depth_trunc=manual_offset + cam_robo_depth - trim_base, align=False, Dbug=Dbug)

        #CROP TO FILTER OUT ROBOT'S SHADOW ADJUST OFFSET ACCORDINGLY
        PC_BBOX = downpcd.get_axis_aligned_bounding_box()
        minB_X = PC_BBOX.min_bound[0]
        maxB_X = PC_BBOX.max_bound[0]
        minB_Y = PC_BBOX.min_bound[1]
        maxB_Y = PC_BBOX.max_bound[1]
        minB_Z = PC_BBOX.min_bound[2]
        maxB_Z = PC_BBOX.max_bound[2]

        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(minB_X+0.01, minB_Y+0.01, minB_Z), max_bound=(maxB_X-0.01, maxB_Y-offset_y, maxB_Z-offset_z))
        downpcd = downpcd.crop(bbox)

        if eval_tag == True:    
            #Fetch world to camera frame
            world_TO_cam_depth_optical = self.fetch_transform('world', 'camera_depth_optical_frame',quat=1)

            # Create a KDL frame out of it for world to camera
            matx = PyKDL.Frame(PyKDL.Rotation.Quaternion(world_TO_cam_depth_optical[3], world_TO_cam_depth_optical[4], world_TO_cam_depth_optical[5], 
                                                                        world_TO_cam_depth_optical[6]),
                                              PyKDL.Vector(world_TO_cam_depth_optical[0], world_TO_cam_depth_optical[1], world_TO_cam_depth_optical[2]))

            #Create 4x4 numpy transformation matrix
            KDL_World_cam_frame = np.array([[matx[0,0], matx[0,1], matx[0,2], matx[0,3]],
                                            [matx[1,0], matx[1,1], matx[1,2], matx[1,3]],
                                            [matx[2,0], matx[2,1], matx[2,2], matx[2,3]],
                                            [0.       , 0.       , 0.       , 1.      ]])

            downpcd.transform(KDL_World_cam_frame)

            #CROP TO FILTER OUT ROBOT'S SHADOW ADJUST OFFSET ACCORDINGLY
            PC_BBOX = downpcd.get_axis_aligned_bounding_box()
            minB_X = PC_BBOX.min_bound[0]
            maxB_X = PC_BBOX.max_bound[0]
            minB_Y = PC_BBOX.min_bound[1]
            maxB_Y = PC_BBOX.max_bound[1]
            minB_Z = PC_BBOX.min_bound[2]
            maxB_Z = PC_BBOX.max_bound[2]

            #bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(minB_X+0.01, minB_Y+0.01, minB_Z+0.001), max_bound=(maxB_X-0.01, maxB_Y-offset_y, maxB_Z)) 
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(minB_X, minB_Y, minB_Z+0.005), max_bound=(maxB_X, maxB_Y, maxB_Z)) #Filter ground
            downpcd = downpcd.crop(bbox)

        if Dbug==True:
            mesh = self.mesh
            o3d.visualization.draw_geometries([downpcd,mesh], window_name="02_Filtered Pointcloud: Remove Ground and Robot", point_show_normal=True, zoom=zoom_def, front=self.front_def, lookat=self.lookat_def, up=self.up_def)

        return downpcd



    

    def load_point_cloud(self, samples, offset_y, offset_z, manual_offset, spacing, trim_base, Hide_prev=False, Dbug=False, eval_tag=False, HPR=False):
        """
        Loads and processes multiple point clouds to generate a cleaned dataset.

        :param samples: Number of point cloud frames to average.
        :param offset_y: Offset in Y-axis.
        :param offset_z: Offset in Z-axis.
        :param manual_offset: Manual depth adjustment.
        :param spacing: Spacing for downsampling.
        :param trim_base: Ground filtering trim value.
        :param Hide_prev: Whether to hide previous visualizations. default: False
        :param Dbug: Debug mode for visualization. default: False
        :param eval_tag: Boolean flag to transform to world coordinates. Also affects how normals are generated. default: False
        :param HPR: Enable or disable Hidden Point Removal default:False
        :return: Filtered and processed point cloud.
        """
        if (Hide_prev==False or Dbug==True):
            mesh = self.mesh

        average_ptcloud = o3d.geometry.PointCloud()
        pts_cloud_tot = []
        pts_tot = []

        for _ in range(samples):
            new_pt = self.get_average_point_cloud(offset_y, offset_z, manual_offset, trim_base, Dbug, eval_tag)
            pts_cloud_tot.append(new_pt)
            pts_tot.append(len(np.asarray(new_pt.points)))

            
        #Collected Point clouds
        if Dbug==True:
            snip = pts_cloud_tot
            snip.append(mesh)
            o3d.visualization.draw_geometries(snip, window_name="03_Sampled Pointclouds", point_show_normal=True, zoom=self.zoom_def, front=self.front_def, lookat=self.lookat_def, up=self.up_def)

        ## We'll select only those point clouds that have a majority of similar "number of points". 
        ## this will prevent any outlier point clouds that have weird data in it (like less points 
        ## or more points than the group)
        bins = np.linspace(np.min(pts_tot), np.max(pts_tot), int(np.ceil(samples / 3)))  #generate bins in range of number of pts in ptcloud
        digitized = np.digitize(pts_tot, bins) #put values in bins based on size of point clouds
        uni, count = np.unique(digitized,return_counts=True) #get the uniques and their counts
        unique_val = uni[np.argmax(count)]  #get the value from unique list where count was maximum
        idx_list = np.where(digitized == unique_val)[0]  #get all indexes from digitized list where it matches unique

        for idx in idx_list:
            average_ptcloud += pts_cloud_tot[idx]

        #Average pointcloud having majority of similar point clouds
        if Dbug==True:
            o3d.visualization.draw_geometries([average_ptcloud,mesh], window_name="04_Filtered Pointcloud: Select valid pointclouds by majority vote", point_show_normal=True, zoom=self.zoom_def, front=self.front_def, lookat=self.lookat_def, up=self.up_def)

        
        if HPR:
            ## Filter out hidden points using Hidden point Removal
            diameter = np.linalg.norm(np.asarray(average_ptcloud.get_max_bound()) - np.asarray(average_ptcloud.get_min_bound()))
            cam = [0, 0, -diameter]  #[0, 0, diameter]
            radius = diameter * 100
            _, pt_map = average_ptcloud.hidden_point_removal(cam, radius) #Get all points that are visible from given view point
            average_ptcloud = average_ptcloud.select_by_index(pt_map)
            
            if Dbug==True:
                o3d.visualization.draw_geometries([average_ptcloud,mesh], window_name="05_Filtered Pointcloud: Hidden point removal", point_show_normal=True, zoom=self.zoom_def, front=self.front_def, lookat=self.lookat_def, up=self.up_def)

                
        #Downsample point cloud
        average_ptcloud = average_ptcloud.voxel_down_sample(voxel_size=spacing)
        if Dbug==True:
            o3d.visualization.draw_geometries([average_ptcloud,mesh], window_name="06_Filtered Pointcloud: Downsample the pointcloud", point_show_normal=True, zoom=self.zoom_def, front=self.front_def, lookat=self.lookat_def, up=self.up_def)

            
        #Estimate Normals
        average_ptcloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.06, max_nn=30)) #radius in meters
        if eval_tag:
            average_ptcloud.orient_normals_towards_camera_location(camera_location=[0, 0,1000])
        else:
            #average_ptcloud.orient_normals_towards_camera_location(camera_location=[0, 0, 0])
            average_ptcloud.orient_normals_towards_camera_location(camera_location=[0, 0,-1000])

        filtered_Pt_Cloud = average_ptcloud
        print("Filtered Point cloud: ", filtered_Pt_Cloud)

        if (Hide_prev==False or Dbug==True):
            o3d.visualization.draw_geometries([filtered_Pt_Cloud,mesh], window_name="07_Filtered Pointcloud: Estimate surface normals", point_show_normal=True, zoom=self.zoom_def, front=self.front_def, lookat=self.lookat_def, up=self.up_def)

        if Dbug==True:
            nor = np.array(filtered_Pt_Cloud.normals)
            pts = np.array(filtered_Pt_Cloud.points)

            #find coordinates that are closest to the center 
            distance,index = spatial.KDTree(pts).query( filtered_Pt_Cloud.get_center() )  

            print(distance)
            print(index)
            print()
            print("Center Coordinates(ground truth):",filtered_Pt_Cloud.get_center())
            print("Points Coordinates(estimated center point):",pts[index])
            print("Normal Coordinates(Normal of estimated center point):",nor[index])

        return(filtered_Pt_Cloud)

    def cluster_point_cloud(self, original_PC, eps=0.02, min_points=10,obstacle=False):
        """
        Segments a point cloud into clusters using the DBSCAN algorithm.

        :param original_PC: Open3D point cloud object.
        :param eps: Maximum distance between points to be considered a cluster.
        :param min_points: Minimum points required to form a cluster.
        :return: List of clustered point clouds.
        """
        labels = np.array(original_PC.cluster_dbscan(eps=eps, min_points=min_points))

        uniques = np.unique(labels)
        clouds = []  # Use a list instead of np.array to store clusters

        if len(uniques) == 1 and uniques[0] == -1:
            clouds.append(original_PC)  # No valid clusters, return the original point cloud
        else:
            for label in uniques:
                if label != -1:  # Ignore noise points
                    idx = np.where(labels == label)[0]
                    cluster_pcd = original_PC.select_by_index(idx)
                    clouds.append(cluster_pcd)

        print(f"Total Number of clusters: {len(clouds)}")      
        #Filter Cluster based on settings
        if self.cluster_discard > 0 and obstacle==False:
            cld_idx_remove = []
            for cld_idx in range (len(clouds)):
                if len(np.array(clouds[cld_idx].points)) <= self.cluster_discard:
                    cld_idx_remove.append(cld_idx)
            clouds = list(np.delete(clouds, cld_idx_remove, axis=0))
            print(f"Number of clusters after filtering clusters with <= {self.cluster_discard} points: {len(clouds)}")
        return sorted(clouds, key=lambda pc: len(pc.points), reverse=True)   #clouds

    def bounds_gen(self, minB, maxB, spacing):
        """
        Generates evenly spaced bounding boxes for segmenting the point cloud.

        :param minB: Minimum boundary.
        :param maxB: Maximum boundary.
        :param spacing: Distance between each boundary box.
        :return: Numpy array of bounding box ranges.
        """
        ctr = 0
        bounds = []
        CurrB = minB

        while CurrB < maxB - 0.5 * spacing: #from left most to right most
            LowerB = minB + ctr * spacing + 0.005 #if we shift the X or Y coordinates in multiples of spacing, we should get different lines.
                                                  #+ some extra width to reduce outlier points in PC
            if ctr == 0:
                LowerB = minB + ctr * spacing
                CurrB = LowerB + spacing / 2 #lower + spacing/2 only for first condition
            else:
                CurrB = LowerB + spacing  #lower + spacing 
            bounds.append([LowerB, CurrB])
            #print("CurrB:",CurrB,"MaxB:",maxB,"diff:",maxB-CurrB)
            ctr += 1

        return np.array(bounds)

    def get_XY_angles_from_PC(self, tst_downpcd_nor):
        """
        Computes angles of point cloud normals relative to the X and Y axes.

        :param point_cloud_normals: List of normal vectors from the point cloud.
        :return: Tuple containing (X angles, Y angles).
        """
        Xs, Ys = [], []
        vec_x = [1., 0., 0.]
        vec_y = [0., 1., 0.]

        for normal_vec in tst_downpcd_nor:
            if np.all(normal_vec == [0., 0., 1.]):
                normal_vec = [0, 0.000001, 0.999999]
            vec1 = normal_vec / np.linalg.norm(normal_vec)
            angle_x = np.round(np.degrees(math.acos(np.dot(vec1, vec_x))))
            angle_y = np.round(np.degrees(math.acos(np.dot(vec1, vec_y))))
            Xs.append(angle_x)
            Ys.append(angle_y)

        return np.array(Xs), np.array(Ys)

    def profile_calc(self, original_PC, spacing):
        """
        Computes available X and Y profiles from a given point cloud.

        :param original_PC: Open3D point cloud object.
        :param spacing: Distance between profiles.
        :return: Tuple containing (X profile bounds, Y profile bounds).
        """
        PC_BBOX = original_PC.get_axis_aligned_bounding_box()
        minB_X, maxB_X = PC_BBOX.min_bound[0], PC_BBOX.max_bound[0]
        minB_Y, maxB_Y = PC_BBOX.min_bound[1], PC_BBOX.max_bound[1]

        profiles_X = self.bounds_gen(minB_Y, maxB_Y, spacing)
        profiles_Y = self.bounds_gen(minB_X, maxB_X, spacing)
        #print("Total number of X Profiles available:",len(profiles_X))
        #print("Total number of Y Profiles available:",len(profiles_Y))
        return profiles_X, profiles_Y

    def profile_gen(self, original_PC, spacing, Cluster_trim, Selected_profiles=0, resample=True, preview=False, auto=False):
        """
        Generates object profiles from a point cloud.

        :param original_PC: Open3D point cloud object.
        :param spacing: Distance between profiles.
        :param Cluster_trim: Trim threshold for filtering.
        :param Selected_profiles: List of predefined profiles (default: 0).
        :param resample: Whether to resample points (default: True).
        :param preview: Whether to preview profiles before filtering (default: False).
        :param auto: Whether to automatically select profiles (default: False).
        :return: List of selected profiles as Open3D point clouds.
        """
        selected_idx, selected_pcd_X, selected_pcd_Y, Profiles_PCD = [], [], [], []

        PC_BBOX = original_PC.get_axis_aligned_bounding_box()
        minB_X, maxB_X = PC_BBOX.min_bound[0], PC_BBOX.max_bound[0]
        minB_Y, maxB_Y = PC_BBOX.min_bound[1], PC_BBOX.max_bound[1]
        minB_Z, maxB_Z = PC_BBOX.min_bound[2], PC_BBOX.max_bound[2]

        bounds_X, bounds_Y = self.profile_calc(original_PC, spacing)  #Calculate total profiles available in the Point cloud

        if auto:
            nor = np.array(original_PC.normals)
            Xs, Ys = self.get_XY_angles_from_PC(nor)
            X_dev, Y_dev = np.std(Xs), np.std(Ys)
            #print("X Standard Dev:", X_dev, "    Y Standard Dev:", Y_dev)

            if X_dev <= Y_dev: #Single Center profile in direction of Y axis 
                #Selected_profiles = [[[1, len(bounds_Y) // 2]], [4], [5, len(bounds_Y) // 2]]
                Selected_profiles = [ [[1, int(np.floor(len(bounds_Y)/2))]], [4], [5, int(np.floor(len(bounds_Y)/2))] ]
                print("Object's curve around Cam X-axis (Y in World)")
            else:
                #Selected_profiles = [[[0, len(bounds_X) // 2]], [4, len(bounds_X) // 2], [5]]
                Selected_profiles = [ [[0, int(np.floor(len(bounds_X)/2))]], [4, int(np.floor(len(bounds_X)/2))], [5] ]
                print("Object's curve around Cam Y-axis (X in World)")
        #print("Selected Profiles:",Selected_profiles)
        data, X_dat, Y_dat = Selected_profiles[0], Selected_profiles[1][1:], Selected_profiles[2][1:] #Profile Data, #X unique profiles (mode 4), #Y unique profiles (mode 5) 

        if not preview:   ### Generate Object Profiles
            if len(Y_dat) > 0:
                for idx in Y_dat:
                    bbox = o3d.geometry.AxisAlignedBoundingBox(
                        min_bound=(bounds_Y[idx][0], minB_Y + Cluster_trim, minB_Z),
                        max_bound=(bounds_Y[idx][1], maxB_Y - Cluster_trim, maxB_Z) 
                    )  #trim edges

                    current_profile = bbox.get_point_indices_within_bounding_box(original_PC.points)
                    while len(current_profile) == 0:  #Reject all Point cloud profile bounds with 0 points... instead take the next one...
                        idx += 1
                        bbox = o3d.geometry.AxisAlignedBoundingBox(
                            min_bound=(bounds_Y[idx][0], minB_Y + Cluster_trim, minB_Z),
                            max_bound=(bounds_Y[idx][1], maxB_Y - Cluster_trim, maxB_Z)
                        ) #trim edges
                        current_profile = bbox.get_point_indices_within_bounding_box(original_PC.points)

                    result = original_PC.select_by_index(current_profile, invert=False) # select inside points = cropped
                    res_pts = np.array(result.points)
                    res_nor = np.array(result.normals)     

                    if resample:
                        min_idx = np.argmin(np.std(res_pts, axis=0))
                        res_pts[:, min_idx] = np.mean(res_pts, axis=0)[min_idx]  #replace values of axis (x,y or z)
                                                                            #that has min std_dev by its mean.
                        result.points = o3d.utility.Vector3dVector(res_pts)
                        result = result.voxel_down_sample(voxel_size=spacing)

                    res_pts = np.array(result.points)
                    res_nor = np.array(result.normals)
                        
                    ind = np.argsort(res_pts[:, 1]) #Sort Y coordinates from lowest to highest, X values are almost constant
                    res_pts = res_pts[ind]  #no longer required to change values to negative 
                    res_nor = res_nor[ind]

                    sorted_pointcloud = o3d.geometry.PointCloud()
                    sorted_pointcloud.points = o3d.utility.Vector3dVector(res_pts)
                    sorted_pointcloud.normals = o3d.utility.Vector3dVector(res_nor)

                    selected_pcd_Y.append(sorted_pointcloud)  #Store Unique Object profiles generated in Y direction... 


            if len(X_dat) > 0:
                for idx in X_dat:
                    bbox = o3d.geometry.AxisAlignedBoundingBox(
                        min_bound=(minB_X + Cluster_trim, bounds_X[idx][0], minB_Z),
                        max_bound=(maxB_X - Cluster_trim, bounds_X[idx][1], maxB_Z)
                    )

                    current_profile = bbox.get_point_indices_within_bounding_box(original_PC.points)
                    while len(current_profile) == 0:  #Reject all Point cloud profile bounds with 0 points... instead take the next one...
                        idx += 1
                        bbox = o3d.geometry.AxisAlignedBoundingBox(
                            min_bound=(minB_X + Cluster_trim, bounds_X[idx][0], minB_Z),
                            max_bound=(maxB_X - Cluster_trim, bounds_X[idx][1], maxB_Z)
                        )
                        current_profile = bbox.get_point_indices_within_bounding_box(original_PC.points)

                    result = original_PC.select_by_index(current_profile, invert=False) # select inside points = cropped
                    res_pts = np.array(result.points)
                    res_nor = np.array(result.normals)
                    
                    if resample:
                        min_idx = np.argmin(np.std(res_pts, axis=0))
                        res_pts[:, min_idx] = np.mean(res_pts, axis=0)[min_idx]  #replace values of axis (x,y or z)
                                                                            #that has min std_dev by its mean.
                        result.points = o3d.utility.Vector3dVector(res_pts)
                        result = result.voxel_down_sample(voxel_size=spacing)

                    res_pts = np.array(result.points)
                    res_nor = np.array(result.normals)

                    ind = np.argsort(res_pts[:, 0]) #Sort X coordinates from lowest to highest, Y values are almost constant
                    res_pts = res_pts[ind]
                    #res_pts[:,2] *=-1  #Change z values to negative
                    res_nor = res_nor[ind]

                    sorted_pointcloud = o3d.geometry.PointCloud()
                    sorted_pointcloud.points = o3d.utility.Vector3dVector(res_pts)
                    sorted_pointcloud.normals = o3d.utility.Vector3dVector(res_nor)

                    selected_pcd_X.append(sorted_pointcloud)  #Store Unique Object profiles generated in X direction... 


            for profiles in data:
                if profiles[0] % 2 == 0:  #Check for X or Y type of profiles 
                    indices_arr = np.nonzero(np.in1d(X_dat, profiles[1:]))[0]  #Search all elements in X_dat and get indices
                                                                             #elements in profiles Must always be sorted!!
                    for idxxx in indices_arr:
                        Profiles_PCD.append(selected_pcd_X[idxxx])

                elif profiles[0] % 2 == 1:  #Check for X or Y type of profiles
                    indices_arr = np.nonzero(np.in1d(Y_dat, profiles[1:]))[0]  #Search all elements in Y_dat and get indices
                                                                             #elements in profiles Must always be sorted!!
                    for idxxx in indices_arr:
                        Profiles_PCD.append(selected_pcd_Y[idxxx])

            return Profiles_PCD

        elif preview == True:  ####################### Running in Preview Mode....
            if len(Y_dat) > 0:
                #bounds = Bounds_gen(minB_X, maxB_X, spacing)
                #print("Total number of Y Profiles available:",len(bounds))

                for idx in Y_dat:
                    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(bounds_Y[idx][0], minB_Y+Cluster_trim, minB_Z), 
                                                               max_bound=(bounds_Y[idx][1], maxB_Y-Cluster_trim, maxB_Z)) #trim edges
                    #selected_idx.append(bbox.get_point_indices_within_bounding_box(original_PC.points))

                    current_profile = bbox.get_point_indices_within_bounding_box(original_PC.points)
                    while len(current_profile) == 0:  #Reject all Point cloud profile bounds with 0 points... instead take the next one...
                        idx+=1 #use next available bound instead...
                        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(bounds_Y[idx][0], minB_Y+Cluster_trim, minB_Z), 
                                                                   max_bound=(bounds_Y[idx][1], maxB_Y-Cluster_trim, maxB_Z)) #trim edges
                        current_profile = bbox.get_point_indices_within_bounding_box(original_PC.points)
                        #print(len(current_profile))
                    selected_idx.append(current_profile)

            if len(X_dat) > 0:
                #bounds = Bounds_gen(minB_Y, maxB_Y, spacing)
                #print("Total number of X Profiles available:",len(bounds))
                for idx in X_dat:
                    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(minB_X+Cluster_trim, bounds_X[idx][0], minB_Z), 
                                                               max_bound=(maxB_X-Cluster_trim, bounds_X[idx][1], maxB_Z))

                    #selected_idx.append(bbox.get_point_indices_within_bounding_box(original_PC.points))
                    current_profile = bbox.get_point_indices_within_bounding_box(original_PC.points)
                    while len(current_profile) == 0:  #Reject all Point cloud profile bounds with 0 points... instead take the next one...
                        idx+=1 #use next available bound instead...
                        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(minB_X+Cluster_trim, bounds_X[idx][0], minB_Z), 
                                                                   max_bound=(maxB_X-Cluster_trim, bounds_X[idx][1], maxB_Z)) #trim edges

                        current_profile = bbox.get_point_indices_within_bounding_box(original_PC.points)
                        #print(len(current_profile))
                    selected_idx.append(current_profile)

            if len(X_dat) > 0 or len(Y_dat) > 0:
                selected_idx = np.array(selected_idx,dtype='object')
                selected_idx = np.hstack(selected_idx).astype('int')  #Flatten all lists into a single one.

                selected_pcd = original_PC.select_by_index(selected_idx, invert=False) # select inside points = cropped 
                selected_pcd.paint_uniform_color([0, 1, 0])
                rejected_pcd = original_PC.select_by_index(selected_idx, invert=True) #select outside points
            else:
                return -1,-1

            return selected_pcd, rejected_pcd

    def getRotation(self, v1):
        """
        Computes the rotation matrix and angles from a normal vector.

        :param v1: A 3D normal vector (list or numpy array).
        :return: Tuple containing (Rotation matrix, angle_x, angle_y, angle_z).
        """
        if np.all(v1 == [0., 0., 1.]): v1 = [0, 0.000001, 0.999999]
        vec_x = [1., 0., 0.]
        vec_y = [0., 1., 0.]
        vec_z = [0., 0., 1.] #>>>>>>>>>>>>>>>>>>>>> change these to -1 to flip again

        vec1 = v1 / np.linalg.norm(v1)
        #vector_x = np.cross(vec1, vec_x)/np.linalg.norm(np.cross(vec1, vec_x))
        angle_x = math.acos(np.dot(vec1, vec_x))
        #vector_y = np.cross(vec1, vec_y)/np.linalg.norm(np.cross(vec1, vec_y))
        angle_y = math.acos(np.dot(vec1, vec_y))
        vector_z = np.cross(vec1, vec_z) / np.linalg.norm(np.cross(vec1, vec_z))
        angle_z = -(math.acos(np.dot(vec1, vec_z)))
        #Rotation = filtered_Pt_Cloud.get_rotation_matrix_from_axis_angle(angle*vector) #alternative Open3D lib.
        Rotation = R.from_rotvec(angle_z * vector_z)
        return Rotation, angle_x, angle_y, angle_z
    
    def generate_coordinates(self, point_cloud):
        """
        Generates 3D point coordinates in both camera and world frames.

        :param point_cloud: Open3D point cloud object with normals.
        :return: Tuple containing (Camera frame coordinates, World frame coordinates).
        """
        #Stores coordinates of Points in Point Cloud wrt the camera_depth_optical_frame
        Camera_points_Coordinates = np.array([[0., 0., 0., 0., 0., 0., 0.]])
        #Stores coordinates of Points in Point Cloud wrt the world frame
        world_points_Coordinates = np.array([[0., 0., 0., 0., 0., 0., 0.]])
        point_cloud_pts = np.asarray(point_cloud.points)
        point_cloud_nor = np.asarray(point_cloud.normals)
        
        #Fetch world to camera frame
        world_TO_cam_depth_optical = self.fetch_transform('world', 'camera_depth_optical_frame', quat=1)
        
        # Create a KDL frame out of it for world to camera
        KDL_World_cam_frame = PyKDL.Frame(
            PyKDL.Rotation.Quaternion(*world_TO_cam_depth_optical[3:]),
            PyKDL.Vector(*world_TO_cam_depth_optical[:3])
        )
        
        #Run through each point in the point cloud
        for index in range(len(point_cloud_pts)):
            #Create a Rotation matrix from normal of the point in point cloud
            rotat, angle_x, angle_y, angle_z = self.getRotation(-point_cloud_nor[index])
            rotat = rotat.as_matrix()

            #Create a KDL frame for the original plane where the point is lying
            KDL_original_plane_frame = PyKDL.Frame(
                PyKDL.Rotation(*rotat.flatten().tolist()),
                PyKDL.Vector(*point_cloud_pts[index])
            )
            
            #Create a KDL frame to mirror plane frame to match camera frame's orientation
            KDL_flip_frame = PyKDL.Frame(PyKDL.Rotation.RPY(0, np.pi, 0), PyKDL.Vector(0, 0, 0))
            #Calculate KDL frame between the camera_depth_optical_frame and point lying on the plane.
            KDL_Camera_plane_frame = KDL_original_plane_frame * KDL_flip_frame
            #Extract Camera coordinates of the points lying on the plane
            KDL_trans = KDL_Camera_plane_frame.p
            KDL_ROT_quat = KDL_Camera_plane_frame.M.GetQuaternion()
            #Store Coordinates of points in Camera frame.
            Camera_points_Coordinates = np.append(
                Camera_points_Coordinates, [[*KDL_trans, *KDL_ROT_quat]], axis=0
            )
            ## Now calculate the Coordinates of points in World frame!!!
            #First,
            #Calculate KDL frame between the World and point lying on the plane. (World to cam X cam to plane)
            KDL_World_plane_frame = KDL_World_cam_frame * KDL_Camera_plane_frame
            #Extract world coordinates of the points lying on the plane
            KDL_trans = KDL_World_plane_frame.p
            KDL_ROT_quat = KDL_World_plane_frame.M.GetQuaternion()
            #Store Coordinates of points in World frame.
            world_points_Coordinates = np.append(
                world_points_Coordinates, [[*KDL_trans, *KDL_ROT_quat]], axis=0
            )

        return np.delete(Camera_points_Coordinates, 0, axis=0), np.delete(world_points_Coordinates, 0, axis=0)

    def filter_generate_coordinates(self, result, spacing, TGT_final_trim, sample=3, remove_close_targets=True):
        """
        Filters generated coordinates based on Z-threshold and proximity.

        :param result: Open3D point cloud object.
        :param spacing: Spacing threshold for filtering.
        :param TGT_final_trim: Z-threshold for filtering points.
        :param sample: Number of samples to take (default: 3).
        :param remove_close_targets: Whether to remove close targets (default: True).
        :return: Tuple containing filtered (Camera frame coordinates, World frame coordinates).
        """
        cam_T, WC_T = self.generate_coordinates(result)
        wc_t_count = len(WC_T)
        #remove any values where z coord. is less than threshold
        ind = np.argwhere(WC_T[:, 2] < TGT_final_trim).flatten() #check if z values less than threshold...  
        WC_T = np.delete(WC_T, ind, axis=0)
        cam_T = np.delete(cam_T, ind, axis=0)
        print("Number of Targets eliminated due to Z thresholding:", wc_t_count - len(WC_T))

        if remove_close_targets and len(WC_T) > 3:  #Triggers if point cloud has more than 3 pts.
            i = 0
            while i < len(WC_T) - 1:
                dis = distance.euclidean(WC_T[i, :3], WC_T[i + 1, :3])
                if dis < 2 * spacing:
                    WC_T = np.delete(WC_T, i, axis=0)
                    cam_T = np.delete(cam_T, i, axis=0)
                else:
                    i += 1

        return cam_T, WC_T
    
    
    def generate_final_coordinates(self, world_coords, z_offset, eef_link, coord_skip=3):
        """
        Converts world coordinates to final target coordinates for both the camera and the end-effector (EEF).

        :param world_coords: Numpy array of world coordinates [x, y, z, qx, qy, qz, qw].
        :param z_offset: Offset in the Z direction for adjustment.
        :param eef_link: Name of the end-effector link in ROS.
        :param coord_skip: Step size for selecting waypoints (default: 3).
        :return: Tuple containing (Camera target coordinates, EEF target coordinates, Filtered world coordinates).
        """
        Cam_target_final_coordinates = []
        EEF_target_final_coordinates = []
        wc_new = []

        #Fetch transform between eef link (link8, or tcp) and optical depth cam
        transform_eef_camera_depth = self.fetch_transform('camera_depth_optical_frame', eef_link, quat=1)

        for id_x in range(0, len(world_coords), coord_skip):
            transform_world_plane = world_coords[id_x]
            wc_new.append(transform_world_plane)

            # Create a PyKDL frame for the world coordinate
            KDL_original_plane_frame = PyKDL.Frame(
                PyKDL.Rotation.Quaternion(*transform_world_plane[3:]),
                PyKDL.Vector(*transform_world_plane[:3])
            )

            # Apply the Z offset, #update original plane frame to new location
            trans_x, trans_y, trans_z = KDL_original_plane_frame * PyKDL.Vector(0, 0, z_offset)
            KDL_original_plane_frame.p = PyKDL.Vector(trans_x, trans_y, trans_z)

            # Mirror the plane frame to match the camera frame
            KDL_flip_frame = PyKDL.Frame(PyKDL.Rotation.RPY(np.pi, 0., np.pi), PyKDL.Vector(0, 0, 0))
            KDL_final_frame = KDL_original_plane_frame * KDL_flip_frame

            # Extract the transformed coordinates
            KDL_trans = KDL_final_frame.p
            KDL_ROT_quat = KDL_final_frame.M.GetQuaternion()
            final_coordinates = [KDL_trans[0], KDL_trans[1], KDL_trans[2], KDL_ROT_quat[0], KDL_ROT_quat[1], KDL_ROT_quat[2], KDL_ROT_quat[3]]

            #self.Publish_coordinates([final_coordinates], "world", 'Camera_Target', static = False)
            # Store final camera target coordinates
            Cam_target_final_coordinates.append(final_coordinates)

            # Convert from camera to EEF coordinates
            KDL_eef_cam_frame = PyKDL.Frame(
                PyKDL.Rotation.Quaternion(*transform_eef_camera_depth[3:]),
                PyKDL.Vector(*transform_eef_camera_depth[:3])
            )

            # Apply the same transformations to obtain the EEF target
            KDL_final_frame = KDL_original_plane_frame * KDL_flip_frame * KDL_eef_cam_frame
            KDL_trans = KDL_final_frame.p
            KDL_ROT_quat = KDL_final_frame.M.GetQuaternion()
            final_coordinates = [KDL_trans[0], KDL_trans[1], KDL_trans[2], KDL_ROT_quat[0], KDL_ROT_quat[1], KDL_ROT_quat[2], KDL_ROT_quat[3]]

            # Store final EEF target coordinates
            EEF_target_final_coordinates.append(final_coordinates)

        wc_new = np.array(wc_new)
        return Cam_target_final_coordinates, EEF_target_final_coordinates, wc_new


    def publish_coordinates(self, Coordinates, parent_name, child_name, static=False):
        """
        Publishes transformation frames in ROS.

        :param Coordinates: List of coordinate frames to publish (format: [[x, y, z, qx, qy, qz, qw], ...]).
        :param parent_name: Name of the parent frame in ROS.
        :param child_name: Name of the child frame in ROS.
        :param static: Whether the transformation is static (default: False).
        """
        for index, coord in enumerate(Coordinates):
            if static:
                static_t = TransformStamped()
                static_t.header.stamp = rospy.Time.now()
                static_t.header.frame_id = parent_name #"camera_depth_optical_frame"
                static_t.child_frame_id = f"{child_name}_static_{index}"
            else:
                t = TransformStamped()
                t.header.stamp = rospy.Time.now()
                t.header.frame_id = parent_name
                t.child_frame_id = f"{child_name}_{index}"

            transform = static_t if static else t
            transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = coord[:3]
            transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w = coord[3:]

            if static:
                self.static_br.sendTransform(transform)
            else:
                self.br.sendTransform(transform)

            time.sleep(0.001)


    def get_planning_frame(self, move_group):
        """
        Retrieves the planning frame of the robot.
        
        :param move_group: MoveIt! move group interface.
        :return: The name of the planning frame.
        """
        return move_group.get_planning_frame()
    
    def get_end_effector_link(self, move_group):
        """
        Retrieves the end-effector link of the robot.
        
        :param move_group: MoveIt! move group interface.
        :return: The name of the end-effector link.
        """
        return move_group.get_end_effector_link()
    
    def get_group_names(self, robot):
        """
        Retrieves the list of available planning groups in the robot.
        
        :param robot: MoveIt! RobotCommander interface.
        :return: List of planning group names.
        """
        return robot.get_group_names()
    
    def get_robot_state(self, robot):
        """
        Retrieves the current state of the robot.
        
        :param robot: MoveIt! RobotCommander interface.
        :return: The current robot state as a MoveIt! RobotState object.
        """
        return robot.get_current_state()    

    def plan_cartesian_path(self, move_group, Batch_Profiles, eef_step=0.01, jump_threshold=0.0, velocity_scale=0.1, acceleration_scale=0.1):
        """
        Plans a Cartesian path for the robot using MoveIt!.
    
        :param move_group: MoveIt! MoveGroupCommander instance.
        :param Batch_Profiles: List of multiple profiles containing camera and end-effector targets [cam_tgt, eef_tgt].
        :param eef_step: Step size for end-effector movement (default: 0.01).
        :param jump_threshold: Maximum allowed jump in joint-space (default: 0.0 to disable jump detection).
        :param velocity_scale: Scaling factor for velocity (default: 0.5).
        :param acceleration_scale: Scaling factor for acceleration (default: 0.5)
        :return: Tuple (trajectory, fraction), where:
                 - trajectory: The planned trajectory if successful.
                 - fraction: Percentage of the path successfully planned (1.0 means full path).
        """

        waypoints=[]
        for profile in Batch_Profiles:  #Multiple Profiles
            #cam_tgt = profile[0]           #Multiple coordinates in each profile...
            eef_tgt = profile[1]
            for id_x in range(0,len(eef_tgt)):  
        
                pose_goal = Pose()
                pose_goal.position.x = eef_tgt[id_x][0]
                pose_goal.position.y = eef_tgt[id_x][1]
                pose_goal.position.z = eef_tgt[id_x][2]
                pose_goal.orientation.x = eef_tgt[id_x][3]
                pose_goal.orientation.y = eef_tgt[id_x][4]
                pose_goal.orientation.z = eef_tgt[id_x][5]
                pose_goal.orientation.w = eef_tgt[id_x][6]
                waypoints.append(pose_goal)
        
        # Compute Cartesian path
        (trajectory, fraction) = move_group.compute_cartesian_path(
            waypoints,  # Waypoints list
            eef_step,   # Step size between waypoints
            jump_threshold  # Maximum allowed joint-space jump
        )
        trajectory = move_group.retime_trajectory(move_group.get_current_state(), trajectory, 
                                           velocity_scaling_factor = velocity_scale, 
                                           acceleration_scaling_factor = acceleration_scale, 
                                           algorithm = "iterative_time_parameterization"
                                          )
        print(f"Planned {fraction * 100:.2f}% of the path.")
        return trajectory, fraction 

    def display_trajectory(self, move_group, robot, display_trajectory_publisher, plan):
        """
        Displays the planned trajectory in RViz.

        :param move_group: MoveIt! move group interface.
        :param robot: MoveIt! RobotCommander interface.
        :param display_trajectory_publisher: Trajectory publish
        :param plan: The planned trajectory (RobotTrajectory).
        """
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = robot.get_current_state()
        display_trajectory.trajectory.append(plan)
        display_trajectory_publisher.publish(display_trajectory)

    def execute_plan(self, move_group, plan):
        """
        Executes a pre-planned trajectory.
        
        :param move_group: MoveIt! move group interface.
        :param plan: The planned trajectory (RobotTrajectory).
        """
        move_group.execute(plan, wait=True)

    def validate_target(self, move_group, coord, plan_time=0.5):
        """
        validates and checks if targets are reachable.
    
        :param move_group: MoveIt! move group interface.
        :param coord: List containing [x, y, z, qx, qy, qz, qw].
        :param plan_time (optional): Set planning time to find solution. Default: 0.5 
        :return: Boolean indicating whether the plan was successful.
        """
        pose_goal = Pose()
        pose_goal.position.x = coord[0]
        pose_goal.position.y = coord[1]
        pose_goal.position.z = coord[2]
        pose_goal.orientation.x = coord[3]
        pose_goal.orientation.y = coord[4]
        pose_goal.orientation.z = coord[5]
        pose_goal.orientation.w = coord[6]
    
        move_group.set_pose_target(pose_goal)
        move_group.set_planning_time(plan_time)
        success = move_group.plan()
        move_group.clear_pose_targets()
        return success[0]
        
    def filter_targets(self, move_group, Batch_Profiles, plan_time=0.5):
        """
        checks if targets are reachable and removes unreachable targets.
    
        :param move_group: MoveIt! move group interface.
        :param Batch_Profiles: List containing multiple profiles.
                               Format:
                                      Batch_Profiles = [prof0, prof1,...] 
                                      > prof0 = [cam_tgts, eef_tgts]
                                      >> cam_tgts = [cam_tgt0, cam_tgt1,...]
                                      >> eef_tgts = [eef_tgt0, eef_tgt1,...]
                                      >>> cam_tgt0 = [x, y, z, qx, qy, qz, qw]
                                      >>> eef_tgt0 = [x, y, z, qx, qy, qz, qw]
    
        :param plan_time (optional): Set planning time to find solution. Default: 0.5 
        :return: Boolean indicating whether the plan was successful.
        """
        
        filtered_Batch_Profiles = []
        for profile in Batch_Profiles:
            cam_tgts = profile[0]
            eef_tgts = profile[1]
            #print("eef tgts", len(eef_tgts))
            new_cam_tgts = []
            new_eef_tgts = []
            for id_x in range(0,len(eef_tgts)):
                
                success = self.validate_target(move_group, eef_tgts[id_x], plan_time=plan_time)
                #print(success)
                if success:
                    #print([cam_tgt[id_x], eef_tgt[id_x]])
                    new_cam_tgts.append(cam_tgts[id_x])
                    new_eef_tgts.append(eef_tgts[id_x])
                    
            if len(new_cam_tgts) and len(new_eef_tgts) > 0: 
                filtered_Batch_Profiles.append([new_cam_tgts,new_eef_tgts])
            
        return filtered_Batch_Profiles
    
    def go_to_coord_goal(self, move_group, coord):
        """
        Moves the robot to a specified coordinate goal.
    
        :param move_group: MoveIt! move group interface.
        :param coord: List containing [x, y, z, qx, qy, qz, qw].
        :return: Boolean indicating whether the motion was successful.
        """
        pose_goal = Pose()
        pose_goal.position.x = coord[0]
        pose_goal.position.y = coord[1]
        pose_goal.position.z = coord[2]
        pose_goal.orientation.x = coord[3]
        pose_goal.orientation.y = coord[4]
        pose_goal.orientation.z = coord[5]
        pose_goal.orientation.w = coord[6]
    
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()
        return success
    
    def go_to_joint_state(self, move_group, joint_goal):
        """
        Moves the robot to a specified joint state.
    
        :param move_group: MoveIt! move group interface.
        :param joint_goal: List of target joint values.
        :return: Boolean indicating whether the motion was successful.
        """
        success = move_group.go(joint_goal, wait=True)
        move_group.stop()
        return success

    
    def create_robot_targets(self, move_group, profiles, plan_time=0.1, remove_close_targets=False):
        # create robot targets
        Batch_Profiles = []
        for idx, profile in enumerate(profiles, start=1):
            points = np.asarray(profile.points)
            print(f"Profile {idx} with {len(points)} points")
            _, world_coords  = self.filter_generate_coordinates(profile, spacing=self.spacing, TGT_final_trim=self.tgt_final_trim, sample=self.tgt_coord_samples, remove_close_targets=remove_close_targets)
            print(f"Pointcloud coordinates: {len(world_coords)}")
            if self.tgt_reverse:
                world_coords = world_coords[::-1]
            eef_link = move_group.get_end_effector_link()
            cam_tgt, eef_tgt, wc_new = self.generate_final_coordinates(world_coords, self.z_offset, eef_link, self.coord_skip)
            print(f"EEF coordinates after filter: {len(eef_tgt)}\n")
            Batch_Profiles.append([cam_tgt, eef_tgt])

        # remove unreachable targets
        Batch_Profiles = self.filter_targets(move_group, Batch_Profiles, plan_time)
        if len(Batch_Profiles)>0:
            cam_targets = [np.array(cam_tgt) for cam_tgt, _ in Batch_Profiles]
            merged_targets = np.vstack(cam_targets)
            print(f"Total Targets after removing unreachable targets: {len(merged_targets)} From {len(Batch_Profiles)} Profiles.")
            if self.tgt_preview:
                self.publish_coordinates(merged_targets, "world", "Camera_Target", static=False)
            else:
                print("Warning No Targets generated!")
        
        return Batch_Profiles

    
    def visualize_point_clouds(self, point_clouds, window_name="Point Cloud Visualization"):
        """
        Visualize one or multiple point clouds using Open3D with different colors and keyboard controls.
        
        Controls:
        - 'N' -> Show the next point cloud
        - 'P' -> Show the previous point cloud
        - 'R' -> Reset and show all point clouds
        - 'Q' -> Quit visualization
    
        :param point_clouds: A single Open3D PointCloud or a list of PointClouds.
        :param window_name: Name of the visualization window.
        """
        if not isinstance(point_clouds, list):
            point_clouds = [point_clouds]  # Convert single point cloud to a list
    
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name)
    
        # Create a coordinate frame for reference
        mesh_frame = self.mesh
    
        # Assign a unique color to each point cloud
        colored_clouds = []
        for pc in point_clouds:
            # Compute normals
            pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.06, max_nn=30))
            pc.orient_normals_towards_camera_location(camera_location=[0, 0, 0])
            # Optional: Visualize normals by increasing the size of normal vectors
            pc.normals = o3d.utility.Vector3dVector(np.asarray(pc.normals) * 0.1)  # Scale normal vectors for better visibility
            color = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
            pc.paint_uniform_color(color)
            colored_clouds.append(pc)


    
        index = [0]  # Using a mutable list so callbacks can modify it
    
        def update_view():
            """ Clears and updates the visualization window with the current point cloud """
            vis.clear_geometries()
            vis.add_geometry(mesh_frame)
            vis.add_geometry(colored_clouds[index[0]])
            # Set view parameters (Zoom, Front, LookAt, Up)
            #view_ctl = vis.get_view_control()
            #view_ctl.set_zoom(self.zoom_def)  # Set zoom level
            #view_ctl.set_front(self.front_def)  # Set front vector
            #view_ctl.set_lookat(self.lookat_def)  # Set look-at point
            #view_ctl.set_up(self.up_def)  # Set up vector
            vis.poll_events()
            vis.update_renderer()
    
        def next_point_cloud(vis):
            """ Show the next point cloud when 'N' is pressed """
            if len(colored_clouds) > 1:
                index[0] = (index[0] + 1) % len(colored_clouds)
                update_view()
    
        def prev_point_cloud(vis):
            """ Show the previous point cloud when 'P' is pressed """
            if len(colored_clouds) > 1:
                index[0] = (index[0] - 1) % len(colored_clouds)
                update_view()
    
        def reset_view(vis):
            """ Reset and show all point clouds when 'R' is pressed """
            vis.clear_geometries()
            vis.add_geometry(mesh_frame)
            for pc in colored_clouds:
                vis.add_geometry(pc)
            vis.poll_events()
            vis.update_renderer()
    
        # Register keyboard callbacks
        vis.register_key_callback(ord("N"), next_point_cloud)
        vis.register_key_callback(ord("P"), prev_point_cloud)
        vis.register_key_callback(ord("R"), reset_view)
        vis.register_key_callback(ord("Q"), lambda vis: vis.destroy_window())
    
        # Start visualization with the first point cloud
        update_view()
        vis.run()

    def array_to_data(self, array):
        im = PIL_img.fromarray(array)
        output_buffer = BytesIO()
        im.save(output_buffer, format="PNG")
        data = output_buffer.getvalue()
        return data

    def fetch_cloud_image(self, pointCloud, RX=0, RY=0, RZ=0):
        mesh = self.mesh
        mesh_pc = mesh.sample_points_uniformly(number_of_points=1000000, use_triangle_normal=False)
        tmp_cloud = copy.deepcopy(pointCloud)  # To avoid overwriting the original point cloud

        tmp_cloud += mesh_pc
        tmp_Rot = pointCloud.get_rotation_matrix_from_xyz((np.radians(RX), np.radians(RY), np.radians(RZ)))

        tmp_cloud.rotate(tmp_Rot, center=(0, 0, 0))

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=640, height=480)
        vis.add_geometry(tmp_cloud)
        vis.poll_events()
        vis.update_renderer()
        color = vis.capture_screen_float_buffer(True)
        vis.destroy_window()
        color = (255.0 * np.asarray(color)).astype(np.uint8)
        color = self.array_to_data(color)  # Format according to the GUI requirements
        return color

    def new_world_coordinates(self, world_coords, eef_link, new_z_offset=0.0, old_z_offset=0.0, coord_skip=1):
        """
        Computes new world coordinates by applying a Z-offset.

        :param world_coords: List of world coordinates.
        :param eef_link: End-effector link.
        :param new_z_offset: New Z-offset to be applied.
        :param old_z_offset: Previous Z-offset.
        :param coord_skip: Step size for selecting waypoints.
        :return: List of updated world coordinates.
        """
        z_offset = new_z_offset
        new_world_coordinates_store = []

        for id_x in range(0, len(world_coords), coord_skip):
            transform_world_plane = world_coords[id_x]
            KDL_original_plane_frame = PyKDL.Frame(
                PyKDL.Rotation.Quaternion(*transform_world_plane[3:]),
                PyKDL.Vector(*transform_world_plane[:3])
            )

            trans_x, trans_y, trans_z = KDL_original_plane_frame * PyKDL.Vector(0, 0, z_offset)  #Add offset
            #update original plane frame to new location
            KDL_original_plane_frame.p = PyKDL.Vector(trans_x, trans_y, trans_z)

            KDL_final_frame = KDL_original_plane_frame
            KDL_trans = KDL_final_frame.p
            KDL_ROT_quat = KDL_final_frame.M.GetQuaternion()
            final_coordinates = [KDL_trans[0], KDL_trans[1], KDL_trans[2], KDL_ROT_quat[0], KDL_ROT_quat[1],
                                 KDL_ROT_quat[2], KDL_ROT_quat[3]]

            new_world_coordinates_store.append(final_coordinates)

        return new_world_coordinates_store

    def cluster_selection_gui(self, clouds):
        """
        Opens a GUI for selecting clusters from a set of point clouds.

        :param clouds: List of point clouds.
        :return: Combined point cloud of selected clusters.
        """
        dat = []  # Store cluster images
        pcd_combined = o3d.geometry.PointCloud()

        for current_cloud in clouds:
              #different point clouds to be viewed, set view by RX,RY,RZ
            color = self.fetch_cloud_image(current_cloud, RX=120, RZ=180)
            dat.append(color)

        result = subprocess.run([sys.executable, "Cluster_selection_gui.py"], capture_output=True, text=True,
                                check=True, shell=False, input=repr(dat))
        OP = literal_eval(result.stdout)
        selected_PC = OP

        for k in selected_PC:
            pcd_combined += clouds[k]  # Combine selected point clouds

        return pcd_combined

    def front_gui(self, data_in=0):
        """
        Opens the main mode selection GUI.

        :param data_in: Input data for the GUI.
        :return: Processed output from the GUI.
        """
        result = subprocess.run([sys.executable, "Front_gui.py"], capture_output=True, text=True, check=True,
                                shell=False, input=repr(data_in))
        data_out = literal_eval(result.stdout)
        return data_out

    def setting_gui(self, data_in):
        """
        Opens the settings GUI.

        :param data_in: Input settings data.
        :return: Processed output from the GUI.
        """
        result = subprocess.run([sys.executable, "Settings_gui.py"], capture_output=True, text=True, check=True,
                                shell=False, input=repr(data_in))
        data_out = literal_eval(result.stdout)
        return data_out

    def manual_gui(self, manual_offset, TGT_save):
        """
        Opens the manual adjustment GUI.

        :param manual_offset: Current manual offset.
        :param TGT_save: Target save state.
        :return: Tuple containing updated pose index, manual offset, target save state, and exit flag.
        """
        dat = []  # Store images

        for f in sorted(glob.iglob("VI_appdata/Robo_object_positions/*")):
            im = PIL_img.open(f)
            output_buffer = BytesIO()
            im.save(output_buffer, format="PNG")
            data = output_buffer.getvalue()
            dat.append(data)

        result = subprocess.run([sys.executable, "Manual_gui.py"], capture_output=True, text=True, check=True,
                                shell=False, input=repr([dat, manual_offset, TGT_save]))

        OP = literal_eval(result.stdout)
        return OP[0], OP[1], OP[2], OP[3]  # pose_idx, manual_offset, TGT_save, Exit_flag

    def profile_selection_gui(self, X_profiles, Y_profiles, selected_Profiles=[]):
        """
        Opens the profile selection GUI.

        :param X_profiles: Available X profiles.
        :param Y_profiles: Available Y profiles.
        :param selected_Profiles: Previously selected profiles.
        :return: Updated selected profiles, X profiles, Y profiles, and OK flag.
        """
        result = subprocess.run([sys.executable, "Profile_selection_gui.py"], capture_output=True, text=True,
                                check=True, shell=False, input=repr([X_profiles, Y_profiles, selected_Profiles]))

        OP = literal_eval(result.stdout)
        return OP[0], OP[1], OP[2], OP[3]  # selected_Profiles, x_pros, y_pros, int(OK_flag)

    
    def clear_collision_scene(self, planning_scene):
        planning_scene.remove_world_object()
        print("Scene cleared!\nObstacle avoidance deactivated!")
    
    
    def update_collision_scene(self, planning_scene, alpha = 0, flip=False, obstacle=True):
        planning_scene.remove_world_object()
        object_mesh = o3d.geometry.TriangleMesh()
        radii = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.04, 0.1]
        
        pcd = self.load_point_cloud(self.samples, self.offset_y, self.offset_z, self.manual_offset, 0.01, self.trim_base, Hide_prev=True, Dbug=False, eval_tag=True) #eval_tag :generates wrt world
        print(f"Created collision cloud with {len(pcd.points)} points.")
        #pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.06, max_nn=30)) #radius in meters
        #pcd.orient_normals_consistent_tangent_plane( k = round( len(pcd.points) / 6) )
        #pcd.orient_normals_towards_camera_location(camera_location=[0, 0, 1])
        if flip:
            pcd.normals = o3d.utility.Vector3dVector(-np.asarray(pcd.normals)) #Orient normals outwards

        results = self.cluster_point_cloud(pcd, eps=self.eps, min_points=self.min_points, obstacle=obstacle) #segment to get individual objects, better for mesh creation

        if alpha > 0:
            print(f"Using Alpha shapes.  alpha={alpha:.3f}")

            for result in results:
                tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(result)
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(result, alpha, tetra_mesh, pt_map)
                object_mesh += mesh
                object_mesh.compute_vertex_normals()
        else:
            for result in results:
                object_mesh += o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(result, o3d.utility.DoubleVector(radii))

            object_mesh.compute_vertex_normals()
        # Save mesh to a temporary file
        o3d.io.write_triangle_mesh("mesh.stl", object_mesh)

        # Add to planning scene
        pose = PoseStamped()
        pose.header.frame_id = "world"
        pose.pose.orientation.w = 1.0
        planning_scene.add_mesh("pointcloud_mesh", pose, "mesh.stl")
        print("Scene updated!\nObstacle avoidance active!")


    
    
    
class PointCloudViewer:
    def __init__(self, pcds, result_holder, say_func, camera_params_path='view.json'):
        if isinstance(pcds, np.ndarray):
            self.pcds = pcds.tolist()
        elif not isinstance(pcds, list):
            self.pcds = [pcds]
        else:
            self.pcds = pcds
        self.say = say_func
        self.camera_params_path = camera_params_path
        self.camera_params = o3d.io.read_pinhole_camera_parameters(camera_params_path)
        #self.camera_params = o3d.io.read_pinhole_camera_parameters("VI_appdata/view.json")
        self.world_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        self.result_holder = result_holder

        self.current_idx = 0
        self.selected_indices = set()
        self.should_quit = [False]
        self.animate = [False]
        self.view_flag = True

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="Point Cloud Viewer", visible=True)
        self.ctr = self.vis.get_view_control()

        self.vis.add_geometry(self.world_mesh)
        self.vis.add_geometry(self.pcds[self.current_idx])

        # Key bindings
        self.vis.register_key_callback(ord("D"), lambda vis: self.next_pc())          #next point cloud
        self.vis.register_key_callback(ord("A"), lambda vis: self.prev_pc())          #previous point cloud
        self.vis.register_key_callback(ord("Q"), lambda vis: self.exit_viewer())      #Quit viewer
        self.vis.register_key_callback(ord("R"), lambda vis: self.rotate_object()) #Start rotation
        self.vis.register_key_callback(ord("W"), lambda vis: self.select_current())    #deselect point cloud
        self.vis.register_key_callback(ord("S"), lambda vis: self.deselect_current()) #select point cloud
        self.vis.register_key_callback(ord("V"), lambda vis: self.reset_view())     #reset view
        self.vis.register_key_callback(ord("T"), lambda vis: self.stop_rotation())  #Stop rotation
        self.vis.register_key_callback(ord("Z"), lambda vis: self.save_view_point()) #Save view point

        self.update_view()

    def save_view_point(self):
        self.say("Saving current viewpoint")
        self.camera_params = self.vis.get_view_control().convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(self.camera_params_path, self.camera_params)
        
    def update_view(self):
        self.vis.clear_geometries()
        self.vis.add_geometry(self.world_mesh)
        self.vis.add_geometry(self.pcds[self.current_idx])
        self.ctr.convert_from_pinhole_camera_parameters(self.camera_params)

    def next_pc(self):
        self.say("Showing next object!")
        self.current_idx = (self.current_idx + 1) % len(self.pcds)
        self.update_view()

    def prev_pc(self):
        self.say("Showing previous object!")
        self.current_idx = (self.current_idx - 1 + len(self.pcds)) % len(self.pcds)
        self.update_view()

    def select_current(self):
        self.say("selecting object")
        self.selected_indices.add(self.current_idx)
        print(f"Selected: {self.current_idx}. Now selected: {sorted(self.selected_indices)}")

    def deselect_current(self):
        self.say("deselecting object")
        self.selected_indices.discard(self.current_idx)
        print(f"Deselected: {self.current_idx}. Now selected: {sorted(self.selected_indices)}")

    def exit_viewer(self):
        self.say("Exiting viewer!") 
        self.should_quit[0] = True

    def rotate_object(self):
        self.say("rotating object!")    
        self.animate[0] = not self.animate[0]
        if self.animate[0]:
            print("Rotation animation: ON")
        else:
            print("Rotation animation: OFF")

    def reset_view(self):
        self.say("Resetting view!")
        self.ctr.convert_from_pinhole_camera_parameters(self.camera_params)
        print("View reset to default camera parameters.")

    def stop_rotation(self):
        self.say("Stopping rotation!")
        self.animate[0] = False
        print("Rotation animation stopped.")
    
    
    def run(self):
        try:
            while not self.should_quit[0]:
                self.vis.poll_events()
                self.vis.update_renderer()
                time.sleep(0.005)

                # Animate rotation if enabled
                if self.animate[0]:
                    self.ctr.rotate(-2, 0)
        finally:
            self.vis.destroy_window()
            selected_pcs = [deepcopy(self.pcds[i]) for i in sorted(self.selected_indices)]
            
            #results = o3d.geometry.PointCloud()
            #if selected_pcs:
            #    #print(f"Retrieved {len(selected_pcs)} selected point clouds.")
            #    for pcd in selected_pcs:
            #        results += pcd
            #self.result_holder.extend(results)
            
            self.result_holder.extend(selected_pcs)
            
            #self.result_holder.append(results)
            print(f"Total selected point clouds: {len(selected_pcs)}")
            


class Inspector:
    def __init__(self, pcds, spacing, result_holder, say_func, camera_params_path='view.json'):
 
        if isinstance(pcds, np.ndarray):
            self.pcds = pcds.tolist()
        elif not isinstance(pcds, list):
            self.pcds = [pcds]
        else:
            self.pcds = pcds
            
        self.current_idx = 0
        self.pcd = self.pcds[self.current_idx]
        self.spacing = spacing
        self.say = say_func
        self.camera_params_path = camera_params_path
        self.camera_params = o3d.io.read_pinhole_camera_parameters(camera_params_path)
        #self.camera_params = o3d.io.read_pinhole_camera_parameters("VI_appdata/view.json")
        self.world_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        self.result_holder = result_holder

        self.should_quit = [False]
        self.animate = [False]
        self.view_flag = True

        self.lock = threading.Lock()
        self.dirty = [False]

        self.selected_profiles_dict = {}  # Store all selected profiles

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="Point Cloud Viewer", visible=True)
        self.ctr = self.vis.get_view_control()

        self.vis.add_geometry(self.world_mesh)
        self.vis.add_geometry(self.pcds[self.current_idx])
        
        self.vis.register_key_callback(ord("D"), lambda vis: self.next_pc())          #next point cloud
        self.vis.register_key_callback(ord("A"), lambda vis: self.prev_pc())          #previous point cloud
        self.vis.register_key_callback(ord("Q"), lambda vis: self.exit_viewer())
        self.vis.register_key_callback(ord("R"), lambda vis: self.rotate_object())
        self.vis.register_key_callback(ord("V"), lambda vis: self.reset_view())
        self.vis.register_key_callback(ord("T"), lambda vis: self.stop_rotation())
        self.vis.register_key_callback(ord("Z"), lambda vis: self.save_view_point())

        self.update_view()

        self.reset_highlight(init=True)

    def save_view_point(self):
        self.say("Saving current viewpoint")
        self.camera_params = self.vis.get_view_control().convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(self.camera_params_path, self.camera_params)
    
    def update_view(self):
        with self.lock:
            self.vis.clear_geometries()
            self.vis.add_geometry(self.world_mesh)
            self.vis.add_geometry(self.pcds[self.current_idx])
            self.pcd = self.pcds[self.current_idx] #select new PCD
            self.ctr.convert_from_pinhole_camera_parameters(self.camera_params)
            self.dirty[0] = False

    def next_pc(self):
        self.say("Showing next object!")
        self.current_idx = (self.current_idx + 1) % len(self.pcds)
        self.update_view()

    def prev_pc(self):
        self.say("Showing previous object!")
        self.current_idx = (self.current_idx - 1 + len(self.pcds)) % len(self.pcds)
        self.update_view()
        
    def exit_viewer(self):
        self.say("Exiting viewer!")
        self.should_quit[0] = True
        print("Viewer quitting...")

    def rotate_object(self):
        self.say("rotating object!")
        self.animate[0] = not self.animate[0]
        print(f"Rotation animation: {'ON' if self.animate[0] else 'OFF'}")

    def reset_view(self):
        self.say("Resetting view!")
        self.ctr.convert_from_pinhole_camera_parameters(self.camera_params)
        self.reset_highlight()
        print("View reset to default camera parameters.")

    def stop_rotation(self):
        self.say("Stopping rotation!")
        self.animate[0] = False
        print("Rotation animation stopped.")

    def reset_highlight(self, init=False):

        if init:
            for cld in self.pcds:
                if not cld.has_colors():
                    return
                colors = np.asarray(cld.colors)
                colors[:] = [0.5, 0.5, 0.5]
                self.dirty[0] = True
                print("Highlight reset.")
        else:
            pcd = self.pcd
            self.selected_profiles_dict.pop(self.current_idx, None) #remove key and values for current point cloud
            if not pcd.has_colors():
                return
            colors = np.asarray(pcd.colors)
            colors[:] = [0.5, 0.5, 0.5]
            self.dirty[0] = True
            print("Highlight reset.")

    
          
    
    def highlight_points(self, indices: list):
        with self.lock:
            pcd = self.pcd
            #self.reset_highlight()
            if not pcd.has_colors():
                pcd.colors = o3d.utility.Vector3dVector(np.full((len(pcd.points), 3), 0.5))
            colors = np.asarray(pcd.colors)
            colors[indices] = [1, 0, 0]
            self.pcd = pcd
            self.dirty[0] = True
            print(f"Highlighted {len(indices)} points in red.")
        
    def _profile_bin(self, original_PC, axis: str):
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
        ctr = 0
        profile_bins = []
        PC_BBOX = original_PC.get_axis_aligned_bounding_box()
        
        minB, maxB = PC_BBOX.min_bound[axis_idx], PC_BBOX.max_bound[axis_idx]   #profiles_X or profiles_Y or profiles_Z

        CurrB = minB
        while CurrB < maxB - 0.5 * self.spacing: #from left most to right most
            LowerB = minB + ctr * self.spacing + 0.005 #if we shift the X or Y coordinates in multiples of spacing, we should get different lines.
                                                  #+ some extra width to reduce outlier points in PC
            if ctr == 0:
                LowerB = minB + ctr * self.spacing
                CurrB = LowerB + self.spacing / 2 #lower + spacing/2 only for first condition
            else:
                CurrB = LowerB + self.spacing  #lower + spacing 
            profile_bins.append(LowerB)
            #print("CurrB:",CurrB,"MaxB:",maxB,"diff:",maxB-CurrB)
            ctr += 1

        return np.array(profile_bins)
    
    def profiles_available(self, axis: str):
        bins = self._profile_bin(self.pcd, axis)
        return len(bins)
        
    def _compute_grid_indices(self, axis: str):
        points = np.asarray(self.pcd.points)
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
        coord = points[:, axis_idx]
        bins = self._profile_bin(self.pcd, axis)
        #bins = np.linspace(coord.min(), coord.max(), num_bins + 1)
        indices = np.digitize(coord, bins) #- 1
        return indices, len(bins) #num_bins
    
    def _filter_points_in_line(self, indices, axis='y', tolerance=0.001):
        points = np.asarray(self.pcd.points)
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
        ref_value = np.median(points[indices, axis_idx])
        deviation = np.abs(points[indices, axis_idx] - ref_value)
        filtered = indices[deviation <= tolerance]
        return filtered
    
    def filter_profile(self, pcd):
        points = np.asarray(pcd.points)
        variances = np.var(points, axis=0) # Compute variances along each axis (x, y, z)
        min_axis = np.argmin(variances) # Find axis with smallest variance, that's the axis around which path is planned.
        medians = np.median(points, axis=0) # Compute medians along each axis (x, y, z)
        points[:, min_axis] = medians[min_axis] # Set all values along that axis to the median (shift that axis coordinate of all points)
        
        pcd.points = o3d.utility.Vector3dVector(points) # Update point cloud and downsample
        return pcd.voxel_down_sample(voxel_size=self.spacing)

    
    def _add_profiles_by_index(self, selected_slice):
    
        if self.current_idx in self.selected_profiles_dict:
            self.selected_profiles_dict[self.current_idx].extend([selected_slice])
        else:
            self.selected_profiles_dict[self.current_idx] = [selected_slice]

    def _select_centered_profile(self, axis, reverse):
        row_idx, num = self._compute_grid_indices(axis)
        target = num // 2
        indices = np.where(row_idx == target)[0]
        filtered = self._filter_points_in_line(indices, axis=axis)
        selected_slice = self.pcd.select_by_index(filtered.tolist()) #select profile
        if self.is_sort_by_axis(selected_slice):
            selected_slice = self.sort_by_axis(selected_slice, axis_str=axis, reverse=reverse, resample=False)
        else:
            selected_slice = self.sort_pcd(selected_slice, reverse=reverse) #Sort it
        self._add_profiles_by_index(selected_slice) #append to selected profiles dict
        #print(f"_select_centered_profile func.  :  {axis}: {self.selected_profiles_dict}\n\n")
        self.highlight_points(filtered.tolist()) #highlight

        
    def select_centered_profile_around_y(self, reverse=False):
        self.say("selecting centered profile around y axis")
        self._select_centered_profile('y', reverse)
        
    def select_centered_profile_around_x(self, reverse=False):
        self.say("selecting centered profile around x axis")
        self._select_centered_profile('x', reverse)
        
    def select_centered_profile_around_z(self, reverse=False):
        self.say("selecting centered profile around z axis")
        self._select_centered_profile('z', reverse)

    
    def _select_specific_profile(self, axis, target_idx, reverse):
        idx, _ = self._compute_grid_indices(axis)
        indices = np.where(idx == target_idx)[0]
        filtered = self._filter_points_in_line(indices, axis=axis)
        selected_slice = self.pcd.select_by_index(filtered.tolist()) #select profile
        if self.is_sort_by_axis(selected_slice):
            selected_slice = self.sort_by_axis(selected_slice, axis_str=axis, reverse=reverse, resample=False)
        else:
            selected_slice = self.sort_pcd(selected_slice, reverse=reverse) #Sort it
        self._add_profiles_by_index(selected_slice) #append to selected profiles dict
        #print(f"_select_specific_profile func.  :  {axis}: {self.selected_profiles_dict}\n\n")
        self.highlight_points(filtered.tolist()) #highlight
    
    
    def select_specific_profile_around_y(self, target_idx, reverse=False):
        self.say(f"selecting {target_idx}th profile around y axis")
        self._select_specific_profile('y', target_idx, reverse=reverse)

    def select_specific_profile_around_x(self, target_idx, reverse=False):
        self.say(f"selecting {target_idx}th profile around x axis")
        self._select_specific_profile('x', target_idx, reverse=reverse)

    def select_specific_profile_around_z(self, target_idx, reverse=False):
        self.say(f"selecting {target_idx}th profile around z axis")
        self._select_specific_profile('z', target_idx, reverse=reverse)           



        
    def select_profile_with_angle(self, angle_degrees=0, direction: str = 'right', width=None, height=None, reverse=False):
        self.say(f"selecting profile {angle_degrees} degrees towards {direction} of the object") 
        pcd = self.pcd
        center = pcd.get_center()
        points = np.asarray(pcd.points)
        
        angle_degrees = abs(max(min(90, angle_degrees), -90))  # clamp angle between -90 and 90
    
        if angle_degrees == 0 or angle_degrees == 90:
            point_spacing = self.spacing
        else:
            point_spacing = self.spacing / 2
    
        if direction == "right":
            angle_degrees = -angle_degrees
        angle_rad = np.deg2rad(angle_degrees)
        direction_vector = np.array([np.cos(angle_rad), np.sin(angle_rad), 0])
        up = np.array([0, 0, 1])
        side = np.cross(direction_vector, up)
    
        # Adjust width based on angle and point spacing
        if width is None:
            adjusted_width = point_spacing / max(abs(np.dot(side[:2], [1, 0])), abs(np.dot(side[:2], [0, 1])))
            width = adjusted_width * 1.1  # small margin
            print(f"Auto-adjusted width: {width:.5f} for angle {angle_degrees}°")
    
        if height is None:
            bbox = pcd.get_axis_aligned_bounding_box()
            height = np.linalg.norm(bbox.get_extent())
    
        # Define oriented bounding box and crop
        R = np.vstack([side, direction_vector, up]).T
        obb = o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=[width, height, 1.0])
        sliced_pcd = pcd.crop(obb)

        # Match points with original point cloud for highlighting
        mask = np.isin(np.round(points, 6), np.round(sliced_pcd.points, 6)).all(axis=1)
        indices = np.where(mask)[0].tolist()
        self.highlight_points(indices)
    
        print(f"Selected profile at {angle_degrees}° — selected {len(sliced_pcd.points)} points")

        if angle_degrees == 0 or abs(angle_degrees) == 90:
            #clean each profile to remove parallel points and get a single smooth profile.
            sliced_pcd = self.filter_profile(sliced_pcd)
        
        #Sort the points in profile.
        sliced_pcd = self.sort_pcd(sliced_pcd, reverse=reverse)
        
        # Store profile
        self._add_profiles_by_index(sliced_pcd) #append to selected profiles dict
        #print(f"select_profile_with_angle func.  :  {self.selected_profiles_dict}\n\n")


    def select_profile_with_angle_old(self, angle_degrees=0, direction: str = 'right', width=None, height=None, reverse=False):
        self.say(f"selecting profile {angle_degrees} degrees towards {direction} of the object")
        pcd = self.pcd
        center = pcd.get_center()
        points = np.asarray(pcd.points)
        
        angle_degrees = abs(max(min(90, angle_degrees), -90))  # clamp angle between -90 and 90
    
        if angle_degrees == 0 or angle_degrees == 90:
            point_spacing = self.spacing
        else:
            point_spacing = self.spacing / 2
    
        if direction == "right":
            angle_degrees = -angle_degrees
        angle_rad = np.deg2rad(angle_degrees)
        direction_vector = np.array([np.cos(angle_rad), np.sin(angle_rad), 0])
        up = np.array([0, 0, 1])
        side = np.cross(direction_vector, up)
    
        # Adjust width based on angle and point spacing
        if width is None:
            adjusted_width = point_spacing / max(abs(np.dot(side[:2], [1, 0])), abs(np.dot(side[:2], [0, 1])))
            width = adjusted_width * 1.1  # small margin
            print(f"Auto-adjusted width: {width:.5f} for angle {angle_degrees}°")
    
        if height is None:
            bbox = pcd.get_axis_aligned_bounding_box()
            height = np.linalg.norm(bbox.get_extent())
    
        # Define oriented bounding box and crop
        R = np.vstack([side, direction_vector, up]).T
        obb = o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=[width, height, 1.0])
        sliced_pcd = pcd.crop(obb)

        # Match points with original point cloud for highlighting
        mask = np.isin(np.round(points, 6), np.round(sliced_pcd.points, 6)).all(axis=1)
        indices = np.where(mask)[0].tolist()
        self.highlight_points(indices)
    
        print(f"Selected profile at {angle_degrees}° — selected {len(sliced_pcd.points)} points")

        if angle_degrees == 0 or abs(angle_degrees) == 90:
            #clean each profile to remove parallel points and get a single smooth profile.
            sliced_pcd = self.filter_profile(sliced_pcd)

        if sliced_pcd.has_colors():
            sliced_colors = np.asarray(sliced_pcd.colors)
        # Sort points along the profile direction
        sliced_points = np.asarray(sliced_pcd.points)
        sliced_normals = np.asarray(sliced_pcd.normals)
        
        projections = sliced_points @ direction_vector  # project onto direction vector
    
        sort_indices = np.argsort(projections)
        if reverse:
            sort_indices = sort_indices[::-1]
    
        sorted_points = sliced_points[sort_indices]
        sorted_normals = sliced_normals[sort_indices]
        sliced_pcd.points = o3d.utility.Vector3dVector(sorted_points)
        sliced_pcd.normals = o3d.utility.Vector3dVector(sorted_normals)
        
        if sliced_pcd.has_colors():
            sorted_colors  = sliced_colors[sort_indices]
            sliced_pcd.colors = o3d.utility.Vector3dVector(sorted_colors)
            
        # Store profile
        self._add_profiles_by_index(sliced_pcd) #append to selected profiles dict
        #print(f"select_profile_with_angle_old func.  :  {self.selected_profiles_dict}\n\n")

        
        
    def _select_multiple_profiles(self, axis, n, reverse):
        row_col_z_idx, num = self._compute_grid_indices(axis)
        if n > num:
            n = num
        spacing = num // n
        targets = [i * spacing + spacing // 2 for i in range(n)]
        all_indices = []
        for i, t in enumerate(targets, start=1):
            idx = np.where(row_col_z_idx == t)[0]
            filtered = self._filter_points_in_line(idx, axis=axis)
            all_indices.extend(filtered.tolist())
            
            selected_slice = self.pcd.select_by_index(filtered.tolist()) #select profile
            if len(selected_slice.points) == 0:
                print(f"Warning: profile {i} is empty, skipping.")
                continue
            if self.is_sort_by_axis(selected_slice):
                selected_slice = self.sort_by_axis(selected_slice, axis_str=axis, reverse=reverse, resample=False)
            else:
                selected_slice = self.sort_pcd(selected_slice, reverse=reverse) #Sort it
            #self.selected_profiles.append(selected_slice) #append to list
            
            self._add_profiles_by_index(selected_slice) #append to selected profiles dict
            #print(f"_select_multiple_profiles func. :  {axis} :  {self.selected_profiles_dict}\n\n")
            
        self.highlight_points(all_indices)  #highlight
        
        
    def select_multiple_profiles_around_y(self, n=3, reverse=False):
        self.say(f"selecting {n} profiles around y axis")
        self._select_multiple_profiles('y', n, reverse=reverse)
        
    def select_multiple_profiles_around_x(self, n=3, reverse=False):
        self.say(f"selecting {n} profiles around x axis")
        self._select_multiple_profiles('x', n, reverse=reverse)

    def select_multiple_profiles_around_z(self, n=3, reverse=False):
        self.say(f"selecting {n} profiles around z axis")
        self._select_multiple_profiles('z', n, reverse=reverse)
        
        
    def select_multiple_profiles_with_angle(self, angle_degrees=0, direction: str ='right', n=5, reverse=False):
        self.say(f"selecting {n} profiles {angle_degrees} degrees towards {direction} of the object")
        #point_spacing = self.spacing
        pcd = self.pcd
        center = pcd.get_center()
        points = np.asarray(pcd.points)
        angle_degrees = abs(max(min(90, angle_degrees), -90)) #keep angle between -90 and +90
        if angle_degrees == 0 or angle_degrees == 90:
            point_spacing = self.spacing
        else:
            point_spacing = self.spacing / 2
            
        if direction =="right":
            angle_degrees = -angle_degrees
        angle_rad = np.deg2rad(angle_degrees)
        direction_vector = np.array([np.cos(angle_rad), np.sin(angle_rad), 0])
        up = np.array([0, 0, 1])
        side = np.cross(direction_vector, up)

        # Height = full cloud length
        height = np.linalg.norm(pcd.get_max_bound() - pcd.get_min_bound())
    
        # Width = spacing corrected for angle
        ortho_proj = np.abs(np.dot(side[:2], [1, 0])) + np.abs(np.dot(side[:2], [0, 1]))
        width = (point_spacing / ortho_proj) * 1.1
    
        # Project all points onto side vector to get range
        rel = points[:, :2] - center[:2]
        proj = rel @ side[:2]
        proj_min, proj_max = proj.min(), proj.max()
        total_span = proj_max - proj_min
        spacing = total_span / max(n - 1, 1)
    
        for i in range(n):
            offset_val = proj_min + i * spacing
            offset_vec = side * offset_val
            slice_center = center + offset_vec
    
            R = np.vstack([side, direction_vector, up]).T
            obb = o3d.geometry.OrientedBoundingBox(center=slice_center, R=R, extent=[width, height, 1.0])
            sliced_pcd = pcd.crop(obb)
            if len(sliced_pcd.points) == 0:
                print(f"Warning: slice {i+1} is empty, skipping.")
                continue
        
            mask = np.isin(np.round(points, 6), np.round(np.asarray(sliced_pcd.points), 6)).all(axis=1)
            indices = np.where(mask)[0].tolist()
            self.highlight_points(indices)
    
            print(f"Slice {i+1}/{n}: {len(sliced_pcd.points)} points at angle {angle_degrees}°")
    
            if angle_degrees == 0 or abs(angle_degrees) == 90:
                #clean each profile to remove parallel points and get a single smooth profile.
                sliced_pcd = self.filter_profile(sliced_pcd)
        
            #Sort the points in profile.
            
            sliced_pcd = self.sort_pcd(sliced_pcd, reverse=reverse)
            
            # Store profile
            self._add_profiles_by_index(sliced_pcd) #append to selected profiles dict
            #print(f"select_multiple_profiles_with_angle func. :  {self.selected_profiles_dict}\n\n")


    def select_multiple_profiles_with_angle_old(self, angle_degrees=0, direction: str ='right', n=5, reverse=False):
        self.say(f"selecting {n} profiles {angle_degrees} degrees towards {direction} of the object")
        #point_spacing = self.spacing
        pcd = self.pcd
        center = pcd.get_center()
        points = np.asarray(pcd.points)
        angle_degrees = abs(max(min(90, angle_degrees), -90)) #keep angle between -90 and +90
        if angle_degrees == 0 or angle_degrees == 90:
            point_spacing = self.spacing
        else:
            point_spacing = self.spacing / 2
            
        if direction =="right":
            angle_degrees = -angle_degrees
        angle_rad = np.deg2rad(angle_degrees)
        direction_vector = np.array([np.cos(angle_rad), np.sin(angle_rad), 0])
        up = np.array([0, 0, 1])
        side = np.cross(direction_vector, up)

        # Height = full cloud length
        height = np.linalg.norm(pcd.get_max_bound() - pcd.get_min_bound())
    
        # Width = spacing corrected for angle
        ortho_proj = np.abs(np.dot(side[:2], [1, 0])) + np.abs(np.dot(side[:2], [0, 1]))
        width = (point_spacing / ortho_proj) * 1.1
    
        # Project all points onto side vector to get range
        rel = points[:, :2] - center[:2]
        proj = rel @ side[:2]
        proj_min, proj_max = proj.min(), proj.max()
        total_span = proj_max - proj_min
        spacing = total_span / max(n - 1, 1)
    
        for i in range(n):
            offset_val = proj_min + i * spacing
            offset_vec = side * offset_val
            slice_center = center + offset_vec
    
            R = np.vstack([side, direction_vector, up]).T
            obb = o3d.geometry.OrientedBoundingBox(center=slice_center, R=R, extent=[width, height, 1.0])
            sliced_pcd = pcd.crop(obb)
            if len(sliced_pcd.points) == 0:
                print(f"Warning: slice {i+1} is empty, skipping.")
                continue
        
            mask = np.isin(np.round(points, 6), np.round(np.asarray(sliced_pcd.points), 6)).all(axis=1)
            indices = np.where(mask)[0].tolist()
            self.highlight_points(indices)
    
            print(f"Slice {i+1}/{n}: {len(sliced_pcd.points)} points at angle {angle_degrees}°")
    
            if angle_degrees == 0 or abs(angle_degrees) == 90:
                #clean each profile to remove parallel points and get a single smooth profile.
                sliced_pcd = self.filter_profile(sliced_pcd)

            if sliced_pcd.has_colors():
                sliced_colors = np.asarray(sliced_pcd.colors)
            # Sort points along the profile direction
            sliced_points = np.asarray(sliced_pcd.points)
            sliced_normals = np.asarray(sliced_pcd.normals)

            projections = sliced_points @ direction_vector  # project onto direction vector

            sort_indices = np.argsort(projections)
            if reverse:
                sort_indices = sort_indices[::-1]

            sorted_points = sliced_points[sort_indices]
            sorted_normals = sliced_normals[sort_indices]
            sliced_pcd.points = o3d.utility.Vector3dVector(sorted_points)
            sliced_pcd.normals = o3d.utility.Vector3dVector(sorted_normals)

            if sliced_pcd.has_colors():
                sorted_colors  = sliced_colors[sort_indices]
                sliced_pcd.colors = o3d.utility.Vector3dVector(sorted_colors)

            # Store profile
            self._add_profiles_by_index(sliced_pcd) #append to selected profiles dict
            #print(f"select_multiple_profiles_with_angle_old func. :  {self.selected_profiles_dict}\n\n")
    



    
    

    def is_sort_by_axis(self, pointcloud, transition_ratio: float = 0.25):
        """
        Determines whether to use sort_by_axis or sort_pcd.

        Args:
            pointcloud: Open3d pointcloud object
            transition_ratio (float): Ratio used for deciding between the two functions in uncertain cases

        Returns:
            boolean: True if sort_by_axis False if sort_pcd
        """
        std_devs = np.std(np.asarray(pointcloud.points),axis=0)

        if len(std_devs) != 3:
            raise ValueError("std_devs must contain 3 values for [x, y, z]")

        smallest, second, largest = sorted(std_devs)
        diff_largest_second = abs(largest - second)
        diff_second_smallest = abs(second - smallest)
        total = diff_largest_second + diff_second_smallest
        ratio = diff_second_smallest / total if total != 0 else 0.5
        return True if ratio < transition_ratio else False


    def sort_by_axis(self, original_PC, axis_str='x', reverse=False, resample=False):

        #had to shift axis because sorting has to be done around axis not along axis. So, around x means along y
        axis_idx = {'x': 1, 'y': 0, 'z': 2}[axis_str]  

        res_pts = np.array(original_PC.points)
        res_nor = np.array(original_PC.normals)
        #To create smoother profiles along a particular axis. this searches for least deviation in particular axis and replacesits value by its mean
        if resample: 
            min_idx = np.argmin(np.std(res_pts, axis=0))
            res_pts[:, min_idx] = np.mean(res_pts, axis=0)[min_idx]  #replace values of axis (x,y or z) #that has min std_dev by its mean.
            original_PC.points = o3d.utility.Vector3dVector(res_pts)
            original_PC = original_PC.voxel_down_sample(voxel_size=self.spacing)

        res_pts = np.array(original_PC.points)
        res_nor = np.array(original_PC.normals)

        ind = np.argsort(res_pts[:, axis_idx]) #Sort around world X axis
        if reverse:
            ind = ind[::-1]
        res_pts = res_pts[ind]
        res_nor = res_nor[ind]

        sorted_pointcloud = o3d.geometry.PointCloud()
        sorted_pointcloud.points = o3d.utility.Vector3dVector(res_pts)
        sorted_pointcloud.normals = o3d.utility.Vector3dVector(res_nor)
        return sorted_pointcloud
    
    
    def sort_pcd(self,pcd, reverse=False):
        points = np.asarray(pcd.points)

        #Fit plane using PCA
        centroid = points.mean(axis=0)
        centered = points - centroid
        U, S, Vt = np.linalg.svd(centered)
        normal = Vt[2]

        #Local 2D axes in the plane
        x_axis = Vt[0]
        y_axis = Vt[1]

        #Project points to local 2D plane
        x_coords = centered @ x_axis
        y_coords = centered @ y_axis

        # Step 4: Compute angles and sort
        angles = np.arctan2(y_coords, x_coords)
        sorted_indices = np.argsort(angles)
        if reverse:
            sorted_indices = sorted_indices[::-1]

        # Step 5: Create ordered point cloud
        sorted_pcd = o3d.geometry.PointCloud()
        sorted_pcd.points = o3d.utility.Vector3dVector(points[sorted_indices])

        #copy colors and normals
        if pcd.has_colors():
            sorted_colors = np.asarray(pcd.colors)[sorted_indices]
            sorted_pcd.colors = o3d.utility.Vector3dVector(sorted_colors)

        if pcd.has_normals():
            sorted_normals = np.asarray(pcd.normals)[sorted_indices]
            sorted_pcd.normals = o3d.utility.Vector3dVector(sorted_normals)

        return sorted_pcd
    
    def preview_points_order(self, pcds, delay=0.05):
        
        if not isinstance(pcds, list):
                pcds = [pcds]
        
        points = np.asarray(pcds[0].points)
        num_points = len(points)
        
        highlight = o3d.geometry.TriangleMesh.create_sphere(radius=0.010)
        highlight.paint_uniform_color([0, 0, 1])
        highlight.translate(points[0])  # Start at first point
    
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="Highlight Points On Keypress")
        vis.add_geometry(self.world_mesh)
        for pcda in pcds:
            vis.add_geometry(pcda)
        vis.add_geometry(highlight)
        
        quit_flag = {"quit": False}
    
        def quit_callback(vis):
            quit_flag["quit"] = True
            print("Quitting...")
            return False
    
        # Press X to quit
        vis.register_key_callback(ord("X"), quit_callback)
        try:
            while not quit_flag["quit"]:
                for pcd in pcds:
                    if quit_flag["quit"]:
                            break
                    points = np.asarray(pcd.points)
                    num_points = len(points)
                    for idx in range(num_points):
                        if quit_flag["quit"]:
                            break
                        else:
                            pt = points[idx]
                            highlight.translate(pt, relative=False)
                            vis.update_geometry(highlight)
                            vis.poll_events()
                            vis.update_renderer()
                            time.sleep(delay)
        finally:
            vis.destroy_window()
    
    def remove_duplicate_profiles(self, pcd_dict, tol=1e-6):
        filtered = []
        ctr = 0
        multi_pcd_list = list(self.selected_profiles_dict.values())  #convert dict to list
        for pcd_list in multi_pcd_list:
            unique = []
            for i, pcd in enumerate(pcd_list):
                pts_i = np.round(np.asarray(pcd.points), decimals=6)
                is_duplicate = False
                for u in unique:
                    pts_u = np.round(np.asarray(u.points), decimals=6)
                    if len(pts_i) == len(pts_u) and np.allclose(pts_i, pts_u, atol=tol):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique.append(pcd)
                    ctr +=1
            print(f"Filtered {len(pcd_list) - len(unique)} duplicates from {len(pcd_list)} profiles.")
            
            filtered.append(unique)
        return filtered, ctr

    def get_selected_profiles(self, reverse=False):
        profiles_list, prof_no = self.remove_duplicate_profiles(self.selected_profiles_dict)  #Remove duplicate profiles from the dict of pointclouds.
        print(f"Total number of profiles generated: {prof_no} from {len(profiles_list)} objects.")
        
        return [item for obj in profiles_list for item in obj] #flatten multi_list

    def run(self):
        try:
            while not self.should_quit[0]:
                self.vis.poll_events()
                self.vis.update_renderer()
                if self.animate[0]:
                    self.ctr.rotate(-2, 0)
                if self.dirty[0]:
                    self.update_view()
                time.sleep(0.001)
        finally:
            self.vis.destroy_window()
            selected_profiles = self.get_selected_profiles()
            
            self.result_holder.extend(selected_profiles)