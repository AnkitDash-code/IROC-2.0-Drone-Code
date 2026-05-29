import os
import open3d as o3d
import numpy as np
import time
import datetime
from ctypes import POINTER, c_ushort, byref, cast, sizeof, c_long

from SynexensPythonSDK import (
        InitSDK, UnInitSDK, FindDevice, OpenDevice, CloseDevice,
        SYDeviceInfo, SYErrorCodeEnum, SYStreamTypeEnum, SYFrameTypeEnum, SYResolutionEnum,
        SYFrameData, SYPointCloudData, GetLastFrameData, GetDepthPointCloud,
        SetFrameResolution, StartStreaming, StopStreaming, PrintErrorCode
)

# --- Global variables to hold SDK state ---
_synexens_device_id = -1
_synexens_sdk_initialized = False


# --- Lidar Lifecycle Management Functions ---
def initialize_lidar_stream() -> bool:
    global _synexens_device_id, _synexens_sdk_initialized

    if _synexens_sdk_initialized:
        return True

    print("Lidar Initializing...")
    errorCodeInitSDK = InitSDK()
    if errorCodeInitSDK != SYErrorCodeEnum.SYERRORCODE_SUCCESS:
        return False

    nDeviceCount = c_long()
    pDeviceInfo = (SYDeviceInfo * 1)()

    errorCodeFindDevice = FindDevice(byref(nDeviceCount), None)
    if errorCodeFindDevice == SYErrorCodeEnum.SYERRORCODE_SUCCESS and nDeviceCount.value > 0:
        errorCodeFindDevice = FindDevice(byref(nDeviceCount), pDeviceInfo)
        if errorCodeFindDevice == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
            _synexens_device_id = pDeviceInfo[0].m_nDeviceID
        else:
            UnInitSDK()
            return False
    else:
        UnInitSDK()
        return False

    errorCodeOpenDevice = OpenDevice(pDeviceInfo[0])
    if errorCodeOpenDevice != SYErrorCodeEnum.SYERRORCODE_SUCCESS:
        UnInitSDK()
        return False

    resolution_enum = SYResolutionEnum.SYRESOLUTION_640_480

    errorCodeSetResolution = SetFrameResolution(_synexens_device_id, SYFrameTypeEnum.SYFRAMETYPE_DEPTH, resolution_enum)
    if errorCodeSetResolution != SYErrorCodeEnum.SYERRORCODE_SUCCESS:
        CloseDevice(_synexens_device_id)
        UnInitSDK()
        return False

    stream_type = SYStreamTypeEnum.SYSTREAMTYPE_DEPTH  # Changed from SYSTREAMTYPE_DEPTHIR
    print("Lidar Stream: Starting streaming...")
    errorCodeStartStreaming = StartStreaming(_synexens_device_id, stream_type)
    if errorCodeStartStreaming != SYErrorCodeEnum.SYERRORCODE_SUCCESS:
        PrintErrorCode("StartStreaming", errorCodeStartStreaming)
        CloseDevice(_synexens_device_id)
        UnInitSDK()
        return False

    time.sleep(2) 
    _synexens_sdk_initialized = True
    print("Lidar Initialized and streaming started successfully.")
    return True

def stop_lidar_stream():
    global _synexens_device_id, _synexens_sdk_initialized
    if _synexens_sdk_initialized and _synexens_device_id != -1:
        StopStreaming(_synexens_device_id)
        CloseDevice(_synexens_device_id)
        UnInitSDK()
        _synexens_device_id = -1
        _synexens_sdk_initialized = False
        print("Lidar Uninitialized successfully.")
    elif _synexens_sdk_initialized:
        print("Lidar was initialized but device might not have been fully open. Uninitializing SDK.")
        UnInitSDK()
        _synexens_sdk_initialized = False
    else:
        print("Lidar not initialized")


def get_pcd_from_stream() -> o3d.geometry.PointCloud | None:
    global _synexens_device_id, _synexens_sdk_initialized

    if not _synexens_sdk_initialized or _synexens_device_id == -1:
        print("Error: Lidar stream not active")
        print("Reinitializing lidar...")
        if initialize_lidar_stream():
            print("Lidar reinitialized successfully, continuing...")
        else:
            print("Failed to reinitialize lidar, returning None")
            return None

    max_retries = 50
    retry_delay_sec = 0.05 
    depth_frame_info = None
    pDepth = None

    for retries in range(max_retries):
        pFrameData = POINTER(SYFrameData)()
        errorCodeLastFrame = GetLastFrameData(_synexens_device_id, byref(pFrameData))

        if errorCodeLastFrame == SYErrorCodeEnum.SYERRORCODE_SUCCESS and pFrameData and pFrameData.contents:
            objFrameData = pFrameData.contents
            
            if objFrameData.m_nFrameCount > 0:
                frame_info = objFrameData.m_pFrameInfo[0]  
                nCount = frame_info.m_nFrameHeight * frame_info.m_nFrameWidth               
                pDepth = cast(objFrameData.m_pData, POINTER(c_ushort))
                depth_frame_info = frame_info
                break
        
        time.sleep(retry_delay_sec)

    if not (depth_frame_info and pDepth):
        print("Lidar failed to acquire valid depth frame after multiple retries.")
        return None

    nFrameWidth = depth_frame_info.m_nFrameWidth
    nFrameHeight = depth_frame_info.m_nFrameHeight
    nCount = nFrameWidth * nFrameHeight

    if not pDepth:
        print("Lidar pDepth is null after frame acquisition attempt.")
        return None

    LP_SYPointCloudData = POINTER(SYPointCloudData)
    data_array = (SYPointCloudData * nCount)()
    pPCLData = cast(data_array, LP_SYPointCloudData)

    errorCodePointCloud = GetDepthPointCloud(_synexens_device_id, nFrameWidth, nFrameHeight, pDepth, pPCLData, True) 

    if errorCodePointCloud != SYErrorCodeEnum.SYERRORCODE_SUCCESS:
        del pPCLData
        del data_array
        return None
    
    points_np = np.empty((nCount, 3), dtype=np.float32)
    for i in range(nCount):
        points_np[i, 0] = data_array[i].m_fltX
        points_np[i, 1] = data_array[i].m_fltY
        points_np[i, 2] = data_array[i].m_fltZ

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    
    del pPCLData 
    del data_array 

    return pcd

def process_pcd_for_angles(pcd_object: o3d.geometry.PointCloud, n_planes=2, visualisation=False) -> tuple[float, float] | None:
    if not pcd_object.has_points() or len(pcd_object.points) == 0:
        print("Processing: Input PCD is empty.")
        return None

    voxel_size_mm = 10 
    pcd_downsampled = pcd_object.voxel_down_sample(voxel_size=voxel_size_mm)

    pcd_cleaned, _ = pcd_downsampled.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    radius_for_outlier_removal = 50 
    pcd_cleaned, _ = pcd_cleaned.remove_radius_outlier(nb_points=16, radius=radius_for_outlier_removal)

    min_points_for_any_processing = 500 
    if not pcd_cleaned.has_points() or len(pcd_cleaned.points) < min_points_for_any_processing:
        print(f"Warning: Cleaned PCD has too few points ({len(pcd_cleaned.points)}) for plane detection. Skipping.")
        if visualisation:
            o3d.visualization.draw_geometries([pcd_cleaned], window_name="Too few points after cleaning")
        return None

    detected_angles = []
    max_planes_to_find = n_planes
    min_points_for_ransac_segmentation = 10000 

    visual_inlier_clouds = []

    for i in range(max_planes_to_find):
        if len(pcd_cleaned.points) < min_points_for_ransac_segmentation:
            break

        distance_threshold = 100 
        ransac_n = 3
        num_iterations = 20000

        plane_model, inliers = pcd_cleaned.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )

        if len(inliers) < (min_points_for_ransac_segmentation / 5): 
            break

        [a, b, c, d] = plane_model
        plane_normal = np.array([a, b, c])
        plane_normal_normalized = plane_normal / np.linalg.norm(plane_normal)
        reference_vector = np.array([0, 0, 1])
        dot_product = np.clip(np.dot(plane_normal_normalized, reference_vector), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(abs(dot_product))) # Use abs for 0-90 degrees range

        detected_angles.append(angle_deg)
        

        inlier_cloud = pcd_cleaned.select_by_index(inliers)

        inlier_cloud.paint_uniform_color([1, 0, 0] if i == 0 else [0, 0, 1]) # Red for first, Blue for second
        visual_inlier_clouds.append(inlier_cloud)


        pcd_cleaned = pcd_cleaned.select_by_index(inliers, invert=True)

    if visualisation:
        print("Processing: Displaying visualization...")
        visual_geometries = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=500, origin=[0, 0, 0])]
        
        visual_geometries.extend(visual_inlier_clouds)

        if pcd_cleaned.has_points():
            pcd_cleaned.paint_uniform_color([0.6, 0.6, 0.6]) # Grey for leftovers
            visual_geometries.append(pcd_cleaned)

        o3d.visualization.draw_geometries(
            visual_geometries,
            window_name="Detected Planes",
            width=1024, height=768,
            left=50, top=50,
            point_show_normal=False
        )
        print("Processing: Visualization window closed.")

    del pcd_downsampled
    del pcd_cleaned
    for cloud in visual_inlier_clouds: 
        del cloud

    if len(detected_angles) >= 2:
        return (detected_angles[0], detected_angles[1])
    elif len(detected_angles) == 1:
        return (detected_angles[0], None)
    else:
        return None

def get_current_plane_angles(visualize_frame: bool = False) -> tuple[float, float] | None:
    pcd_frame = None
    try:
        pcd_frame = get_pcd_from_stream()
        if pcd_frame is None:
            print("Failed to get PCD frame from stream.")
            return None

        angles = process_pcd_for_angles(pcd_frame, visualisation=visualize_frame)
        
        return angles

    except Exception as e:
        print(f"An error occurred in get_current_plane_angles: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if pcd_frame:
            del pcd_frame 


if __name__ == "__main__":
    initialize_lidar_stream()
    time.sleep(3)
    start = time.perf_counter()
    angles = get_current_plane_angles(visualize_frame=False)
    duration = time.perf_counter() - start
    print("Angles:", angles)
    print("Time taken:", duration, "seconds")
    stop_lidar_stream()
