// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/common.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/io/pcd_io.h>
#include <pcl_ros/transforms.h>
#include <sensor_msgs/PointCloud2.h>
#include <fast_lio/BoolStamped.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/TwistStamped.h>
#include <livox_ros_driver/CustomMsg.h>
#include "preprocess.h"
#include <name_pcd.h>
#include <ikd-Tree/ikd_Tree.h>
#include <common_lib.h>
#include <jsk_rviz_plugins/OverlayText.h>
#include "obb.h"
#include <swri_profiler/profiler.h>

#define INIT_TIME (0.1)
#define LASER_POINT_COV (0.001)
#define MAXN (720000)
#define PUBFRAME_PERIOD (20)

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool runtime_pos_log = false, pcd_save_en = true, time_sync_en = false, extrinsic_est_en = true, path_en = true;
/**************************/

float res_last[100000] = {0.0};
float DET_RANGE = 500.0f;
const float MOV_THRESHOLD = 2.5f;
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
bool point_selected_surf[100000] = {0};
bool lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool scan_pub_en = true, dense_pub_en = false, scan_body_pub_en = false;
string map_frame_name, body_frame_name;
double max_x, min_x, max_y, min_y, max_z, min_z;
int idle_start_l1_sec = 10; 
int idle_start_l2_sec = 30; 
int idle_pub_freq_sec = 5;
float speed_meter_s = 0;
//std::vector<float> front_lidar_2_base, rear_lidar_2_base;
std::vector<float> base_2_front_lidar, base_2_rear_lidar;
tf::Transform *trans_ptr;
tf::Transform tf_front_lidar_2_base;
tf::Transform tf_rear_lidar_2_base;
int using_lidar_dirct;
Eigen::Affine3f *transform_lidar_2_base_ptr;

vector<vector<int>> pointSearchInd_surf;
vector<BoxPointType> cub_needrm;
vector<PointVector> Nearest_Points;
vector<double> extrinT(3, 0.0);
vector<double> extrinR(9, 0.0);
vector<double> ext_front_base2lidar(6, 0.0);

deque<double> time_buffer;
deque<PointCloudXYZI::Ptr> lidar_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr _featsArray;

PointCloudXYZI::Ptr pcl_feature_full(new PointCloudXYZI());
PointCloudXYZI::Ptr pcl_feature_surf(new PointCloudXYZI());
PointCloudXYZI::Ptr pcl_feature_corn(new PointCloudXYZI());
//PointCloudXYZI::Ptr pcl_cuboid_8pts(new PointCloudXYZI());
pcl::PointCloud<pcl::PointXYZ>::Ptr  pcl_cuboid_8pts(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr  pcl_cuboid_8pts_lidar_frame(new pcl::PointCloud<pcl::PointXYZ>);
//PointCloudXYZI::Ptr pcl_cuboid_8pts_lidar_frame(new PointCloudXYZI());
pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_orig_ptr(new pcl::PointCloud<pcl::PointXYZI>());

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;
pcl::VoxelGrid<PointType> downSizeFilterForObb;

obb_struct obb_data;
float obb_freq_sec = 1;
float time_last_obb_sec = 0;
float time_curr_obb_sec = 0;

ros::Publisher pubMarkerText;
ros::Publisher pubOverlayText;

KD_TREE<PointType> ikdtree;

V3F XAxisPoint_body( LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lid;

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose;

FILE_PATH savePcdPath;
std::string prefixName;
std::string postfixName;
std::string folderString_default;

bool if_log_debug_print = false;
bool if_log_idle_print = true;
bool if_log_speed_print = true;
bool if_cropself = true;
bool if_init_time_sec = false;
bool if_moving = false;
bool enable_adaptive_filter_size = true;

bool if_start_idle_1 = false;
bool if_start_idle_2 = false;
bool if_start_idle_mins = false;

bool if_pub_idle_bool = true;
bool if_idle_published = false;
ros::Publisher pub_idle;

double time_init_sec;
double time_curr_sec;
double time_last_logtag_sec;
double time_last_idletag_sec;
double time_idle_start_sec;
double time_period_idle_sec;
double frequency_hz_log_speed = 3.0;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

std::vector<int> CropboxPolygonIndicesHandle(const livox_ros_driver::CustomMsg::ConstPtr &msg, Eigen::Affine3f * trans_aff_ptr)
{
    cloud_orig_ptr->clear();
    cloud_orig_ptr->points.resize(int(msg->point_num));
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_tran_ptr(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_tran_crop_ptr(new pcl::PointCloud<pcl::PointXYZI>());
    for (int i = 0; i < msg->point_num; i++)
    {
        cloud_orig_ptr->points[i].x = msg->points[i].x;
        cloud_orig_ptr->points[i].y = msg->points[i].y;
        cloud_orig_ptr->points[i].z = msg->points[i].z;
        cloud_orig_ptr->points[i].intensity= msg->points[i].reflectivity;
    }
    cloud_orig_ptr->width = cloud_orig_ptr->points.size();
    cloud_orig_ptr->height = 1;

    cloud_tran_ptr = cloud_orig_ptr;
    //pcl::io::savePCDFileASCII("/home/loopx/rosbag/no_crop.pcd", *cloud_orig_ptr);
    //std::cout<<"save no_crop.pcd"<<std::endl;
    /* frame_id lidar to base_link */
    //pcl::transformPointCloud (*cloud_orig_ptr, *cloud_tran_ptr, *trans_aff_ptr,false);
    //pcl_ros::transformPointCloud(*cloud_orig_ptr, *cloud_tran_ptr, trans);

    //pcl::io::savePCDFileASCII("/home/loopx/rosbag/no_crop_trans.pcd", *cloud_tran_ptr);
    //std::cout<<"save no_crop_trans.pcd"<<std::endl;
    /*crop self / crop a cuboid out */
    pcl::CropBox<pcl::PointXYZI> cropBoxFilter(true);
    cropBoxFilter.setInputCloud(cloud_orig_ptr);
    cropBoxFilter.setNegative(true);

    pcl::PointXYZ minPt, maxPt;
    pcl::getMinMax3D<pcl::PointXYZ> (*pcl_cuboid_8pts_lidar_frame, minPt, maxPt);
    Eigen::Vector4f min_pt(minPt.x, minPt.y, minPt.z, 1.0f);
    Eigen::Vector4f max_pt(maxPt.x, maxPt.y, maxPt.z, 1.0f);

    // Cropbox slighlty bigger then bounding box of points
    cropBoxFilter.setMin(min_pt);
    cropBoxFilter.setMax(max_pt);

    // Indices
    vector<int> indices;
    cropBoxFilter.filter(indices);
    // cropBoxFilter.filter(*cloud_tran_crop_ptr);
    // cropBoxFilter.getRemovedIndices(indices);
    //std::cout<<"save crop_trans.pcd"<<std::endl;
    //pcl::io::savePCDFileASCII("/home/loopx/rosbag/crop_trans.pcd", *cloud_tran_crop_ptr);
    return (indices);
}

void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

inline void dump_lio_state_to_log(FILE *fp)
{
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                            // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2));    // Pos
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                                 // omega
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2));    // Vel
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                                 // Acc
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));       // Bias_g
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));       // Bias_a
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a
    fprintf(fp, "\r\n");
    fflush(fp);
}

void pointBodyToWorld_ikfom(PointType const *const pi, PointType *const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void pointBodyToWorld(PointType const *const pi, PointType *const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template <typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const *const pi, PointType *const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const *const pi, PointType *const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I * p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    SWRI_PROFILE("lasermap_fov_segment");
    cub_needrm.clear();
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiD = pos_lid;
    if (!Localmap_Initialized)
    {
        for (int i = 0; i < 3; i++)
        {
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++)
    {
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
            need_move = true;
    }
    if (!need_move)
        return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD - 1)));
    for (int i = 0; i < 3; i++)
    {
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
        else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if (cub_needrm.size() > 0)
        kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

/*
void base_2_lidar_br(const sensor_msgs::PointCloud2::ConstPtr &msg_in,
                           sensor_msgs::PointCloud2::ConstPtr &cloud2_out){
    static tf::TransformBroadcaster br;
    tf::Transform transform(tf::createQuaternionFromRPY(ext_front_base2lidar[3],ext_front_base2lidar[4],ext_front_base2lidar[5]),
                                            tf::Vector3(ext_front_base2lidar[0],ext_front_base2lidar[1],ext_front_base2lidar[2]));
    pcl_ros::transformPointCloud(body_frame_name,transform, *msg_in, *cloud2_out);
    br.sendTransform(tf::StampedTransform(transform, msg_in->header.stamp, body_frame_name, msg_in->header.frame_id));
}*/

void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg_in)
{
    //std::cout << "standard_pcl_cbk" << std::endl;

    static tf::TransformBroadcaster br;
    
    tf::Transform transform(tf::createQuaternionFromRPY(ext_front_base2lidar[3],ext_front_base2lidar[4],ext_front_base2lidar[5]),
                                            tf::Vector3(ext_front_base2lidar[0],ext_front_base2lidar[1],ext_front_base2lidar[2]));
    //br.sendTransform(tf::StampedTransform(transform, msg_in->header.stamp, body_frame_name, msg_in->header.frame_id));
    
    sensor_msgs::PointCloud2::Ptr msg(new sensor_msgs::PointCloud2());
    pcl_ros::transformPointCloud(body_frame_name,transform, *msg_in,*msg);

    if (!if_init_time_sec){
        time_init_sec = msg->header.stamp.toSec();
        if_init_time_sec = true;
        time_last_logtag_sec = time_init_sec;
        time_idle_start_sec = time_init_sec;
        time_last_obb_sec = time_init_sec;
    }
    time_curr_sec = msg->header.stamp.toSec();
    mtx_buffer.lock();
    scan_count++;
    double preprocess_start_time = omp_get_wtime();
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    //std::vector<int> index_cropself = CropboxPolygonIndicesHandle(msg, transform_lidar_2_base_ptr);
    p_pre->process(msg, pcl_feature_full, pcl_feature_surf, pcl_feature_corn);
    sensor_msgs::PointCloud2 ros_pc_msg_feature_full;
    sensor_msgs::PointCloud2 ros_pc_msg_feature_surf;
    sensor_msgs::PointCloud2 ros_pc_msg_feature_corn;

    pcl::toROSMsg(*pcl_feature_full, ros_pc_msg_feature_full);
    pcl::toROSMsg(*pcl_feature_surf, ros_pc_msg_feature_surf);
    pcl::toROSMsg(*pcl_feature_corn, ros_pc_msg_feature_corn);
    
    std::cout<< "pcl_surface_full.points.size()!= "<< pcl_feature_full->points.size()<<std::endl;
    std::cout<< "pcl_feature_surf.points.size()!= "<< pcl_feature_surf->points.size()<<std::endl;
    std::cout<< "pcl_feature_corn.points.size()!= "<< pcl_feature_corn->points.size()<<std::endl;
    //lidar_buffer.push_back(pcl_feature_full);
    lidar_buffer.push_back(pcl_feature_full);
    time_buffer.push_back(msg->header.stamp.toSec());
    last_timestamp_lidar = msg->header.stamp.toSec();
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool timediff_set_flg = false;

void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg){
    if (!if_init_time_sec){
        time_init_sec = msg->header.stamp.toSec();
        if_init_time_sec = true;
        time_last_logtag_sec = time_init_sec;
        time_idle_start_sec = time_init_sec;
    }
    time_curr_sec = msg->header.stamp.toSec();
    mtx_buffer.lock();
    double preprocess_start_time = omp_get_wtime();
    scan_count++;
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec();

    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty())
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n", last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    sensor_msgs::PointCloud2 ros_pc_msg_feature_full;
    sensor_msgs::PointCloud2 ros_pc_msg_feature_surf;
    sensor_msgs::PointCloud2 ros_pc_msg_feature_corn;

    std::vector<int> index_cropself = CropboxPolygonIndicesHandle(msg, transform_lidar_2_base_ptr);
    p_pre->process(msg, index_cropself,pcl_feature_full, pcl_feature_surf, pcl_feature_corn);
    //pcl::transformPointCloud (*pcl_feature_full, *pcl_feature_full, *transform_lidar_2_base_ptr,false);
    //pcl::transformPointCloud (*pcl_feature_surf, *pcl_feature_surf, *transform_lidar_2_base_ptr,false);
    //pcl::transformPointCloud (*pcl_feature_corn, *pcl_feature_corn, *transform_lidar_2_base_ptr,false);
    
    pcl::toROSMsg(*pcl_feature_full, ros_pc_msg_feature_full);
    pcl::toROSMsg(*pcl_feature_surf, ros_pc_msg_feature_surf);
    pcl::toROSMsg(*pcl_feature_corn, ros_pc_msg_feature_corn);

    //lidar_buffer.push_back(pcl_feature_surf);
    lidar_buffer.push_back(pcl_feature_full);
    time_buffer.push_back(last_timestamp_lidar);
    static int count_laserin = 0;
    // std::cout<<count_laserin++<<" pcl_feature_surf_pointin.size() = "<<pcl_feature_surf->points.size()<<std::endl;

    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in)
{
    publish_count++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() - time_diff_lidar_to_imu);
    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp =
            ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }

    double timestamp = msg->header.stamp.toSec();

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double lidar_mean_scantime = 0.0;
int scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{
    //std::cout<< "lidar_buffer.empty() = " <<lidar_buffer.empty()<< ", imu_buffer.empty() = "<< imu_buffer.empty()<<std::endl;
   if (lidar_buffer.empty() || imu_buffer.empty())
    {
        return false;
    }

    /*** push a lidar scan ***/
    if (!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();
        if (meas.lidar->points.size() <= 1) // time too little
        { 
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            ROS_WARN("Too few input point cloud!\n");
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }
    
    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if (imu_time > lidar_end_time)
            break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

int process_increments = 0;
void map_incremental()
{
    SWRI_PROFILE("map_incremental()");
    
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    //std::cout<<"feats_down_size = " <<feats_down_size<<std::endl;
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point;
            mid_point.x = floor(feats_down_world->points[i].x / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            float dist = calc_dist(feats_down_world->points[i], mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min)
            {
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++)
            {
                if (points_near.size() < NUM_MATCH_POINTS)
                    break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add)
                PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false);
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
}

void publish_cloud(const ros::Publisher &pub, PointCloudXYZI::Ptr pcl_in, std::string frame_id)
{
    sensor_msgs::PointCloud2 ros_pc2_msg;
    pcl::toROSMsg(*pcl_in, ros_pc2_msg);
    ros_pc2_msg.header.stamp = ros::Time().fromSec(lidar_end_time);
    ros_pc2_msg.header.frame_id = frame_id;
    pub.publish(ros_pc2_msg);
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());

void defaultPathName()
{
    savePcdPath.filepath = "~/bags/pcd";
    savePcdPath.filepath = folderString_default;
    savePcdPath.extension = "pcd";
}

void updateTimeName()
{
    time_t rawtime;
    rawtime = time(NULL);
    struct tm *timeinfo;
    char buffer[127];
    timeinfo = localtime(&rawtime);

    strftime(buffer, 127, "%G%m%d_%H%M%SScan", timeinfo);
    savePcdPath.filename = std::string(buffer);
    if(if_log_debug_print){
        ROS_INFO_STREAM("[Mapping][pcd name] timeString = " << buffer);
    }
}

void updateTimeName(ros::Time ros_time_in)
{
    //time_t ros_time = ros::Time().fromSec(ros_time_in);
    time_t ros_time = ros_time_in.toSec();
    struct tm *timeinfo;
    char buffer[127];
    timeinfo = localtime(&ros_time);
    strftime(buffer, 127, "%G%m%d_%H%M%SScan", timeinfo);
    savePcdPath.filename = std::string(buffer);
    if(if_log_debug_print){
        ROS_INFO_STREAM("[Mapping][pcd name] timeString = " << buffer);
    }
}

void publish_frame_world(const ros::Publisher &pubLaserCloudFull)
{
    if (scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(
            new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i],
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = map_frame_name;
        pubLaserCloudFull.publish(laserCloudmsg);
        //std::cout<<"laserCloudWorld = "<< laserCloudWorld->points.size()<<std::endl;
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en)
    {
        defaultPathName();
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i],
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval)
        {
            updateTimeName(ros::Time().fromSec(lidar_end_time));
            pcd_index++;
            try {
                pcl::io::savePCDFileASCII(fullFileName(savePcdPath), *pcl_wait_save);
                if (if_log_debug_print){
                    ROS_INFO_STREAM( "[Mapping][pcd path]current scan saved to " << fullFileName(savePcdPath)<<"." );
                }
            }catch(std::exception &e){
                ROS_WARN_STREAM("[Mapping][pcd path] CAN NOT save pcd file to "<< savePcdPath.filename);
                ROS_WARN_STREAM("[Mapping][pcd path] save current scan to default folder" << folderString_default<<"/"<< savePcdPath.filename<<"."<<savePcdPath.extension);
                pcl::io::savePCDFileASCII( folderString_default+ "/" +  FileNameExt(savePcdPath), *pcl_wait_save);
            }
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}

void publish_effect_world(const ros::Publisher &pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld(
        new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i],
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudFullRes3.header.frame_id = map_frame_name;
    pubLaserCloudEffect.publish(laserCloudFullRes3);
}
/*
void publish_map(const ros::Publisher &pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = map_frame_name;
    pubLaserCloudMap.publish(laserCloudMap);
}*/

template <typename T>
void set_posestamp(T &out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
}

template <typename T>
void set_twiststamp(T &out)
{
    out.twist.linear.x = state_point.vel(0);
    out.twist.linear.y = state_point.vel(1);
    out.twist.linear.z = state_point.vel(2);
    out.twist.angular.x = Measures.imu.back()->angular_velocity.x;
    out.twist.angular.y = Measures.imu.back()->angular_velocity.y;
    out.twist.angular.z = Measures.imu.back()->angular_velocity.z;
}

void publish_odometry(const ros::Publisher &pubOdomAftMapped, const ros::Publisher &pubTwist)
{
    odomAftMapped.header.frame_id = map_frame_name;
    odomAftMapped.child_frame_id = body_frame_name;
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time); // // ros::Time::now();
    set_posestamp(odomAftMapped.pose);
    set_twiststamp(odomAftMapped.twist);
    pubOdomAftMapped.publish(odomAftMapped);
    auto P = kf.get_P();
    for (int i = 0; i < 6; i++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i * 6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i * 6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i * 6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i * 6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i * 6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i * 6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x,
                                    odomAftMapped.pose.pose.position.y,
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation(q);
    //br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, map_frame_name, body_frame_name));
    br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), map_frame_name, body_frame_name));
    // Publish Twist on base_link frame
    geometry_msgs::TwistStamped twist;
    twist.header.stamp = odomAftMapped.header.stamp;
    twist.header.frame_id = "base_link"; // body_frame_name;
    twist.twist.linear.x = sqrt(odomAftMapped.twist.twist.linear.x * odomAftMapped.twist.twist.linear.x + odomAftMapped.twist.twist.linear.y * odomAftMapped.twist.twist.linear.y);

    tf::Matrix3x3 rotation(transform.getRotation());
    double roll, pitch, yaw;
    rotation.getRPY(roll, pitch, yaw);

    // Note: When the velocity(map frame) vector angle is within ±90° of the body angle: moving forward.
    double vel_vector_angle = atan2(odomAftMapped.twist.twist.linear.y, odomAftMapped.twist.twist.linear.x);
    double diff = abs(vel_vector_angle - yaw); // yaw: -PI ~ PI, vel_vector_angle: -PI ~ PI

    static int body_direction = 1;                                                  // 1: forward, -1: backward
    if (diff > M_PI / 2 && diff < 3 * M_PI / 2 && abs(twist.twist.linear.x) > 0.05){ // At lower speeds, the estimate may not be accurate, use last direction
        body_direction = -1;
    }else{
        body_direction = 1;
    }
    twist.twist.linear.x *= body_direction;
    twist.twist.angular.z = -odomAftMapped.twist.twist.angular.z;

    if (if_log_debug_print){
        ROS_INFO_STREAM("[Mapping][Twist]yaw(degree):" << yaw * 180.0 / M_PI << " vel_vector_angle: " << vel_vector_angle << " diff: " << diff << " twist: " << twist.twist.linear.x);
    }
    speed_meter_s = sqrt(twist.twist.linear.x * twist.twist.linear.x +
                               twist.twist.linear.y * twist.twist.linear.y +
                               twist.twist.linear.z * twist.twist.linear.z);

    //std::cout<< time_curr_sec <<"," << time_last_logtag_sec <<"," << frequency_hz_log_speed <<std::endl;
    //std::cout <<"diff * freq =" <<(time_curr_sec - time_last_logtag_sec) * frequency_hz_log_speed <<std::endl;
    if (if_log_speed_print){
        if ((time_curr_sec - time_last_logtag_sec) * frequency_hz_log_speed > 1.0){
            ROS_INFO_STREAM("[Mapping][Speed]" <<std::setw(5)<<std::fixed<<std::setprecision(2)<<speed_meter_s << "m/s, "
                                               <<std::setw(5)<<std::fixed<<std::setprecision(2)<<speed_meter_s * 3.6 << "km/h");
            time_last_logtag_sec = time_curr_sec;
        }
    }

    if (if_log_idle_print){
        if(speed_meter_s < 0.8){
            if (if_moving == 1){
                if_moving = false;
                time_idle_start_sec = time_curr_sec;
            }else{
                time_period_idle_sec = time_curr_sec - time_idle_start_sec;
            }
        }else if (if_moving == false){
            if_moving = true;
            if(time_period_idle_sec > 30){
               ROS_INFO_STREAM("[Mapping][idle_end]"<< "idle time is over, idle time = "<< time_period_idle_sec <<"sec" );
               if_start_idle_1 = false;
               if_start_idle_2 = false;
               if_start_idle_mins = false;
            }
            time_period_idle_sec = 0; 
        }
    }
    
    if (if_pub_idle_bool){
        fast_lio::BoolStamped if_idle_msg;
        if_idle_msg.header.stamp =  ros::Time().fromSec(lidar_end_time);
        if_idle_msg.header.frame_id = map_frame_name;
        if(time_period_idle_sec > idle_start_l1_sec){
            if_idle_msg.bool_data = true;
        }else{         
            if_idle_msg.bool_data = false ;
        }
    
        if(int(time_curr_sec - time_init_sec)%idle_pub_freq_sec==0 && !if_idle_published ){
            pub_idle.publish(if_idle_msg);
            if_idle_published = true;
        }else if (int(time_curr_sec - time_init_sec)%idle_pub_freq_sec!=0 && if_idle_published){
            if_idle_published = false;
        }
    }

    if(int (time_period_idle_sec) == idle_start_l1_sec && !if_start_idle_1){
        ROS_INFO_STREAM("[Mapping][idle_start]"<< "idle time = "<<std::fixed<<std::setprecision(2)<< time_period_idle_sec <<"sec" );
        if_start_idle_1 = true;
    }
    if(int (time_period_idle_sec) == idle_start_l2_sec && !if_start_idle_2){
        ROS_INFO_STREAM("[Mapping][idle_2l]"<< "idle time = "<<std::fixed<<std::setprecision(2)<< time_period_idle_sec <<"sec" );
        if_start_idle_2 = true;
    }
    if(int (time_period_idle_sec > 60 && int(time_period_idle_sec)%60==0 && if_start_idle_mins)){
        ROS_INFO_STREAM("[Mapping][idle_mins]"<< "idle time = "<<std::fixed<<std::setprecision(2)<< time_period_idle_sec/60 <<" Minutes" );
        if_start_idle_mins = false;
    }

    // tf::Vector3 twist_rot(odomAftMapped.twist.twist.angular.x,
    //                       odomAftMapped.twist.twist.angular.y,
    //                       odomAftMapped.twist.twist.angular.z);
    // tf::Vector3 twist_vel(odomAftMapped.twist.twist.linear.x,
    //                       odomAftMapped.twist.twist.linear.y,
    //                       odomAftMapped.twist.twist.linear.z);

    // tf::Transform inverseTransform = transform.inverse();
    // tf::Vector3 out_rot = inverseTransform.getBasis() * twist_rot;
    // tf::Vector3 out_vel = inverseTransform.getBasis()* twist_vel + inverseTransform.getOrigin().cross(out_rot);

    // Temporary use: exit the program when the speed exceeds the threshold
    if (abs(twist.twist.linear.x) > 8.3|| abs(twist.twist.linear.y) > 8.3)
    {
        ROS_ERROR("Speed exceeds the threshold, exit the program!");
        exit(0);
    }
    pubTwist.publish(twist);
}

void pubJskOverlay(geometry_msgs::PoseStamped pose_in)
{
    std::ostringstream tostr;
    if(if_moving){
    // variable for text view facing marker
    /*std::cout<< std::setfill(' ')
          << std::setw(8) << "x:" <<std::setw(8) << std::fixed << std::setprecision(2) << pose_in.pose.position.x << "\n"
          << std::setw(8) << "y:" <<std::setw(8) << std::fixed << std::setprecision(2) << pose_in.pose.position.y << "\n"
          << std::setw(8) << "z:"  <<std::setw(8) << std::fixed << std::setprecision(2) << pose_in.pose.position.z << "\n"
          << std::setw(8) << "Speed:" << std::right <<std::setw(8) << std::fixed << std::setprecision(2) << speed_meter_s * 3.6 << "km/h" << std::endl;*/
          tostr << std::setfill(' ')
          << std::setw(8) << "x:" <<std::setw(8) << std::fixed << std::setprecision(2) << pose_in.pose.position.x <<"\n"
          << std::setw(8) << "y:" <<std::setw(8) << std::fixed << std::setprecision(2) << pose_in.pose.position.y <<"\n"
          << std::setw(8) << "z:"  <<std::setw(8) << std::fixed << std::setprecision(2) << pose_in.pose.position.z <<"\n"
          << std::setw(8) << "Speed:" << std::right <<std::setw(8) << std::fixed << std::setprecision(2) << speed_meter_s * 3.6 <<" km/h" << std::endl;
    }else if (time_period_idle_sec>idle_start_l1_sec){
        if(time_period_idle_sec<60){
          tostr << std::setfill(' ')
          << std::setw(8) << "idle time:"<<std::setw(8) << std::fixed << std::setprecision(0) <<time_period_idle_sec<<" sec"<<"\n";
        }
        if(time_period_idle_sec>60 && time_period_idle_sec<120){
             tostr << std::setfill(' ')
             << std::setw(8) << "idle time: \n" <<std::setw(8) << std::fixed << std::setprecision(0) <<" 1 min & "<< int(time_period_idle_sec)%60<<" sec"<<"\n";
        }
        if(time_period_idle_sec>=120 && time_period_idle_sec < 36000){
            tostr << std::setfill(' ')
             << std::setw(8) << "idle time: \n" <<std::setw(8) << std::fixed << std::setprecision(0) << time_period_idle_sec/60 <<" mins & "<< int(time_period_idle_sec)%60<<" sec"<<"\n";
        }
        if(time_period_idle_sec>=3600 && time_period_idle_sec < 86400){
            tostr << std::setfill(' ')
             << std::setw(8) << "idle time: \n" <<std::setw(8) << std::fixed << std::setprecision(0) <<time_period_idle_sec/3600<<" hr & "<<int(int(time_period_idle_sec)%3600)%60<<"  min & "<< int(time_period_idle_sec)%60<<" sec"<<"\n";
        }
    }

    std::string txt_o = tostr.str();
    //std::cout<< "pubJskOverlay :"<<txt_o<<std::endl;
    jsk_rviz_plugins::OverlayText overlay_txt;
    overlay_txt.text   = txt_o;
    overlay_txt.width  = 400;
    overlay_txt.height = 600;
    overlay_txt.left   = 100;
    overlay_txt.top    = 100;
    overlay_txt.text_size = 20;
    overlay_txt.line_width = 3;
    overlay_txt.font = "DejaVu Sans Mono";
 
    std_msgs::ColorRGBA fg; fg.a = 236/255; fg.b = 245/255; fg.g = 39/255.0; fg.r=0.8;
    overlay_txt.fg_color = fg;
    std_msgs::ColorRGBA bg; fg.a = 0.0;     fg.b = 0.0;     fg.g = 0.0;      fg.r=0.2;
    overlay_txt.bg_color = bg;
    pubOverlayText.publish(overlay_txt);
}

void pubMarker(geometry_msgs::PoseStamped pose_in)
{
    std::ostringstream tostr;
    // variable for text view facing marker
    tostr << "x:" << pose_in.pose.position.x << ", y:" << pose_in.pose.position.y << ", z:" << pose_in.pose.position.z << std::endl;
    visualization_msgs::Marker text_view;
    text_view.text = tostr.str();

    // marker type , scale,colour
    text_view.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    text_view.action = visualization_msgs::Marker::ADD;
    text_view.header.stamp = ros::Time().fromSec(lidar_end_time);
    text_view.header.frame_id = map_frame_name;
    text_view.scale.z = 0.30;
    // fill text coordinates
    text_view.pose.position.x = pose_in.pose.position.x;
    text_view.pose.position.y = pose_in.pose.position.y;
    text_view.pose.position.z = pose_in.pose.position.z;

    text_view.color.r = 0.0f;
    text_view.color.g = 1.0f;
    text_view.color.b = 0.0f;
    text_view.color.a = 1.0f;
    pubMarkerText.publish(text_view);

/*
    visualization_msgs::Marker model_marker;
    model_marker.type = visualization_msgs::Marker::MESH_RESOURCE;
    model_marker.mesh_resource = "package://fast_lio/meshes/r1600.dae";
    model_marker.header.stamp = ros::Time().fromSec(lidar_end_time);
    model_marker.header.frame_id = body_frame_name;

    pubMarkerText.publish(model_marker);
    */
}

void publish_path(const ros::Publisher pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = map_frame_name;

    /*** if path is too large, the rvis will crash ***/
    static int count_path = 0;
    count_path++;
    if (count_path % 10 == 0)
    {
        path.poses.push_back(msg_body_pose);
        pubPath.publish(path);
        pubMarker(msg_body_pose);
        pubJskOverlay(msg_body_pose);
    }
}

void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{

    SWRI_PROFILE("h_share_model");
    double match_start = omp_get_wtime();
    laserCloudOri->clear();
    corr_normvect->clear();
    total_residual = 0.0;

/** closest surface search and residual computation **/
#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body = feats_down_body->points[i];
        PointType &point_world = feats_down_world->points[i];

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false
                                                                                                                                : true;
        }

        if (!point_selected_surf[i])
            continue;

        VF(4)
        pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = abs(pd2);
            }
        }
    }

    effct_feat_num = 0;

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i];
            effct_feat_num++;
        }
    }

    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        ROS_WARN("[Mapping][h_share_model]No Effective Points! \n");
        return;
    }

    res_mean_last = total_residual / effct_feat_num;
    match_time += omp_get_wtime() - match_start;
    double solve_start_ = omp_get_wtime();

    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); // 23
    ekfom_data.h.resize(effct_feat_num);

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat << SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() * norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); // s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;
    }
    solve_time += omp_get_wtime() - solve_start_;
}
void set_self_cuboid(){
    pcl_cuboid_8pts->resize(8);
    pcl_cuboid_8pts->points[0].x=min_x;
    pcl_cuboid_8pts->points[0].y=min_y;
    pcl_cuboid_8pts->points[0].z=min_z;

    pcl_cuboid_8pts->points[1].x=min_x;
    pcl_cuboid_8pts->points[1].y=max_y;
    pcl_cuboid_8pts->points[1].z=min_z;

    pcl_cuboid_8pts->points[2].x=min_x;
    pcl_cuboid_8pts->points[2].y=max_y;
    pcl_cuboid_8pts->points[2].z=max_z;

    pcl_cuboid_8pts->points[3].x=min_x;
    pcl_cuboid_8pts->points[3].y=min_y;
    pcl_cuboid_8pts->points[3].z=max_z;

    pcl_cuboid_8pts->points[4].x=max_x;
    pcl_cuboid_8pts->points[4].y=min_y;
    pcl_cuboid_8pts->points[4].z=min_z;

    pcl_cuboid_8pts->points[5].x=max_x;
    pcl_cuboid_8pts->points[5].y=max_y;
    pcl_cuboid_8pts->points[5].z=min_z;

    pcl_cuboid_8pts->points[6].x=max_x;
    pcl_cuboid_8pts->points[6].y=min_y;
    pcl_cuboid_8pts->points[6].z=max_z;

    pcl_cuboid_8pts->points[7].x=max_x;
    pcl_cuboid_8pts->points[7].y=max_y;
    pcl_cuboid_8pts->points[7].z=max_z;
    //pcl::io::savePCDFileASCII("/home/loopx/rosbag/box_self.pcd", *pcl_cuboid_8pts);
}

void set_nocrop(){
   max_x =  0.01; max_y =  0.01; max_z =  0.01;
   min_x = -0.01; min_y = -0.01; min_z = -0.01;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;
    //ros::NodeHandle pnh("~"); // private node handle

    nh.param<string>("publish/map_frame_name", map_frame_name, "map");
    nh.param<string>("publish/body_frame_name", body_frame_name, "base_link");
    nh.param<bool>  ("publish/path_en", path_en, true);
    nh.param<bool>  ("publish/scan_publish_en", scan_pub_en, true);
    nh.param<bool>  ("publish/dense_publish_en", dense_pub_en, true);
    nh.param<bool>  ("publish/scan_bodyframe_pub_en", scan_body_pub_en, true);

    nh.param<int>   ("max_iteration", NUM_MAX_ITERATIONS, 4);
    nh.param<string>("map_file_path", map_file_path, "");
    nh.param<double>("filter_size_corner", filter_size_corner_min, 0.5);
    nh.param<double>("filter_size_surf", filter_size_surf_min, 0.5);
    nh.param<double>("filter_size_map", filter_size_map_min, 0.5);
    nh.param<double>("cube_side_length", cube_len, 200);

    nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");
    nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");
    nh.param<bool>  ("common/time_sync_en", time_sync_en, false);
    nh.param<double>("common/time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);

    nh.param<float> ("mapping/det_range", DET_RANGE, 500.f);
    nh.param<double>("mapping/fov_degree", fov_deg, 180);
    nh.param<double>("mapping/gyr_cov", gyr_cov, 0.1);
    nh.param<double>("mapping/acc_cov", acc_cov, 0.1);
    nh.param<double>("mapping/b_gyr_cov", b_gyr_cov, 0.0001);
    nh.param<double>("mapping/b_acc_cov", b_acc_cov, 0.0001);
    nh.param<bool>  ("mapping/extrinsic_est_en", extrinsic_est_en, true);
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());
    nh.param<vector<double>>("sensorkit/front/ext_base2lidar", ext_front_base2lidar, vector<double>());

    nh.param<double>("preprocess/blind", p_pre->blind, 0.1);
    nh.param<int>   ("preprocess/lidar_type", p_pre->lidar_type, AVIA);
    nh.param<int>   ("preprocess/scan_line", p_pre->N_SCANS, 16);
    nh.param<int>   ("preprocess/timestamp_unit", p_pre->time_unit, US);
    nh.param<int>   ("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    nh.param<bool>  ("preprocess/if_cropself", if_cropself, true);
    ROS_INFO_STREAM("[Mapping] if_cropself = " << if_cropself);

    nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);
    nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, false);
    nh.param<bool>("runtime_pos_log_enable", runtime_pos_log, 0);

    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, true);
    ROS_INFO_STREAM("[Mapping] pcd_save_en = " << pcd_save_en);
    nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
    nh.param<std::string>("pcd_save/folderString", folderString_default, "/home/loopx/rosbag/pcd");
    ROS_INFO_STREAM("[Mapping] folderString_default = " << folderString_default);

    nh.getParam("front_lidar_base2lidar", base_2_front_lidar);
    nh.getParam("rear_lidar_base2lidar", base_2_rear_lidar);
    //nh.getParam("front_lidar_base2lidar", front_lidar_2_base);
    //nh.getParam("rear_lidar_base2lidar", rear_lidar_2_base);
    nh.getParam("using_lidar_dirct", using_lidar_dirct);

    nh.param<bool>("log/if_log_debug_print", if_log_debug_print, false);
    ROS_INFO_STREAM("[Mapping] if_log_debug_print = " << if_log_debug_print);
    p_pre->set_if_log_debug_print(if_log_debug_print);
    nh.param<bool>("log/if_log_speed_print", if_log_speed_print, true);
    ROS_INFO_STREAM("[Mapping] if_log_speed_print = " << if_log_speed_print);
    nh.param<bool>("log/if_log_idle_print", if_log_idle_print, true);
    ROS_INFO_STREAM("[Mapping] if_log_idle_print = " << if_log_idle_print);
    nh.param<double>("log/frequency_hz_log_speed", frequency_hz_log_speed, 3);

    nh.param<bool>("if_pub_idle_bool", if_pub_idle_bool, false);
    nh.param<int>("idle_start_l1_sec", idle_start_l1_sec, 10);
    nh.param<int>("idle_start_l2_sec", idle_start_l2_sec, 30);
    nh.param<int>("idle_pub_freq_sec", idle_pub_freq_sec, 10);
    nh.param<bool>("enable_adaptive_filter_size", enable_adaptive_filter_size, true);

    nh.getParam("max_x", max_x);
    nh.getParam("min_x", min_x);
    nh.getParam("max_y", max_y);
    nh.getParam("min_y", min_y);
    nh.getParam("max_z", max_z);
    nh.getParam("min_z", min_z);
    if(!if_cropself){
       set_nocrop();
    }
    set_self_cuboid();

    if (if_log_debug_print){
        ROS_INFO_STREAM("[Mapping] Crop self box polygon : max_x = " << max_x << ", min_x = " << min_x << ", max_y = " << max_y << ", min_y = " << min_y << ", max_z = " << max_z << ", min_z = " << min_z);
        ROS_INFO_STREAM("[Mapping] tf_front_lidar_2_base : x = " << base_2_front_lidar[0] << ", y = " << base_2_front_lidar[1] << ", z = " << base_2_front_lidar[2] << ", roll = " << base_2_front_lidar[3] << ", pitch = " << base_2_front_lidar[4] << ", yaw = " << base_2_front_lidar[5] << ".");
        ROS_INFO_STREAM("[Mapping] tf_rear_lidar_2_base : x = " << base_2_rear_lidar[0] << ", y = " << base_2_rear_lidar[1] << ", z = " << base_2_rear_lidar[2] << ", roll = " << base_2_rear_lidar[3] << ", pitch = " << base_2_rear_lidar[4] << ", yaw = " << base_2_rear_lidar[5] << ".");
    }

    Eigen::Affine3f transform_front_lidar_2_base =  Eigen::Affine3f::Identity();
    transform_front_lidar_2_base.rotate(Eigen::AngleAxisf(base_2_front_lidar[3], Eigen::Vector3f::UnitX())
                                      * Eigen::AngleAxisf(base_2_front_lidar[4], Eigen::Vector3f::UnitY())
                                      * Eigen::AngleAxisf(base_2_front_lidar[5], Eigen::Vector3f::UnitZ()));
    transform_front_lidar_2_base.translation() << base_2_front_lidar[0], base_2_front_lidar[1] ,base_2_front_lidar[2]+min_z;
    //std::cout<<"transform_front_lidar_2_base"<<std::endl;
    //std::cout<<transform_front_lidar_2_base.matrix()<<std::endl;

    Eigen::Affine3f transform_rear_lidar_2_base =  Eigen::Affine3f::Identity();
    transform_rear_lidar_2_base.rotate(Eigen::AngleAxisf(base_2_rear_lidar[3], Eigen::Vector3f::UnitX())
                                     * Eigen::AngleAxisf(base_2_rear_lidar[4], Eigen::Vector3f::UnitY())
                                     * Eigen::AngleAxisf(base_2_rear_lidar[5], Eigen::Vector3f::UnitZ()));
    transform_rear_lidar_2_base.translation() << base_2_rear_lidar[0], base_2_rear_lidar[1] ,base_2_rear_lidar[2]+min_z;

    ROS_INFO_STREAM("[Mapping] using_lidar_dirct = "<<using_lidar_dirct<<" ");
    switch (using_lidar_dirct){
    case FRONT:
         pcl::transformPointCloud (*pcl_cuboid_8pts, *pcl_cuboid_8pts_lidar_frame, transform_front_lidar_2_base.inverse(),false);
         std::cout<<"transform_front_lidar_2_base.inverse()"<<std::endl;
         std::cout<<transform_front_lidar_2_base.inverse().matrix()<<std::endl;
         //transform_lidar_2_base_ptr = &transform_front_lidar_2_base;
         break;
    case REAR:
         pcl::transformPointCloud (*pcl_cuboid_8pts, *pcl_cuboid_8pts_lidar_frame, transform_rear_lidar_2_base.inverse(),false);
         std::cout<<"transform_rear_lidar_2_base.inverse()"<<std::endl;
         std::cout<<transform_rear_lidar_2_base.inverse().matrix()<<std::endl;
         //transform_lidar_2_base_ptr = &transform_rear_lidar_2_base;
         break;
    case LEFT:
         break;
    case RIGHT:
         break;
    case FRONT_REAR:
         break;
    default:
         pcl::transformPointCloud (*pcl_cuboid_8pts, *pcl_cuboid_8pts_lidar_frame, transform_front_lidar_2_base.inverse(),false);
         //transform_lidar_2_base_ptr = &transform_front_lidar_2_base;
         break;
    }
    // pcl::io::savePCDFileASCII("/home/loopx/rosbag/trans_8pts.pcd", *pcl_cuboid_8pts_lidar_frame);
    
    cout << "p_pre->lidar_type " << p_pre->lidar_type << endl;
    cout << "p_pre->blind " << p_pre->blind << endl;
    cout << "lid_topic = " << lid_topic << endl;
    
    path.header.stamp = ros::Time::now();
    path.header.frame_id = map_frame_name;

    /*** variables definition ***/
    int effect_feat_num = 0, frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;

    FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
    HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

    _featsArray.reset(new PointCloudXYZI());

    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));

    Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

    double epsi[23] = {0.001};
    fill(epsi, epsi + 23, 0.001);
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

    /*** debug record ***/
    FILE *fp;
    string pos_log_dir = root_dir + "/Log/pos_log.txt";
    fp = fopen(pos_log_dir.c_str(), "w");

    ofstream fout_pre, fout_out, fout_dbg;
    fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"), ios::out);
    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"), ios::out);
    fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"), ios::out);
    if (fout_pre && fout_out)
        cout << "~~~~" << ROOT_DIR << " file opened" << endl;
    else
        cout << "~~~~" << ROOT_DIR << " doesn't exist" << endl;

    /*** ROS subscribe initialization ***/
    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100000);

    ros::Publisher pub_full = nh.advertise<sensor_msgs::PointCloud2>("/cloud_feature_full", 100000);
    ros::Publisher pub_surf = nh.advertise<sensor_msgs::PointCloud2>("/cloud_feature_surf", 100000);
    ros::Publisher pub_corn = nh.advertise<sensor_msgs::PointCloud2>("/cloud_feature_corn", 100000);
    ros::Publisher pub_orig = nh.advertise<sensor_msgs::PointCloud2>("/cloud_orig", 100000);
    ros::Publisher pub_cuboid_8pts =  nh.advertise<sensor_msgs::PointCloud2>("/self_cuboid", 1);
    pub_idle = nh.advertise<fast_lio::BoolStamped>("/idle_bool", 1);
    
    // ros::Publisher pubLaserFeaturePoints = nh.advertise<sensor_msgs::PointCloud2>
    //         ("/Laser_feature_points", 100000);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
    ros::Publisher pubPath = nh.advertise<nav_msgs::Path>("/path", 100000);
    // publish twist
    ros::Publisher pubTwist = nh.advertise<geometry_msgs::TwistStamped>("/twist", 100000);
    pubMarkerText = nh.advertise<visualization_msgs::Marker>("visualization_markers_text", 5);
    pubOverlayText = nh.advertise<jsk_rviz_plugins::OverlayText>("jsk_text",5);
    //------------------------------------------------------------------------------------------------------
    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    bool status = ros::ok();
    static int count_while = 0;
    while (status)
    {
        count_while++;
        if (flg_exit)
            break;
        ros::spinOnce();
        //std::cout<<"sync_packages(Measures) = "<<sync_packages(Measures)<<"~~"<<std::endl;
        if (sync_packages(Measures))
        {
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                continue;
            }

            double t0, t1, t2, t3, t4, t5, match_start, solve_start, svd_time;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            svd_time = 0;
            t0 = omp_get_wtime();

            p_imu->Process(Measures, kf, feats_undistort);
            /*feats_undistort = pcl_feature_full;
            updateTimeName(ros::Time().fromSec(lidar_end_time));
            if(feats_undistort->points.size()>0){
                feats_undistort->height =1;
                feats_undistort->width  =feats_undistort->points.size() ;
                try {
                    pcl::io::savePCDFileASCII(fullFileName(savePcdPath), *feats_undistort);
                    ROS_INFO_STREAM( "[Mapping][feats_undistort]current scan saved to " << fullFileName(savePcdPath)<<"." );
                }catch(std::exception &e){
                    pcl::io::savePCDFileASCII( folderString_default+ "/" +  FileNameExt(savePcdPath), *feats_undistort);
                }
            }*/
            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                ROS_WARN("No point, skip this scan!!\n");
                continue;
            }

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;
            /*** Segment the map in lidar FOV ***/
            lasermap_fov_segment();

            /***** Adaptive filter size*****/
            if(enable_adaptive_filter_size && 
               ros::Time().fromSec(lidar_end_time).toSec()-time_last_obb_sec > obb_freq_sec){

               downSizeFilterForObb.setLeafSize(1.5, 1.5, 1.5);
               downSizeFilterForObb.setInputCloud(feats_undistort);
               downSizeFilterForObb.filter(*feats_down_body);
               calulate_obb(feats_down_body,&obb_data);

               if (obb_data.length_x*obb_data.width_y*obb_data.hight_z>10000){
                   filter_size_surf_min = 0.6;
                   filter_size_map_min =0.6;
                   p_pre->point_filter_num = 3;
               }else{
                   filter_size_surf_min = 0.3;
                   filter_size_map_min = 0.3;
                   p_pre->point_filter_num = 2;
               }
               time_last_obb_sec = ros::Time().fromSec(lidar_end_time).toSec()-time_last_obb_sec;
            }

            downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
            downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);

            /*** downsample the feature points in a scan ***/
            //std::cout<<obb_data.length_x<<"*"<<obb_data.width_y<<"*"<<obb_data.hight_z<<", "<<filter_size_surf_min<<std::endl;
            //downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
            downSizeFilterSurf.setInputCloud(feats_undistort);
            //downSizeFilterSurf.setInputCloud(pcl_feature_full);
            downSizeFilterSurf.filter(*feats_down_body);
            t1 = omp_get_wtime();
            feats_down_size = feats_down_body->points.size();
            /*** initialize the map kdtree ***/
            if (ikdtree.Root_Node == nullptr)
            {
                if (feats_down_size > 5)
                {
                    ikdtree.set_downsample_param(filter_size_map_min);
                    feats_down_world->resize(feats_down_size);
                    for (int i = 0; i < feats_down_size; i++)
                    {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                    }
                    ikdtree.Build(feats_down_world->points);
                }
                continue;
            }
            int featsFromMapNum = ikdtree.validnum();
            kdtree_size_st = ikdtree.size();
            //std::cout<<"ikdtree.size() = "<<ikdtree.size()<<std::endl;
            //cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < 5)
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            /* sensor_msgs::PointCloud2 laserFeatureMsg;
               pcl::toROSMsg(*feats_down_world, laserFeatureMsg);
               laserFeatureMsg.header.frame_id = map_frame_name;
               pubLaserFeaturePoints.publish(laserFeatureMsg);  */

            V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
            fout_pre << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose() << " " << ext_euler.transpose() << " " << state_point.offset_T_L_I.transpose() << " " << state_point.vel.transpose()
                     << " " << state_point.bg.transpose() << " " << state_point.ba.transpose() << " " << state_point.grav << endl;

            if (0) // If you need to see map point, change to "if(1)"
            {
                PointVector().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
            }

            pointSearchInd_surf.resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);
            int rematch_num = 0;
            bool nearest_search_en = true; //

            t2 = omp_get_wtime();

            /*** iterated state estimation ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
            state_point = kf.get_x();
            euler_cur = SO3ToEuler(state_point.rot);
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];

            double t_update_end = omp_get_wtime();

            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped, pubTwist);

            /*** add the feature points to map kdtree ***/
            t3 = omp_get_wtime();
            map_incremental();
            t5 = omp_get_wtime();

            /******* Publish points *******/
            // std::cout<<"path_en = "<<path_en<<", scan_pub_en = "<<scan_pub_en<<", pcd_save_en = "<<pcd_save_en <<", scan_body_pub_en = " <<scan_body_pub_en<<std::endl;
            if (path_en)
                publish_path(pubPath);
            if (scan_pub_en || pcd_save_en)
                publish_frame_world(pubLaserCloudFull);
            if (scan_pub_en && scan_body_pub_en)
                //publish_frame_body(pubLaserCloudFull_body);
                publish_cloud(pub_full, pcl_feature_full, body_frame_name);
                publish_cloud(pub_corn, pcl_feature_corn, body_frame_name);
                publish_cloud(pub_surf, pcl_feature_surf, body_frame_name);

            PointCloudXYZI::Ptr pcl_cuboid_8pts_fit(new PointCloudXYZI());
            pcl_cuboid_8pts_fit->points.resize(8);
            if (pub_cuboid_8pts.getNumSubscribers()!= 0){
                for(int i=0; i<int(pcl_cuboid_8pts->points.size());i++ ){
                   pcl_cuboid_8pts_fit->points[i].x= pcl_cuboid_8pts->points[i].x + min_x;
                   pcl_cuboid_8pts_fit->points[i].y= pcl_cuboid_8pts->points[i].y;
                   pcl_cuboid_8pts_fit->points[i].z= pcl_cuboid_8pts->points[i].z;
                }
                //publish_cloud(pub_cuboid_8pts, pcl_cuboid_8pts_fit, body_frame_name);
            }
            if (pub_orig.getNumSubscribers()!= 0){
                sensor_msgs::PointCloud2 ros_pc2_msg;
                pcl::toROSMsg(*cloud_orig_ptr, ros_pc2_msg);
                ros_pc2_msg.header.stamp = ros::Time().fromSec(lidar_end_time);
                ros_pc2_msg.header.frame_id = body_frame_name;
                pub_orig.publish(ros_pc2_msg);
            }
            // publish_effect_world(pubLaserCloudEffect);
            // publish_map(pubLaserCloudMap);

            /*** Debug variables ***/
            if (runtime_pos_log)
            {
                frame_num++;
                kdtree_size_end = ikdtree.size();
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                aver_time_icp = aver_time_icp * (frame_num - 1) / frame_num + (t_update_end - t_update_start) / frame_num;
                aver_time_match = aver_time_match * (frame_num - 1) / frame_num + (match_time) / frame_num;
                aver_time_incre = aver_time_incre * (frame_num - 1) / frame_num + (kdtree_incremental_time) / frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1) / frame_num + (solve_time + solve_H_time) / frame_num;
                aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1) / frame_num + solve_time / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = t5 - t0;
                s_plot2[time_log_counter] = feats_undistort->points.size();
                s_plot3[time_log_counter] = kdtree_incremental_time;
                s_plot4[time_log_counter] = kdtree_search_time;
                s_plot5[time_log_counter] = kdtree_delete_counter;
                s_plot6[time_log_counter] = kdtree_delete_time;
                s_plot7[time_log_counter] = kdtree_size_st;
                s_plot8[time_log_counter] = kdtree_size_end;
                s_plot9[time_log_counter] = aver_time_consu;
                s_plot10[time_log_counter] = add_point_size;
                time_log_counter++;
                
                /*printf("[mapping]time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n", t1 - t0, aver_time_match, aver_time_solve, t3 - t1, t5 - t3, aver_time_consu, aver_time_icp, aver_time_const_H_time);
                ext_euler = SO3ToEuler(state_point.offset_R_L_I);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose() << " " << ext_euler.transpose() << " " << state_point.offset_T_L_I.transpose() << " " << state_point.vel.transpose()
                         << " " << state_point.bg.transpose() << " " << state_point.ba.transpose() << " " << state_point.grav << " " << feats_undistort->points.size() << endl;
                */
                dump_lio_state_to_log(fp);
            }
        }

        status = ros::ok();
        rate.sleep();
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    std::cout<<"pcl_wait_save->size() = " <<pcl_wait_save->size()<<std::endl; 
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        if(if_log_debug_print){
           ROS_INFO_STREAM("current scan saved to /PCD/" << file_name);
        }
        //cout << "current scan saved to /PCD/" << file_name << endl;
        try {
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
        }catch(std::exception &e){
            std::cout<<" Warning!!!!! "<< e.what()<<std::endl;
        }
    }

    fout_out.close();
    fout_pre.close();

    if (runtime_pos_log)
    {
        vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;
        FILE *fp2;
        string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
        fp2 = fopen(log_dir.c_str(), "w");
        fprintf(fp2, "time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
        for (int i = 0; i < time_log_counter; i++)
        {
            fprintf(fp2, "%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n", T1[i], s_plot[i], int(s_plot2[i]), s_plot3[i], s_plot4[i], int(s_plot5[i]), s_plot6[i], int(s_plot7[i]), int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
            t.push_back(T1[i]);
            s_vec.push_back(s_plot9[i]);
            s_vec2.push_back(s_plot3[i] + s_plot6[i]);
            s_vec3.push_back(s_plot4[i]);
            s_vec5.push_back(s_plot[i]);
        }
        fclose(fp2);
    }

    return 0;
}
