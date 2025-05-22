#ifndef _OBB_H_
#define _OBB_H_

#include <vector>
#include <cmath>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

struct obb_struct {
    pcl::PointXYZ position;
    Eigen::Matrix3f rotational_matrix;
    pcl::PointCloud<pcl::PointXYZ>::Ptr obb_cuboid{new pcl::PointCloud<pcl::PointXYZ>};

    float length_x;
    float width_y;
    float hight_z;
};

void set_aabb_cuboid(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, 
                         float max_x, float min_x, 
                         float max_y, float min_y, 
                         float max_z, float min_z){
        cloud_in->resize(9);
        cloud_in->points[0].x=min_x;
        cloud_in->points[0].y=min_y;
        cloud_in->points[0].z=min_z;
    
        cloud_in->points[1].x=min_x;
        cloud_in->points[1].y=max_y;
        cloud_in->points[1].z=min_z;
    
        cloud_in->points[2].x=min_x;
        cloud_in->points[2].y=max_y;
        cloud_in->points[2].z=max_z;
    
        cloud_in->points[3].x=min_x;
        cloud_in->points[3].y=min_y;
        cloud_in->points[3].z=max_z;
    
        cloud_in->points[4].x=max_x;
        cloud_in->points[4].y=min_y;
        cloud_in->points[4].z=min_z;
    
        cloud_in->points[5].x=max_x;
        cloud_in->points[5].y=max_y;
        cloud_in->points[5].z=min_z;
    
        cloud_in->points[6].x=max_x;
        cloud_in->points[6].y=min_y;
        cloud_in->points[6].z=max_z;
    
        cloud_in->points[7].x=max_x;
        cloud_in->points[7].y=max_y;
        cloud_in->points[7].z=max_z;

        cloud_in->points[8].x=0;
        cloud_in->points[8].y=0;
        cloud_in->points[8].z=0;

        cloud_in->width=1;
        cloud_in->height=cloud_in->points.size();
       // pcl::io::savePCDFileASCII("aabb_box.pcd", *cloud_in);
    }

void set_obb_cuboid(obb_struct *obb_info, 
                       pcl::PointXYZ min_point_OBB, 
                       pcl::PointXYZ max_point_OBB,
                       pcl::PointXYZ position_OBB,
                       Eigen::Matrix3f rotational_matrix_OBB){
       obb_info->obb_cuboid->clear();
       obb_info->obb_cuboid->resize(9);
       Eigen::Vector3f pos (position_OBB.x,position_OBB.y,position_OBB.z);
       Eigen::Vector3f p1 (min_point_OBB.x, min_point_OBB.y, min_point_OBB.z);
       Eigen::Vector3f p2 (min_point_OBB.x, min_point_OBB.y, max_point_OBB.z);
       Eigen::Vector3f p3 (max_point_OBB.x, min_point_OBB.y, max_point_OBB.z);
       Eigen::Vector3f p4 (max_point_OBB.x, min_point_OBB.y, min_point_OBB.z);
       Eigen::Vector3f p5 (min_point_OBB.x, max_point_OBB.y, min_point_OBB.z);
       Eigen::Vector3f p6 (min_point_OBB.x, max_point_OBB.y, max_point_OBB.z);
       Eigen::Vector3f p7 (max_point_OBB.x, max_point_OBB.y, max_point_OBB.z);
       Eigen::Vector3f p8 (max_point_OBB.x, max_point_OBB.y, min_point_OBB.z);

       p1 = rotational_matrix_OBB * p1 + pos;
       p2 = rotational_matrix_OBB * p2 + pos;
       p3 = rotational_matrix_OBB * p3 + pos;
       p4 = rotational_matrix_OBB * p4 + pos;
       p5 = rotational_matrix_OBB * p5 + pos;
       p6 = rotational_matrix_OBB * p6 + pos;
       p7 = rotational_matrix_OBB * p7 + pos;
       p8 = rotational_matrix_OBB * p8 + pos;
       
       obb_info->obb_cuboid->points[0].x = p1(0);
       obb_info->obb_cuboid->points[0].y = p1(1);
       obb_info->obb_cuboid->points[0].z = p1(2);

       obb_info->obb_cuboid->points[1].x = p2(0);
       obb_info->obb_cuboid->points[1].y = p2(1);
       obb_info->obb_cuboid->points[1].z = p2(2);

       obb_info->obb_cuboid->points[2].x = p3(0);
       obb_info->obb_cuboid->points[2].y = p3(1);
       obb_info->obb_cuboid->points[2].z = p3(2);

       obb_info->obb_cuboid->points[3].x = p4(0);
       obb_info->obb_cuboid->points[3].y = p4(1);
       obb_info->obb_cuboid->points[3].z = p4(2);
       
       obb_info->obb_cuboid->points[4].x = p5(0);
       obb_info->obb_cuboid->points[4].y = p5(1);
       obb_info->obb_cuboid->points[4].z = p5(2);

       obb_info->obb_cuboid->points[5].x = p6(0);
       obb_info->obb_cuboid->points[5].y = p6(1);
       obb_info->obb_cuboid->points[5].z = p6(2);
    
       obb_info->obb_cuboid->points[6].x = p7(0);
       obb_info->obb_cuboid->points[6].y = p7(1);
       obb_info->obb_cuboid->points[6].z = p7(2);

       obb_info->obb_cuboid->points[7].x = p8(0);
       obb_info->obb_cuboid->points[7].y = p8(1);
       obb_info->obb_cuboid->points[7].z = p8(2);

       obb_info->obb_cuboid->points[8] = position_OBB;

       obb_info->hight_z =
       sqrt(std::pow(obb_info->obb_cuboid->points[0].x - obb_info->obb_cuboid->points[1].x,2))
     + sqrt(std::pow(obb_info->obb_cuboid->points[0].y - obb_info->obb_cuboid->points[1].y,2))
     + sqrt(std::pow(obb_info->obb_cuboid->points[0].z - obb_info->obb_cuboid->points[1].z,2));

       obb_info->length_x = 
       sqrt(std::pow(obb_info->obb_cuboid->points[2].x - obb_info->obb_cuboid->points[1].x,2))
     + sqrt(std::pow(obb_info->obb_cuboid->points[2].y - obb_info->obb_cuboid->points[1].y,2))
     + sqrt(std::pow(obb_info->obb_cuboid->points[2].z - obb_info->obb_cuboid->points[1].z,2));

       obb_info->width_y = 
       sqrt(std::pow(obb_info->obb_cuboid->points[0].x - obb_info->obb_cuboid->points[4].x,2)) 
     + sqrt(std::pow(obb_info->obb_cuboid->points[0].y - obb_info->obb_cuboid->points[4].y,2))
     + sqrt(std::pow(obb_info->obb_cuboid->points[0].z - obb_info->obb_cuboid->points[4].z,2));

       obb_info->obb_cuboid->width=1;
       obb_info->obb_cuboid->height=obb_info->obb_cuboid->points.size();
       //pcl::io::savePCDFileASCII("obb_box.pcd", *obb_info->obb_cuboid);
  }

template <typename T>
void calulate_obb(T &cloud_in_, obb_struct *obb_info){

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointXYZ pt;
    for(int i=0; i<cloud_in_->points.size();i++){
        pt.x = cloud_in_->points[i].x;
        pt.y = cloud_in_->points[i].y;
        pt.z = cloud_in_->points[i].z;
        cloud_xyz->points.push_back(pt);

    }
    // 创建惯性矩估算对象，设置输入点云，并进行计算
    pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;
    feature_extractor.setInputCloud (cloud_xyz);
    feature_extractor.compute();

    std::vector <float> moment_of_inertia;
    std::vector <float> eccentricity;
    //pcl::PointXYZ min_point_AABB;
    //pcl::PointXYZ max_point_AABB;
    pcl::PointXYZ min_point_OBB;
    pcl::PointXYZ max_point_OBB;
    pcl::PointXYZ position_OBB;
    Eigen::Matrix3f rotational_matrix_OBB;
    float major_value, middle_value, minor_value;
    Eigen::Vector3f major_vector, middle_vector, minor_vector;
    Eigen::Vector3f mass_center;

    // 获取惯性矩
    feature_extractor.getMomentOfInertia (moment_of_inertia);
    // 获取离心率
    feature_extractor.getEccentricity (eccentricity);
    // 获取AABB盒子
    // feature_extractor.getAABB (min_point_AABB, max_point_AABB);
    // 获取OBB盒子
    feature_extractor.getOBB (min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);
    feature_extractor.getEigenValues (major_value, middle_value, minor_value);
    // 获取主轴major_vector，中轴middle_vector，辅助轴minor_vector
    feature_extractor.getEigenVectors (major_vector, middle_vector, minor_vector);
    // 获取质心
    feature_extractor.getMassCenter (mass_center);

    //pcl::PointCloud<pcl::PointXYZ>::Ptr AABB_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    //set_aabb_cuboid(AABB_cloud,max_point_AABB.x,min_point_AABB.x,
    //                           max_point_AABB.y,min_point_AABB.y,
    //                           max_point_AABB.z,min_point_AABB.z);

    pcl::PointCloud<pcl::PointXYZ>::Ptr OBB_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    set_obb_cuboid(obb_info,min_point_OBB, max_point_OBB,position_OBB,rotational_matrix_OBB);
}

#endif