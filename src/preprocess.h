#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <livox_ros_driver/CustomMsg.h>

using namespace std;

#define IS_VALID(a)  ((abs(a)>1e8) ? true : false)

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

enum LID_TYPE{AVIA = 1, VELO16, OUST64}; //{1, 2, 3}
enum TIME_UNIT{SEC = 0, MS = 1, US = 2, NS = 3};
enum Feature{Normal_Point, // normal points  //Nor
             Poss_Plane,   // point is possible in plane
             Real_Plane,   // definetly the point in a plane
             Edge_Jump,    // the point jumps over
             Edge_Plane,   // the point on jumps over plane
             Wire,         // small section wire
             ZeroPoint};
enum Surround{Prev, Next};

enum E_jump{Nr_nor,    //normal
            Nr_zero,   //0
            Nr_180,    //180
            Nr_inf,    //infinit
            Nr_blind}; //blind area

enum LID_DIRCT{FRONT = 1, REAR, LEFT, RIGHT, FRONT_REAR}; //{1, 2, 3, 4}

struct orgtype
{
  // Lidar origin : O, Previous point: P, Current point: C, Next point: N

  //          p     C      N
  //          .     .      .
  //           \    |     /
  //            \   |    /
  //             \  |   /
  //              \ |  /
  //               \| /
  //                . 
  //                O  lidar     
  double dist_2_xy_plane;  //point to xy plane lidar origin //range
  double dist_2_next_pt;   //distance from this point the next one. //dista
  double angle[2];   //cos(∠OCP), cos(∠OCN)
  double intersect;  //cos(∠PCN)
  E_jump etype_adjacent_pts[2];     //edge feature type of Previous point and next point  //edj      
  Feature ftype;
  orgtype()
  {
    dist_2_xy_plane = 0;
    etype_adjacent_pts[Prev] = Nr_nor;
    etype_adjacent_pts[Next] = Nr_nor;
    ftype = Normal_Point;
    intersect = 2;
  }
};

namespace velodyne_ros {
  struct EIGEN_ALIGN16 Point {
      PCL_ADD_POINT4D;
      float intensity;
      float time;
      uint16_t ring;
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
}  // namespace velodyne_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_ros::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (float, time, time)
    (uint16_t, ring, ring)
)

namespace ouster_ros {
  struct EIGEN_ALIGN16 Point {
      PCL_ADD_POINT4D;
      float intensity;
      uint32_t t;
      uint16_t reflectivity;
      uint8_t  ring;
      uint16_t ambient;
      uint32_t range;
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
}  // namespace ouster_ros

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(ouster_ros::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    // use std::uint32_t to avoid conflicting with pcl::uint32_t
    (std::uint32_t, t, t)
    (std::uint16_t, reflectivity, reflectivity)
    (std::uint8_t, ring, ring)
    (std::uint16_t, ambient, ambient)
    (std::uint32_t, range, range)
)

class Preprocess
{
  public:
//   EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Preprocess();
  ~Preprocess();
  
  void process(const livox_ros_driver::CustomMsg::ConstPtr &msg,
                     vector<int> indices, 
                     PointCloudXYZI::Ptr &pcl_out);
  void process(const livox_ros_driver::CustomMsg::ConstPtr &msg, 
                     vector<int> indices,
                     PointCloudXYZI::Ptr &pcl_full_out,
                     PointCloudXYZI::Ptr &pcl_surf_out,
                     PointCloudXYZI::Ptr &pcl_corn_out);
  void process(const sensor_msgs::PointCloud2::ConstPtr &msg,
                     PointCloudXYZI::Ptr &pcl_full_out,
                     PointCloudXYZI::Ptr &pcl_surf_out,
                     PointCloudXYZI::Ptr &pcl_corn_out);
  void set(bool feat_en, int lid_type, double bld, int pfilt_num);
  void set_if_log_debug_print(bool if_in);

  // sensor_msgs::PointCloud2::ConstPtr pointcloud;
  PointCloudXYZI pl_full, pl_corn, pl_surf;
  PointCloudXYZI pl_buff[128]; //maximum 128 line lidar
  vector<orgtype> typess[128]; //maximum 128 line lidar
  float time_unit_scale;
  int lidar_type, point_filter_num, N_SCANS, SCAN_RATE, time_unit;
  double blind;
  bool feature_enabled, given_offset_time;
  //ros::Publisher pub_full, pub_surf, pub_corn;

  private:
  void avia_handler(const livox_ros_driver::CustomMsg::ConstPtr &msg, std::vector<int> indices);
  void oust64_handler(const sensor_msgs::PointCloud2::ConstPtr &msg);
  void velodyne_handler(const sensor_msgs::PointCloud2::ConstPtr &msg);
  void give_feature(PointCloudXYZI &pl, vector<orgtype> &types);
  void pub_func(PointCloudXYZI &pl, ros::Publisher publisher, ros::Time ct);
  int  plane_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i, uint &i_nex, Eigen::Vector3d &curr_direct);
  bool small_plane(const PointCloudXYZI &pl, vector<orgtype> &types, uint i_cur, uint &i_nex, Eigen::Vector3d &curr_direct);
  bool edge_jump_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i, Surround nor_dir);
  
  int group_size;
  double inf_bound;   //if distance > inf_bound, then invalid
  double disA, disB ; //threshold for determing if the point in a plane
  double limit_maxmid, limit_midmin, limit_maxmin; // mid point to left, mid point to right, left to right
  double pt_2_line_ratio;  //threshold > distance from point to line,  then it is a plane //p2l_ratio
  double jump_up_limit, jump_down_limit;
  double cos160;
  double edgea, edgeb;   //point to point  // distance between two pts > edgeb // block
  double smallp_intersect, smallp_ratio; //
  pcl::PointXYZ vector_pt;
  //double vx, vy, vz;
  bool if_log_debug_print;
  double edge_point_angle_min, edge_point_angle_max;
};