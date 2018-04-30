#ifndef NDT_HH_
#define NDT_HH_

#include <iostream>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Pose.h>

// pcl related inputs
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>


class Voxel
{
public:
    Voxel(){
        mean.setZero();
        covariance.setZero();
        numPoints = 0;
    }

    Eigen::Vector3d mean;
    Eigen::Matrix3d covariance;
    int numPoints;
};

class VoxelGrid
{public:
    typedef std::vector<std::vector<std::vector<Voxel> > > Grid;
    Grid grid;
    void resize(const int& nx, const int& ny,const int& nz){
        grid.clear();
        grid.resize(nx);
        for (Grid::iterator it=grid.begin();it!=grid.end();it++){
            it->resize(ny);
            for(std::vector<std::vector<Voxel> >::iterator it2=it->begin();it2!=it->end();it2++)
            {
                it2->resize(nz);
            }

        }

    }
};


class NDT
{

public: NDT(ros::NodeHandle &n);

public:  ~NDT();
public: typedef pcl::PointCloud<pcl::PointXYZ> Cloud;

    // Subscriber for initial pose guess
public: void initialPoseCallback(const geometry_msgs::Pose::ConstPtr& input);

    // Subscriber for map pointcloud
public: void mapCallback (const sensor_msgs::PointCloud2ConstPtr& input);

//public: void readMap();

// Subscriber for pointcloud scan data
public: void scanCallback (const sensor_msgs::PointCloud2ConstPtr& input);

public: void computeTransform(const Cloud::Ptr &map_cloud, const Cloud::Ptr &scan_cloud, const geometry_msgs::Pose::ConstPtr &init_pose);

public: void voxelize_find_boundaries(const Cloud& cloud, VoxelGrid& vgrid);

private: ros::Subscriber mapSub;
private: ros::Subscriber scanSub;
private: ros::Subscriber initialPoseSub;
private: ros::NodeHandle nh;
public: Cloud mapCloud;
public: Cloud scanCloud;
public: std::string mapfile;
public: geometry_msgs::Pose initPose;
public: geometry_msgs::Pose currentPose;
public: Eigen::Matrix<double, 6, 1> CurrentPoseRPY;

private:
    double score;
    double  x_min, x_max, y_min, y_max, z_min, z_max;
    double gaussian_d1, gaussian_d2;

    double outlier_ratio_, resolution_;

//    Eigen::Matrix<double, 3, 6> pt_gradient_;
//
//    Eigen::Matrix<double, 18, 6> pt_hessian_;

    void computeHessian(const Eigen::Matrix<double, 6, 1> &p, const pcl::PointXYZ &pt, Eigen::Matrix<double, 18, 6>& pt_hessian_);

    void computeDerivative(const Eigen::Matrix<double, 6, 1> &p, const pcl::PointXYZ &pt, Eigen::Matrix<double, 3, 6>& pt_gradient_);
    void loadMap(std::string map_file);
};


#endif
