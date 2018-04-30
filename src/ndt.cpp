
#include <ndt.hpp>
#include <ros/ros.h>
#include <tf/transform_datatypes.h>
#include <pcl/io/pcd_io.h>

NDT::NDT(ros::NodeHandle &n): nh(n)
{
    scanSub = nh.subscribe("/filtered_points", 10, &NDT::scanCallback, this);
//    mapSub = nh.subscribe("/cloud_pcd", 10, &NDT::mapCallback, this);
    initialPoseSub = nh.subscribe("/initialpose", 10, &NDT::initialPoseCallback, this);

    // Current pose publisher
    posePub = nh.advertise<geometry_msgs::Pose>("/ndt/pose", 100);

    // Initializing values for the gaussians
    double  gaussian_c1, gaussian_c2, gaussian_d3;
    outlier_ratio_ = 0.55;
    resolution_ = 10;
    gaussian_c1 = 10*(1-outlier_ratio_);
    gaussian_c2 = outlier_ratio_/pow(resolution_,3);
    gaussian_d3 = -log (gaussian_c2);
    gaussian_d1 = -log (gaussian_c1+gaussian_c2) - gaussian_d3;
    gaussian_d2 = -2*log((-log(gaussian_c1*exp(-0.5)+gaussian_c2)-gaussian_d3)/gaussian_d1);

    pcl::PCDReader reader;

    reader.read("/home/rsalem/apex/workspace/src/ndt/data/map.pcd",scanCloud);
    ROS_INFO("Got point cloud with %ld points", scanCloud.size());
    initPose_set = false;
}

NDT::~NDT()
{

}

void NDT::voxelize_find_boundaries(const Cloud& cloud, VoxelGrid& vgrid)
{
//    std::cout<< "inside voxelize" << "\n";
//    double  x_min, x_max, y_min, y_max, z_min, z_max;
    double  x, y, z ;


    // initialize first point to min and max values
    x_min = x_max = cloud.points.at(0).x;
    y_min = y_max = cloud.points.at(0).y;
    z_min = z_max = cloud.points.at(0).z;

    // finding boundaries of the cloud
    for (size_t i = 0; i < cloud.size(); i++)
    {

        x = cloud.points.at(i).x;
        y = cloud.points.at(i).y;
        z = cloud.points.at(i).z;

        if (x_min > x)
        {
            x_min = x;
        }
        if (x_max < x)
        {
            x_max = x;
        }
        if (y_min > y)
        {
            y_min = y;
        }
        if (y_max < y)
        {
           y_max = y;
        }
        if (z_min > z)
        {
            z_min = z;
        }
        if (z_max < z)
        {
            z_max = z;
        }
    }
//    std::cout << " x_min of the voxel grid :" << x_min <<std::endl;


    int voxel_num_x, voxel_num_y, voxel_num_z;

    voxel_num_x = floor((x_max-x_min)/resolution_);
    voxel_num_y = floor((y_max-y_min)/resolution_);
    voxel_num_z = floor((z_max-z_min)/resolution_);

    // Creating a voxel grid
//    std::cout << " creating voxel grid"<<std::endl;
    vgrid.voxel_numx = voxel_num_x +1;
    vgrid.voxel_numy = voxel_num_y +1;
    vgrid.voxel_numz = voxel_num_z +1;
    vgrid.resize(voxel_num_x + 1, voxel_num_y + 1, voxel_num_z + 1);
    vgrid.reset(voxel_num_x + 1, voxel_num_y + 1, voxel_num_z + 1);

//    std::cout << " created  voxel grid :"  <<std::endl;

    // Mean
    for (size_t i = 0; i < cloud.size(); i++) {
        x = cloud.points.at(i).x;
        y = cloud.points.at(i).y;
        z = cloud.points.at(i).z;

        int idx = floor((x - x_min) / resolution_);
        int idy = floor((y - y_min) / resolution_);
        int idz = floor((z - z_min) / resolution_);
//        std::cout << idx << " " << idy <<  " " << idz << "\n";
//        std::cout << voxel_num_x << " " << voxel_num_y <<  " " << voxel_num_z<< "\n";
        Eigen::Vector3d point(x, y, z);

        vgrid.grid[idx][idy][idz].mean += point;
        vgrid.grid[idx][idy][idz].numPoints += 1;
//        std::cout << vgrid.grid[idx][idy][idz].numPoints << "\n";
//        std::cout << " inside calculating mean :" <<vgrid.grid[idx][idy][idz].mean  <<std::endl;

    }

    // computing mean in each voxel
    for (size_t i=0; i<= voxel_num_x; i++) {
//        std::cout << "x mean" << "\n";
        for(size_t j =0; j<= voxel_num_y; j++ ) {
            for (size_t k = 0; k<= voxel_num_z; k++) {
                if (vgrid.grid[i][j][k].numPoints != 0)
//                    std::cout << " inside calculating mean for each voxel :"  <<std::endl;

                vgrid.grid[i][j][k].mean /= vgrid.grid[i][j][k].numPoints;
            }
        }
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigenslover;
    Eigen::Matrix3d eigen_val;
    Eigen::Matrix3d eigenvecs;
    Eigen::Matrix3d icov;
    double min_covar_eigenvalue = 0.01;
    // Covariance
    for (size_t i =  0; i< cloud.size(); i++) {

        x = cloud.points.at(i).x;
        y = cloud.points.at(i).y;
        z = cloud.points.at(i).z;

        // to find out which point lies in which voxel
        int idx = floor((x - x_min) / resolution_);
        int idy = floor((y - y_min) / resolution_);
        int idz = floor((z - z_min) / resolution_);

        Eigen::Vector3d point(x, y, z);
        Eigen::Vector3d v;

        v = point - vgrid.grid[idx][idy][idz].mean;
        vgrid.grid[idx][idy][idz].covariance += v*v.transpose();

    }

    // Computing covariance in each voxel
    for (size_t i=0; i< voxel_num_x; i++) {
        for (size_t j = 0; j <  voxel_num_y; j++) {
            for (size_t k = 0; k <voxel_num_z; k++) {
//                std::cout << vgrid.grid[i][j][k].numPoints << "\n";
                if (vgrid.grid[i][j][k].numPoints > 6)

                vgrid.grid[i][j][k].covariance /= (vgrid.grid[i][j][k].numPoints-1);

                //Normalizing covariance to remove singularities
                eigenslover.compute(vgrid.grid[i][j][k].covariance);
                eigen_val = eigenslover.eigenvalues().asDiagonal();
                eigenvecs = eigenslover.eigenvectors();
                if (eigen_val(0,0)<0 || eigen_val(1,1) <0 || eigen_val (2,2) <=0)
                {
                    vgrid.grid[i][j][k].numPoints =-1;
                    continue;
                }
                if (eigen_val(0,0) < min_covar_eigenvalue)
                {
                    eigen_val(0,0) = min_covar_eigenvalue;
                    if (eigen_val(1,1) < min_covar_eigenvalue)
                    {
                        eigen_val(1,1)  = min_covar_eigenvalue;
                    }
                    vgrid.grid[i][j][k].covariance = eigenvecs*eigen_val*eigenvecs.inverse();
//                    std::cout << "Inside normalizing step " << "\n";
//                    std::cout << vgrid.grid[i][j][k].covariance << "\n";
                    icov = vgrid.grid[i][j][k].covariance.inverse();
                    if (icov.maxCoeff() == std::numeric_limits<float>::infinity()
                            || icov.minCoeff() == -std::numeric_limits<float>::infinity())
                    {
                        vgrid.grid[i][j][k].numPoints =-1;
                    }
                }
            }
        }
    }
}


void NDT::initialPoseCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& input)
{   currentPose.position.x = input->pose.pose.position.x;
    currentPose.position.y = input->pose.pose.position.y;
    currentPose.position.z = input->pose.pose.position.z;
    currentPose.orientation.x = input->pose.pose.orientation.x;
    currentPose.orientation.y = input->pose.pose.orientation.y;
    currentPose.orientation.z = input->pose.pose.orientation.z;
    double  roll, pitch, yaw;
    tf::Quaternion q(
            input->pose.pose.orientation.x,
            input->pose.pose.orientation.y,
            input->pose.pose.orientation.z,
            input->pose.pose.orientation.w);
    tf::Matrix3x3(q).getRPY(roll, pitch, yaw);
    CurrentPoseRPY << input->pose.pose.position.x , input->pose.pose.position.y ,input->pose.pose.position.z , roll ,pitch , yaw;
    initPose_set = true;
    std::cout << "initial Pose is set " << "\n";
}

void NDT::mapCallback (const sensor_msgs::PointCloud2ConstPtr& input)
{
    Cloud::Ptr cloud(new Cloud());
    mapCloud = *cloud;
    pcl::fromROSMsg(*input, mapCloud);
//    ROS_INFO("Got point cloud with %ld points", mapCloud.size());
}


void NDT::scanCallback (const sensor_msgs::PointCloud2ConstPtr& input)
{
    // Run only if the initial pose is set
    if (initPose_set == true) {

        Cloud::Ptr cloud(new Cloud());
        scanCloud = *cloud;
        pcl::fromROSMsg(*input, scanCloud);
        ROS_INFO("Got point cloud with %ld points", scanCloud.size());

        // store voxel grid
        VoxelGrid vgrid;
        NDT::voxelize_find_boundaries(scanCloud, vgrid);

        double x, y, z;

        // For each point in the scan
        // tranfrom the point cloud T(p,x)
        Cloud::Ptr transCloud(new Cloud());

//        Eigen::Matrix<double, 3, 1> position(currentPose.position.x, currentPose.position.y, currentPose.position.z);
//        Eigen::Quaternion<double> rotation(currentPose.orientation.w, currentPose.orientation.x,
//                                           currentPose.orientation.y, currentPose.orientation.z);

        Eigen::Matrix<double, 6, 1> grad;
        Eigen::Matrix<double, 6, 6> hessian;
        Eigen::Matrix<double, 6, 1> deltaP;

//        std::cout << "transforming Pointcloud" << std::endl;
//        pcl::transformPointCloud(scanCloud, *transCloud, position, rotation);
//        std::cout << "transforming Pointcloud done...." << "\n";
        deltaP << 1, 1, 1, 1, 1, 1;
//    grad.setZero();
//    hessian.setZero();

        // find the cell in which the point is lying
        int n_iterations = 0;

        Eigen::Matrix<double, 3, 6> pt_gradient_;
        Eigen::Matrix<double, 18, 6> pt_hessian_;

//    pt_gradient_.setZero();
//    pt_gradient_.block<3, 3>(0, 0).setIdentity ();
//    pt_hessian_.setZero();
//    score = 0;

    while(deltaP(0) > 0.5 && deltaP(1) > 0.5 && deltaP(2) > 0.5 && deltaP(3) > 0.2 && deltaP(4) > 0.2 && deltaP(5) > 0.3 && n_iterations < 10 )  {
            grad.setZero();
            hessian.setZero();
            pt_gradient_.setZero();
            pt_gradient_.block<3, 3>(0, 0).setIdentity();
            pt_hessian_.setZero();
            score = 0;

            Eigen::Matrix<double, 3, 1> position(currentPose.position.x, currentPose.position.y,
                                                 currentPose.position.z);
            Eigen::Quaternion<double> rotation(currentPose.orientation.w, currentPose.orientation.x,
                                               currentPose.orientation.y, currentPose.orientation.z);

            pcl::transformPointCloud(scanCloud, *transCloud, position, rotation);
            std::cout << "#################Iteration Number########### " << n_iterations << "\n";

            n_iterations += 1;
            std::cout << "computing score" << std::endl;
            bool compute_gradient = false;
            for (size_t i = 0; i < transCloud->size(); i++) {
                // Find the cell in which the point is lying
                x = transCloud->points.at(i).x;
                y = transCloud->points.at(i).y;
                z = transCloud->points.at(i).z;
                int idx = floor((x - x_min) / resolution_);
                int idy = floor((y - y_min) / resolution_);
                int idz = floor((z - z_min) / resolution_);

                if ( (idx >= 0) && (idx < vgrid.voxel_numx) && (idy >= 0) && (idy < vgrid.voxel_numy) &&(idz >= 0) && (idz < vgrid.voxel_numz)) {
//                    std::cout << "inside computing voxel loop" << "\n";
//                    std:: cout << "idx: " << idx << " " << "idy: " << idy << " " << "idz: "<< idz << "\n";
//                    std:: cout << "maxx: " << vgrid.voxel_numx << " " << "maxy:" << vgrid.voxel_numy  << " " << "maxz: " <<vgrid.voxel_numz << "\n";
                    Eigen::Vector3d mean = vgrid.grid[idx][idy][idz].mean;
                    Eigen::Matrix3d covar = vgrid.grid[idx][idy][idz].covariance;
                    Eigen::Matrix3d covarinv = covar.inverse();
                    Eigen::Vector3d Xpt(x, y, z); //Store the point
                    Eigen::Vector3d diff = Xpt - mean;

//            std::cout << covar << "\n";
//            std::cout << "score:: " << score << "\n";
                    // Computing PDFs only for those voxels with more than 5 points
                    if (vgrid.grid[idx][idy][idz].numPoints > 6) {
                        compute_gradient = true;
                        double a = diff.transpose() * covarinv * diff;
                        std::cout << "Covariance : " << covar;
                        std::cout << "Covariance inverse     : " << covarinv;
                        std::cout << "a" << a << "\n";
                        score = score + gaussian_d1 * (-exp(-(a * gaussian_d2 / 2)));
                        std::cout << "score: " << score << "\n";
                    }
                }
            }

            // Compute gradient
            if (compute_gradient == true) {
                std::cout << "Computing gradient" << std::endl;
                for (size_t i = 0; i < transCloud->size(); i++) {

                    x = transCloud->points.at(i).x;
                    y = transCloud->points.at(i).y;
                    z = transCloud->points.at(i).z;

                    int idx = floor((x - x_min) / resolution_);
                    int idy = floor((y - y_min) / resolution_);
                    int idz = floor((z - z_min) / resolution_);
                    if ((idx >= 0) && (idx < vgrid.voxel_numx) && (idy >= 0) && (idy < vgrid.voxel_numy) &&
                        (idz >= 0) && (idz < vgrid.voxel_numz)) {
                        Eigen::Vector3d mean = vgrid.grid[idx][idy][idz].mean;
                        Eigen::Matrix3d covar = vgrid.grid[idx][idy][idz].covariance;
                        Eigen::Matrix3d covarinv = covar.inverse();
                        Eigen::Vector3d Xpt(x, y, z);

                        // Subtracting mean from the point
                        Eigen::Vector3d X = Xpt - mean;

                        computeDerivative(CurrentPoseRPY, transCloud->points.at(i), pt_gradient_);
                        computeHessian(CurrentPoseRPY, transCloud->points.at(i), pt_hessian_);

                        // update gradient
                        if (vgrid.grid[idx][idy][idz].numPoints > 6) {
                            for (size_t i = 0; i < 6; i++) {
                                double a = X.transpose() * covarinv * X;
//                    std::cout << "a: " << a << std::endl;
                                double b = X.transpose() * covarinv * pt_gradient_.col(i);
//                    std::cout << "b: " << b << std::endl;
                                grad(i) += gaussian_d1 * gaussian_d2 * b * exp(-(a * gaussian_d2 / 2));
//                    std::cout << "grad (i): " << grad(i) << std::endl;
                                // update hessian
                                for (size_t j = 0; j < 6; j++) {
                                    double c = X.transpose() * covarinv * pt_gradient_.col(j);
                                    double d = X.transpose() * covarinv * pt_hessian_.block<3, 1>(3 * i, j);
                                    double e = pt_gradient_.col(j).transpose() * covarinv * pt_gradient_.col(i);
                                    double f = b * c;
                                    hessian(i, j) +=
                                            gaussian_d1 * gaussian_d2 * exp((-gaussian_d2 / 2) * a) *
                                            (-gaussian_d2 * f) + d +
                                            e;
//                        std::cout << "Hess(i,j)" << hessian(i,j) << std::endl;
                                }
                            }
                        }
                    }

                }


                std::cout << "gradient : " << grad << std::endl;
                std::cout << "Hessian : " << hessian << std::endl;

                // Solve for detlaP = - Hinv*g
                deltaP = -hessian.inverse() * grad;

                std::cout << "Hessian inverse: " << hessian.inverse() << std::endl;

                std::cout << "deltaP" << deltaP << std::endl;

                CurrentPoseRPY = CurrentPoseRPY + deltaP;

                tf::Quaternion q = tf::createQuaternionFromRPY(CurrentPoseRPY(3), CurrentPoseRPY(4), CurrentPoseRPY(5));
                currentPose.position.x = CurrentPoseRPY(0);
                currentPose.position.y = CurrentPoseRPY(1);
                currentPose.position.z = CurrentPoseRPY(2);
                currentPose.orientation.w = q.w();
                currentPose.orientation.x = q.x();
                currentPose.orientation.y = q.y();
                currentPose.orientation.z = q.z();
                std::cout << "current Pose :" << std::endl;
                std::cout << CurrentPoseRPY(0) << " " << CurrentPoseRPY(1) << " " << CurrentPoseRPY(2) << std::endl;
                std::cout << deltaP(0) << " " << deltaP(1) << " " << deltaP(2) << " " << deltaP(3) << " " << deltaP(4)
                          << deltaP(5) << "\n";
            }

        }
        posePub.publish(currentPose);
    }
}


// input is a vector with [trans, roll, pitch , yaw]
void NDT::computeDerivative(const Eigen::Matrix<double, 6, 1> &p, const pcl::PointXYZ &pt,Eigen::Matrix<double, 3, 6>& pt_gradient_)
{
    double cx, cy, cz, sx, sy, sz;

    // Computing  Eq 6.18

    cx = cos(p[3]);
    sx = sin(p[3]);
    cy = cos(p[4]);
    sy = sin(p[4]);
    cz = cos(p[5]);
    sz = sin(p[5]);

    Eigen::Vector3d j_a, j_b, j_c, j_d, j_e, j_f, j_g, j_h;

    // Populating the vectors that can be used to find dot product with the position vector later
    j_a << -sx * sz + cx * sy * cz,-sx * cz -cx * sy * sz, -cx * cy;
    j_b << cx * sz + sx * sy * cz, cx * cz - sx * sy * sz, -sx * cy;
    j_c << -sy * cz , sy * sz, cy;
    j_d << sx * cy * cz , -sx * cy * sz, sx * sy;
    j_e << -cx * cy * cz, cx * cy * sz, -cx * sy;
    j_f << -cy * sz, -cy * cz, 0;
    j_g << cx * cz - sx * sy * sz, -cx * sz - sx * sy * cz, 0;
    j_h << sx * cz + cx * sy * sz, cx * sy * cz - sx * sz, 0;

    // To store the point from the point cloud
    Eigen::Vector3d X(pt.x, pt.y, pt.z);

    // Computing gradient for a point
    pt_gradient_.setZero();
    pt_gradient_(0,0) = 1;
    pt_gradient_(1,1) = 1;
    pt_gradient_(2,2) = 1;
    pt_gradient_(1,3) = X.dot(j_a);
    pt_gradient_(2,3) = X.dot(j_b);
    pt_gradient_(0,4) = X.dot(j_c);
    pt_gradient_(1,4) = X.dot(j_d);
    pt_gradient_(2,4) = X.dot(j_e);
    pt_gradient_(0,5) = X.dot(j_f);
    pt_gradient_(1,5) = X.dot(j_g);
    pt_gradient_(2,5) = X.dot(j_h);

}

void NDT::computeHessian(const Eigen::Matrix<double, 6, 1> &p, const pcl::PointXYZ &pt,  Eigen::Matrix<double, 18, 6>& pt_hessian_)
{

    Eigen::Vector3d h_a2_, h_a3_,
            h_b2_, h_b3_,
            h_c2_, h_c3_,
            h_d1_, h_d2_, h_d3_,
            h_e1_, h_e2_, h_e3_,
                    h_f1_, h_f2_, h_f3_;

    double cx, cy, cz, sx, sy, sz;

    cx = cos(p[3]);
    sx = sin(p[3]);
    cy = cos(p[4]);
    sy = sin(p[4]);
    cz = cos(p[5]);
    sz = sin(p[5]);

    h_a2_ << (-cx * sz - sx * sy * cz), (-cx * cz + sx * sy * sz), sx * cy;
    h_a3_ << (-sx * sz + cx * sy * cz), (-cx * sy * sz - sx * cz), (-cx * cy);

    h_b2_ << (cx * cy * cz), (-cx * cy * sz), (cx * sy);
    h_b3_ << (sx * cy * cz), (-sx * cy * sz), (sx * sy);

    h_c2_ << (-sx * cz - cx * sy * sz), (sx * sz - cx * sy * cz), 0;
    h_c3_ << (cx * cz - sx * sy * sz), (-sx * sy * cz - cx * sz), 0;

    h_d1_ << (-cy * cz), (cy * sz), (sy);
    h_d2_ << (-sx * sy * cz), (sx * sy * sz), (sx * cy);
    h_d3_ << (cx * sy * cz), (-cx * sy * sz), (-cx * cy);

    h_e1_ << (sy * sz), (sy * cz), 0;
    h_e2_ << (-sx * cy * sz), (-sx * cy * cz), 0;
    h_e3_ << (cx * cy * sz), (cx * cy * cz), 0;

    h_f1_ << (-cy * cz), (cy * sz), 0;
    h_f2_ << (-cx * sz - sx * sy * cz), (-cx * cz + sx * sy * sz), 0;
    h_f3_ << (-sx * sz + cx * sy * cz), (-cx * sy * sz - sx * cz), 0;

    // To store the point from the point cloud
    Eigen::Vector3d X(pt.x, pt.y, pt.z);

    Eigen::Vector3d a, b, c, d, e, f;
    a << 0, X.dot(h_a2_), X.dot(h_a3_);
    b << 0, X.dot(h_b2_), X.dot(h_b3_);
    c << 0, X.dot(h_c2_), X.dot(h_c3_);
    d << X.dot(h_d1_), X.dot(h_d2_), X.dot(h_d3_);
    e << X.dot(h_e1_), X.dot(h_e2_), X.dot(h_e3_);
    f << X.dot(h_f1_), X.dot(h_f2_), X.dot(h_f3_);

    //computing hessian for a point
    pt_hessian_.setZero();

    pt_hessian_.block<3,1>(9,3) = a;
    pt_hessian_.block<3, 1>(12, 3) = b;
    pt_hessian_.block<3, 1>(15, 3) = c;
    pt_hessian_.block<3, 1>(9, 4) = b;
    pt_hessian_.block<3, 1>(12, 4) = d;
    pt_hessian_.block<3, 1>(15, 4) = e;
    pt_hessian_.block<3, 1>(9, 5) = c;
    pt_hessian_.block<3, 1>(12, 5) = e;
    pt_hessian_.block<3, 1>(15, 5) = f;

}


int main (int argc, char **argv)
{
    ros::init(argc, argv, "NDTmatching");
    ros::NodeHandle nh;
    NDT obj(nh);

    std::string mapfile = "/home/rsalem/apex/workspace/src/ndt/data/map.pcd";
    obj.mapfile = mapfile;
    //    pcl::PCDReader reader;
    //    reader.read
    //    nh.param<std::string>("map_file", mapfile, "map.pcd");

    while(ros::ok()){
        ros::spin();
    }

    return 0;
}


