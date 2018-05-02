
#include <ndt.hpp>
#include <ros/ros.h>
#include <tf/transform_datatypes.h>
#include <pcl/io/pcd_io.h>
#include <visualization_msgs/Marker.h>
#include <ctime>
#include <tf/transform_listener.h>
NDT::NDT(ros::NodeHandle &n): nh(n)
{
    scanSub = nh.subscribe("/filtered_points", 10, &NDT::scanCallback, this);
    initialPoseSub = nh.subscribe("/initialpose", 20, &NDT::initialPoseCallback, this);
    mapPub = nh.advertise<sensor_msgs::PointCloud2>("/map_pcd", 1, true);
    transformCloudPub = nh.advertise<sensor_msgs::PointCloud2>("/my_pointcloud", 10);
    // Current pose publisher
    posePub = nh.advertise<geometry_msgs::PoseStamped>("/estimated_pose", 100);

    nh.getParam("outlier_ratio", outlier_ratio_);
    nh.getParam("resolution", resolution_);
    nh.getParam("max_iterations", max_iterations_);
    nh.getParam("transformation_eps", transformation_eps);

    // Initializing values for the gaussians
    double  gaussian_c1, gaussian_c2, gaussian_d3;
    gaussian_c1 = 10*(1-outlier_ratio_);
    gaussian_c2 = outlier_ratio_/pow(resolution_,3);
    gaussian_d3 = -log (gaussian_c2);
    gaussian_d1 = -log (gaussian_c1+gaussian_c2) - gaussian_d3;
    gaussian_d2 = -2*log((-log(gaussian_c1*exp(-0.5)+gaussian_c2)-gaussian_d3)/gaussian_d1);

    pcl::PCDReader reader;
    nh.getParam("map_file", mapfile);
    reader.read(mapfile,mapCloud);
    ROS_INFO("Loaded Map cloud, Got point cloud with %ld points", mapCloud.size());

    // Publsihing map on /map_pcd for viewing in rviz
    mapCloud.header.frame_id = "/map";
    sensor_msgs::PointCloud2 msg;
    pcl::toROSMsg(mapCloud,msg);
    msg.header.frame_id = "/map";
    mapPub.publish(msg);

    // Create voxel grid
    NDT::voxelize_find_boundaries(mapCloud, vgrid);

    // Flag which is set to false until the initialpose is set
    initPose_set = false;

}

NDT::~NDT()
{

}

/// brief: function to create voxel grids:
///         Given an input cloud, the boundaries in x , y and z are find out
///         Voxels are  created by diving the limits by resolution
///         All points are assigned to their respective voxels
///         Mean and covariance are calculated for each voxel
///         Calculating covariance some specical considerations are made to avoid singularitites
void NDT::voxelize_find_boundaries(const Cloud& cloud, VoxelGrid& v_grid)
{

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


    int voxel_num_x, voxel_num_y, voxel_num_z;

    voxel_num_x = floor((x_max-x_min)/resolution_);
    voxel_num_y = floor((y_max-y_min)/resolution_);
    voxel_num_z = floor((z_max-z_min)/resolution_);

    //   Creating a voxel grid
    v_grid.voxel_numx = voxel_num_x +1;
    v_grid.voxel_numy = voxel_num_y +1;
    v_grid.voxel_numz = voxel_num_z +1;

    v_grid.resize(voxel_num_x + 1, voxel_num_y + 1, voxel_num_z + 1);

    v_grid.reset(voxel_num_x + 1, voxel_num_y + 1, voxel_num_z + 1);

    // Mean
    for (size_t i = 0; i < cloud.size(); i++) {
        x = cloud.points.at(i).x;
        y = cloud.points.at(i).y;
        z = cloud.points.at(i).z;

        int idx = floor((x - x_min) / resolution_);
        int idy = floor((y - y_min) / resolution_);
        int idz = floor((z - z_min) / resolution_);

        Eigen::Vector3d point(x, y, z);

        v_grid.grid[idx][idy][idz].mean += point;
        v_grid.grid[idx][idy][idz].numPoints += 1;
    }

    // computing mean in each voxel
    for (size_t i=0; i<= voxel_num_x; i++) {
        for(size_t j =0; j<= voxel_num_y; j++ ) {
            for (size_t k = 0; k<= voxel_num_z; k++) {
                if (v_grid.grid[i][j][k].numPoints != 0)
                v_grid.grid[i][j][k].mean /= v_grid.grid[i][j][k].numPoints;
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

        v = point - v_grid.grid[idx][idy][idz].mean;
        v_grid.grid[idx][idy][idz].covariance += v*v.transpose();

    }

    int voxels_with_covariance = 0;
    // Computing covariance in each voxel
    for (size_t i=0; i< voxel_num_x; i++) {
        for (size_t j = 0; j <  voxel_num_y; j++) {
            for (size_t k = 0; k <voxel_num_z; k++) {
                if (v_grid.grid[i][j][k].numPoints > 6)
                {
                 voxels_with_covariance+=1;
                v_grid.grid[i][j][k].covariance /= (v_grid.grid[i][j][k].numPoints-1);

                //Normalizing covariance to remove singularities
                eigenslover.compute(v_grid.grid[i][j][k].covariance);
                eigen_val = eigenslover.eigenvalues().asDiagonal();
                eigenvecs = eigenslover.eigenvectors();
                if (eigen_val(0,0)<0 || eigen_val(1,1) <0 || eigen_val (2,2) <=0)
                {
                    v_grid.grid[i][j][k].numPoints =-1;
                    continue;
                }
                if (eigen_val(0,0) < min_covar_eigenvalue)
                {
                    eigen_val(0,0) = min_covar_eigenvalue;
                    if (eigen_val(1,1) < min_covar_eigenvalue)
                    {
                        eigen_val(1,1)  = min_covar_eigenvalue;
                    }
                    v_grid.grid[i][j][k].covariance = eigenvecs*eigen_val*eigenvecs.inverse();

                    icov = v_grid.grid[i][j][k].covariance.inverse();
                    if (icov.maxCoeff() == std::numeric_limits<float>::infinity()
                            || icov.minCoeff() == -std::numeric_limits<float>::infinity())
                    {
                        v_grid.grid[i][j][k].numPoints =-1;
                    }
                }

                }
            }
        }
    }
}


/// Brief : callback to subscribe initial pose and store it in the current pose :
///         Here current pose is also computed to get RPY values which are useful in further calculations
void NDT::initialPoseCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& input)
{   currentPose.pose.position.x = input->pose.pose.position.x;
    currentPose.pose.position.y = input->pose.pose.position.y;
    currentPose.pose.position.z = input->pose.pose.position.z;
    currentPose.pose.orientation.x = input->pose.pose.orientation.x;
    currentPose.pose.orientation.y = input->pose.pose.orientation.y;
    currentPose.pose.orientation.z = input->pose.pose.orientation.z;
    currentPose.pose.orientation.w = input->pose.pose.orientation.w;
    currentPose.header.frame_id = input->header.frame_id;
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

/// Brief: callback function where the poincloud scan is being match with base or map scan
void NDT::scanCallback (const sensor_msgs::PointCloud2ConstPtr& input)
{
    // Run only if the initial pose is set
    if (initPose_set == true) {

        Cloud::Ptr cloud(new Cloud());
        scanCloud = *cloud;
        sensor_msgs::PointCloud2 msg = *input;
        msg.header.frame_id = "/map";
        pcl::fromROSMsg(msg, scanCloud);

        double x, y, z;

        // For each point in the scan
        // tranfrom the point cloud T(p,x)
        Cloud::Ptr transCloud(new Cloud());

        Eigen::Matrix<double, 6, 1> grad;
        Eigen::Matrix<double, 6, 6> hessian;
        Eigen::Matrix<double, 6, 1> deltaP;
        deltaP << 1, 1, 1, 1, 1, 1;

        // find the cell in which the point is lying
        int n_iterations = 0;

        Eigen::Matrix<double, 3, 6> pt_gradient_;
        Eigen::Matrix<double, 18, 6> pt_hessian_;

            bool converged = false;

            while (!converged )

            {
            n_iterations += 1;
            score = 0;
            grad.setZero();
            hessian.setZero();
            pt_gradient_.setZero();
            pt_gradient_.block<3, 3>(0, 0).setIdentity();
            pt_hessian_.setZero();

            tf::TransformListener listener;
            tf::StampedTransform transform;

            try {
                listener.waitForTransform("/map", "/world",
                                          ros::Time(0), ros::Duration(5.0));
                listener.lookupTransform("/map", "/world", ros::Time(0), transform);
            }
            catch (tf::TransformException &ex) {
                       ROS_ERROR("%s",ex.what());
                       ros::Duration(1.0).sleep();
                       continue;
                     }
//            Eigen::Matrix<double, 3, 1> position(currentPose.pose.position.x + transform.getOrigin().x() , currentPose.pose.position.y + transform.getOrigin().y(),
//                                                 currentPose.pose.position.z+ transform.getOrigin().z());
            Eigen::Matrix<double, 3, 1> position(currentPose.pose.position.x, currentPose.pose.position.y ,
                                                 currentPose.pose.position.z);
            Eigen::Quaternion<double> rotation(currentPose.pose.orientation.w, currentPose.pose.orientation.x,
                                               currentPose.pose.orientation.y, currentPose.pose.orientation.z);

            pcl::transformPointCloud(scanCloud, *transCloud, position, rotation);

            transCloud->header.frame_id = "/map";
            sensor_msgs::PointCloud2 msg;
            pcl::toROSMsg(*transCloud,msg);
            msg.header.frame_id = "/map";
            transformCloudPub.publish(msg);
//            std::cout << "#################Iteration Number########### " << n_iterations << "\n";

            bool compute_gradient = false;

            for (size_t i = 0; i < transCloud->size(); i++) {
                // Find the cell in which the point is lying
                x = transCloud->points.at(i).x;
                y = transCloud->points.at(i).y;
                z = transCloud->points.at(i).z;
                int idx = floor((x - x_min) / resolution_);
                int idy = floor((y - y_min) / resolution_);
                int idz = floor((z - z_min) / resolution_);
                std::cout << "size of the transformed pointcloud is :" << transCloud->size() << "\n";

                // Checking if the transofrmed point is within the limits of the voxel grid we have for map
                if ((idx >= 0) && (idx < vgrid.voxel_numx) && (idy >= 0) && (idy < vgrid.voxel_numy) && (idz >= 0) &&
                    (idz < vgrid.voxel_numz)) {

                    Eigen::Vector3d mean = vgrid.grid[idx][idy][idz].mean;
                    Eigen::Matrix3d covar = vgrid.grid[idx][idy][idz].covariance;
                    Eigen::Matrix3d covarinv = covar.inverse();
                    Eigen::Vector3d Xpt(x, y, z); //Store the point
                    Eigen::Vector3d X = Xpt - mean;

                    // Computing PDFs only for those voxels with more than 4 points
                    if (vgrid.grid[idx][idy][idz].numPoints > 4) {
                        compute_gradient = true;
                        double a = X.transpose() * covarinv * X;
                        std::cout << "Covariance : " << "\n" << covar << "\n";
                        std::cout << "Covariance inverse     : " << covarinv << "\n";
                        score = score + gaussian_d1 * (-exp(-(a * gaussian_d2 / 2)));
//                        std::cout << "score: " << score << "\n";

                        // Sending X = Xpt- mean instead of Xpt can changed back to Xpt
                        computeDerivative(CurrentPoseRPY, Xpt, pt_gradient_);
                        computeHessian(CurrentPoseRPY, Xpt, pt_hessian_);

                        // update gradient
                        if (vgrid.grid[idx][idy][idz].numPoints > 6) {
                            for (size_t i = 0; i < 6; i++) {
                                double a = X.transpose() * covarinv * X;
                                double b = X.transpose() * covarinv * pt_gradient_.col(i);
                                grad(i) += gaussian_d1 * gaussian_d2 * b * exp(-(a * gaussian_d2 / 2));
                                // update hessian
                                for (size_t j = 0; j < 6; j++) {
                                    double c = X.transpose() * covarinv * pt_gradient_.col(j);
                                    double d = X.transpose() * covarinv * pt_hessian_.block<3, 1>(3 * i, j);
                                    double e = pt_gradient_.col(j).transpose() * covarinv * pt_gradient_.col(i);
                                    double f = b * c;
                                    hessian(i, j) +=
                                            gaussian_d1 * gaussian_d2 * exp((-gaussian_d2 / 2) * a) *
                                            ((-gaussian_d2 * f) + d + e);

                                }
                            }
                        }
                    }
                }
            }

            // updates only if any of the transformed points align with the mapcloud voxels
                if (compute_gradient == true) {
//                    std::cout << "gradient : " << grad << std::endl;
//                    std::cout << "Hessian : " << hessian << std::endl;

                    // Solve for detlaP = - Hinv*g

                    deltaP = -hessian.inverse() * grad;

//                    std::cout << "Hessian inverse: " << hessian.inverse() << std::endl;
//                    std::cout << "deltaP" << deltaP << std::endl;

                    CurrentPoseRPY = CurrentPoseRPY + deltaP;

                    tf::Quaternion q = tf::createQuaternionFromRPY(CurrentPoseRPY(3), CurrentPoseRPY(4),
                                                                   CurrentPoseRPY(5));
                    currentPose.pose.position.x = CurrentPoseRPY(0);
                    currentPose.pose.position.y = CurrentPoseRPY(1);
                    currentPose.pose.position.z = CurrentPoseRPY(2);
                    currentPose.pose.orientation.w = q.w();
                    currentPose.pose.orientation.x = q.x();
                    currentPose.pose.orientation.y = q.y();
                    currentPose.pose.orientation.z = q.z();

                    // Publishing the current pose on
//                    geometry_msgs::PoseStamped ps;
//                    ps.header.frame_id = "/map";
//                    ps.pose = currentPose;
                    posePub.publish(currentPose);
                    if (n_iterations > max_iterations_ || deltaP.norm() < transformation_eps){
                        converged = true;
                    }
                }

                if (n_iterations>max_iterations_)
                {
                    converged = true;
                }
            }
        }
        // Publishing the current pose on
//        geometry_msgs::PoseStamped ps;
//        ps.header.frame_id = "/map";
//        ps.pose = currentPose;
//        posePub.publish(ps);
//          posePub.publish(currentPose);

}



/// Brief: computes a point gradient which is used in derivate uodate step
void NDT::computeDerivative(const Eigen::Matrix<double, 6, 1> &p, const Eigen::Vector3d &X ,Eigen::Matrix<double, 3, 6>& pt_gradient_)
{
    double cx, cy, cz, sx, sy, sz;

    // Computing  Eq 6.18

    if (fabs (p (3)) < 10e-5)  //p(3) = 0;
    {
        cx = 1.0;
        sx = 0.0;
    }
    else
    {
        cx = cos (p (3));
        sx = sin (p (3));
    }
    if (fabs (p (4)) < 10e-5) //p(4) = 0;
    {
        cy = 1.0;
        sy = 0.0;
    }
    else
    {
        cy = cos (p (4));
        sy = sin (p (4));
    }

    if (fabs (p (5)) < 10e-5) //p(5) = 0;
    {
        cz = 1.0;
        sz = 0.0;
    }
    else
    {
        cz = cos (p (5));
        sz = sin (p (5));
    }

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

/// Brief: this method computes the point hessain which is used in the hessain update step
void NDT::computeHessian(const Eigen::Matrix<double, 6, 1> &p, const Eigen::Vector3d & X,  Eigen::Matrix<double, 18, 6>& pt_hessian_)
{

    Eigen::Vector3d h_a2_, h_a3_,
            h_b2_, h_b3_,
            h_c2_, h_c3_,
            h_d1_, h_d2_, h_d3_,
            h_e1_, h_e2_, h_e3_,
                    h_f1_, h_f2_, h_f3_;

    double cx, cy, cz, sx, sy, sz;

    // simplification for near angles near 0:

    if (fabs (p (3)) < 10e-5)  //p(3) = 0;
    {
        cx = 1.0;
        sx = 0.0;
    }
    else
    {
        cx = cos (p (3));
        sx = sin (p (3));
    }
    if (fabs (p (4)) < 10e-5) //p(4) = 0;
    {
        cy = 1.0;
        sy = 0.0;
    }
    else
    {
        cy = cos (p (4));
        sy = sin (p (4));
    }

    if (fabs (p (5)) < 10e-5) //p(5) = 0;
    {
        cz = 1.0;
        sz = 0.0;
    }
    else
    {
        cz = cos (p (5));
        sz = sin (p (5));
    }

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

    ros::Rate rate(10);
    while(ros::ok()){
        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}


