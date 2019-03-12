#include <iostream>
#include <Eigen/Dense>
#include "../include/modern_robotics.h"
#include "gtest/gtest.h"

# define M_PI           3.14159265358979323846  /* pi */

TEST(MRTest, InverseDynamicsTest) {
	Eigen::VectorXd thetalist(3);
	thetalist << 0.1, 0.1, 0.1;
	Eigen::VectorXd dthetalist(3);
	dthetalist << 0.1, 0.2, 0.3;
	Eigen::VectorXd ddthetalist(3);
	ddthetalist << 2, 1.5, 1;
	Eigen::VectorXd g(3);
	g << 0, 0, -9.8;
	Eigen::VectorXd Ftip(6);
	// Ftip << 1, 1, 1, 1, 1, 1;
    Ftip << 0, 0, 0, 0, 0, 0;

	std::vector<Eigen::MatrixXd> Mlist;
	std::vector<Eigen::MatrixXd> Glist;

	Eigen::Matrix4d M01;
	M01 << 1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0.089159,
		0, 0, 0, 1;
	Eigen::Matrix4d M12;
	M12 << 0, 0, 1, 0.28,
		0, 1, 0, 0.13585,
		-1, 0, 0, 0,
		0, 0, 0, 1;
	Eigen::Matrix4d M23;
	M23 << 1, 0, 0, 0,
		0, 1, 0, -0.1197,
		0, 0, 1, 0.395,
		0, 0, 0, 1;
	Eigen::Matrix4d M34;
	M34 << 1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0.14225,
		0, 0, 0, 1;

	Mlist.push_back(M01);
	Mlist.push_back(M12);
	Mlist.push_back(M23);
	Mlist.push_back(M34);

	Eigen::VectorXd G1(6);
	G1 << 0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7;
	Eigen::VectorXd G2(6);
	G2 << 0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393;
	Eigen::VectorXd G3(6);
	G3 << 0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275;

	Glist.push_back(G1.asDiagonal());
	Glist.push_back(G2.asDiagonal());
	Glist.push_back(G3.asDiagonal());

	Eigen::MatrixXd SlistT(3, 6);
	SlistT << 1, 0, 1, 0, 1, 0,
		0, 1, 0, -0.089, 0, 0,
		0, 1, 0, -0.089, 0, 0.425;
	Eigen::MatrixXd Slist = SlistT.transpose();

	Eigen::VectorXd taulist = mr::InverseDynamics(thetalist, dthetalist, ddthetalist, g,
		Ftip, Mlist, Glist, Slist);

	Eigen::VectorXd result(3);
	result << 74.6962, -33.0677, -3.23057;

	ASSERT_TRUE(taulist.isApprox(result, 4));
}

TEST(MRTest, MyInverseDynamicsTest) {
	Eigen::VectorXd thetalist(3);
	thetalist << 0.1, 0.1, 0.1;
	Eigen::VectorXd dthetalist(3);
	dthetalist << 0.1, 0.2, 0.3;
	Eigen::VectorXd ddthetalist(3);
	ddthetalist << 2, 1.5, 1;
	Eigen::VectorXd g(3);
	g << 0, 0, -9.8;

    Eigen::MatrixXd FlistT(3, 6);
    FlistT << 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0,
             -1, -1, -1, -1, -1, -1;
    Eigen::MatrixXd Flist = FlistT.transpose();
    
    Eigen::VectorXd Ftip(6);
    Ftip << 1, 1, 1, 1, 1, 1;

	std::vector<Eigen::MatrixXd> Mlist;
	std::vector<Eigen::MatrixXd> Glist;

	Eigen::Matrix4d M01;
	M01 << 1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0.089159,
		0, 0, 0, 1;
	Eigen::Matrix4d M12;
	M12 << 0, 0, 1, 0.28,
		0, 1, 0, 0.13585,
		-1, 0, 0, 0,
		0, 0, 0, 1;
	Eigen::Matrix4d M23;
	M23 << 1, 0, 0, 0,
		0, 1, 0, -0.1197,
		0, 0, 1, 0.395,
		0, 0, 0, 1;
	Eigen::Matrix4d M34; // link3 to end effector
	M34 << 1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1;

	Mlist.push_back(M01);
	Mlist.push_back(M12);
	Mlist.push_back(M23);
	Mlist.push_back(M34);

	Eigen::VectorXd G1(6);
	G1 << 0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7;
	Eigen::VectorXd G2(6);
	G2 << 0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393;
	Eigen::VectorXd G3(6);
	G3 << 0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275;

	Glist.push_back(G1.asDiagonal());
	Glist.push_back(G2.asDiagonal());
	Glist.push_back(G3.asDiagonal());

	Eigen::MatrixXd SlistT(3, 6);
	SlistT << 1, 0, 1, 0, 1, 0,
		0, 1, 0, -0.089, 0, 0,
		0, 1, 0, -0.089, 0, 0.425;
	Eigen::MatrixXd Slist = SlistT.transpose();
    
	Eigen::VectorXd Mytaulist = mr::myInverseDynamics(thetalist, dthetalist, ddthetalist, g,
		Flist, Mlist, Glist, Slist);

    Eigen::VectorXd taulist = mr::InverseDynamics(thetalist, dthetalist, ddthetalist, g,
		Ftip, Mlist, Glist, Slist);

    // std::cout   <<Mytaulist<<std::endl
    //             <<taulist<<std::endl;
	ASSERT_TRUE(Mytaulist.isApprox(taulist, 1e-4));
}