/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

 #include "UpdaterSchurVINS.hpp"

 #include "UpdaterHelper.h"
 
 #include "feat/Feature.h"
 #include "feat/FeatureInitializer.h"
 #include "state/State.h"
 #include "state/StateHelper.h"
 #include "types/LandmarkRepresentation.h"
 #include "utils/colors.h"
 #include "utils/print.h"
 #include "utils/quat_ops.h"
 
 #include <boost/date_time/posix_time/posix_time.hpp>
 #include <boost/math/distributions/chi_squared.hpp>
#include <cstdint>
 
 using namespace ov_core;
 using namespace ov_type;
 using namespace ov_msckf;
 
 UpdaterSchurVINS::UpdaterSchurVINS(UpdaterOptions &options, ov_core::FeatureInitializerOptions &feat_init_options) : _options(options) {
 
   // Save our raw pixel noise squared
   _options.sigma_pix_sq = std::pow(_options.sigma_pix, 2);
 
   // Save our feature initializer
   initializer_feat = std::shared_ptr<ov_core::FeatureInitializer>(new ov_core::FeatureInitializer(feat_init_options));
 
   // Initialize the chi squared test table with confidence level 0.95
   // https://github.com/KumarRobotics/msckf_vio/blob/050c50defa5a7fd9a04c1eed5687b405f02919b5/src/msckf_vio.cpp#L215-L221
   for (int i = 1; i < 500; i++) {
     boost::math::chi_squared chi_squared_dist(i);
     chi_squared_table[i] = boost::math::quantile(chi_squared_dist, 0.95);
   }
 }
 
 void UpdaterSchurVINS::update(std::shared_ptr<State> state, std::vector<std::shared_ptr<Feature>> &feature_vec) {
   // Return if no features
   if (feature_vec.empty()) {
    return;
   }
 
   // Start timing
   boost::posix_time::ptime rT0, rT1, rT2, rT3, rT4, rT5;
   rT0 = boost::posix_time::microsec_clock::local_time();
 
   // 0. Get all timestamps our clones are at (and thus valid measurement times)
   std::vector<double> clonetimes;
   for (const auto &clone_imu : state->_clones_IMU) {
     clonetimes.emplace_back(clone_imu.first);
   }
 
   // 1. Clean all feature measurements and make sure they all have valid clone times
   auto it0 = feature_vec.begin();
   while (it0 != feature_vec.end()) {
     // Clean the feature
     (*it0)->clean_old_measurements(clonetimes);
 
     // Count how many measurements
     int ct_meas = 0; // 在clones中追踪的次数
     for (const auto &pair : (*it0)->timestamps) {
       ct_meas += (*it0)->timestamps[pair.first].size();
     }
 
     // Remove if we don't have enough
     if (ct_meas < 2) {
       (*it0)->to_delete = true;
       it0 = feature_vec.erase(it0);
     } else {
       it0++;
     }
   }
   rT1 = boost::posix_time::microsec_clock::local_time();
 
   // 2. Create vector of cloned *CAMERA* poses at each of our clone timesteps
   std::unordered_map<size_t, std::unordered_map<double, FeatureInitializer::ClonePose>> clones_cam; // <相机id -> <IMU clone时间戳 -> IMU clone对应的相机位姿>>
   for (const auto &clone_calib : state->_calib_IMUtoCAM) {
 
     // For this camera, create the vector of camera poses
     std::unordered_map<double, FeatureInitializer::ClonePose> clones_cami; // <IMU clone时间戳 -> IMU clone对应的相机位姿>
     for (const auto &clone_imu : state->_clones_IMU) {
 
       // Get current camera pose
       Eigen::Matrix<double, 3, 3> R_GtoCi = clone_calib.second->Rot() * clone_imu.second->Rot();
       Eigen::Matrix<double, 3, 1> p_CioinG = clone_imu.second->pos() - R_GtoCi.transpose() * clone_calib.second->pos();
 
       // Append to our map
       clones_cami.insert({clone_imu.first, FeatureInitializer::ClonePose(R_GtoCi, p_CioinG)});
     }
 
     // Append to our map
     clones_cam.insert({clone_calib.first, clones_cami});
   }
 
   // 3. Try to triangulate all MSCKF or new SLAM features that have measurements
   auto it1 = feature_vec.begin();
   // 遍历每一个特征
   while (it1 != feature_vec.end()) {
 
     // Triangulate the feature and remove if it fails
     bool success_tri = true;
     if (initializer_feat->config().triangulate_1d) {
       success_tri = initializer_feat->single_triangulation_1d(*it1, clones_cam);
     } else {
       success_tri = initializer_feat->single_triangulation(*it1, clones_cam);
     }

     // Gauss-newton refine the feature
     bool success_refine = true;
     if (initializer_feat->config().refine_features) {
       success_refine = initializer_feat->single_gaussnewton(*it1, clones_cam);
     }
 
    //  Remove the feature if not a success
     if (!success_tri || !success_refine) {
       (*it1)->to_delete = true;
       it1 = feature_vec.erase(it1);
      //  PRINT_INFO("[SchurVINS] Feature %d failed to triangulate or refine Erased!\n", (*it1)->featid);
       continue;
     }
     it1++;
   }

   if(feature_vec.empty()) {
      PRINT_WARNING("[SchurVINS] No features to update after triangulation!\n");
      return;
   }
   rT2 = boost::posix_time::microsec_clock::local_time();
 
   // Calculate the max possible measurement size
   size_t max_meas_size = 0; // nFeatures * 2 * TrackLength
   for (size_t i = 0; i < feature_vec.size(); i++) {
     for (const auto &pair : feature_vec.at(i)->timestamps) {
       max_meas_size += 2 * feature_vec.at(i)->timestamps[pair.first].size();
     }
   }
 
   // Calculate max possible state size (i.e. the size of our covariance)
   // NOTE: that when we have the single inverse depth representations, those are only 1dof in size
   size_t max_hx_size = state->max_covariance_size();
   for (auto &landmark : state->_features_SLAM) {
     max_hx_size -= landmark.second->size();
   }
 
   // Large Jacobian and residual of *all* features for this update
   Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size);
   Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size);
   std::unordered_map<std::shared_ptr<Type>, size_t> Hx_mapping;
   std::vector<std::shared_ptr<Type>> Hx_order_big;
   size_t ct_jacob = 0;
   size_t ct_meas = 0;


   Matrix2o3d dr_dpc = Matrix2o3d::Zero();
   Matrix3o6d dpc_dpos = Matrix3o6d::Zero();
   Matrix2o6d jx = Matrix2o6d::Zero();
   Matrix2o3d jf = Matrix2o3d::Zero();
   Eigen::Vector2d r = Eigen::Vector2d::Zero();
 
   // 4. Compute linear system for each feature, nullspace project, and reject
   auto it2 = feature_vec.begin();
   std::vector<UpdaterHelper::UpdaterHelperFeature> landmarkVec;
   std::unordered_map<std::shared_ptr<Type>, size_t> map_hx;

   PRINT_INFO("[SchurVINS] Feature size: %d\n", feature_vec.size());
   for (size_t f = 0; f < feature_vec.size(); f++) {
    feature_vec[f]->to_delete = true;
  }
   std::vector<std::shared_ptr<Type>> Hx_order;

   auto iter = feature_vec.begin();

   int total_hx = 0;
   while (iter != feature_vec.end()) {
    auto& feature = **iter;
    for (auto const &pair : feature.timestamps) {
      for (size_t m = 0; m < feature.timestamps[pair.first].size(); m++) {
        const double measure_timestamp = feature.timestamps[pair.first].at(m);
        // Add this clone if it is not added already
        std::shared_ptr<PoseJPL> clone_Ci = state->_clones_IMU.at(measure_timestamp);
        if (map_hx.find(clone_Ci) == map_hx.end()) {
          map_hx.insert({clone_Ci, total_hx});
          Hx_order.push_back(clone_Ci);
          total_hx += clone_Ci->size();
        }
      }
    }
    iter++;
   }
   Eigen::MatrixXd Amtx = Eigen::MatrixXd::Zero(total_hx, total_hx);
   Eigen::VectorXd Bvct = Eigen::VectorXd::Zero(total_hx);


   while (it2 != feature_vec.end()) {
 
     // Convert our feature into our current format
     UpdaterHelper::UpdaterHelperFeature feat;
     feat.featid = (*it2)->featid;
     feat.uvs = (*it2)->uvs;
     feat.uvs_norm = (*it2)->uvs_norm;
     feat.timestamps = (*it2)->timestamps;
     feat.V.setZero();
     feat.V_inv.setZero();
     feat.gv.setZero();

     // If we are using single inverse depth, then it is equivalent to using the msckf inverse depth
     feat.feat_representation = state->_options.feat_rep_msckf;
     if (state->_options.feat_rep_msckf != LandmarkRepresentation::Representation::GLOBAL_3D) {
       // TODO: Finish other representation with SV
       PRINT_ERROR("Feature representation MUST be GLOBAL_3D in SchurVINS\n");
       PRINT_ERROR("Current Setting: %d\n", state->_options.feat_rep_msckf);
      //  feat.feat_representation = LandmarkRepresentation::Representation::ANCHORED_MSCKF_INVERSE_DEPTH;
       std::exit(0);
     }
 
     // Save the position and its fej value
      feat.p_FinG = (*it2)->p_FinG;
      feat.p_FinG_fej = (*it2)->p_FinG;
 
     int total_meas = 0;
     for (auto const &pair : feat.timestamps) {
       total_meas += (int)pair.second.size();
     }

    // Derivative of p_FinG in respect to feature representation.
    // This only needs to be computed once and thus we pull it out of the loop
    Eigen::MatrixXd dpfg_dlambda;
    std::vector<Eigen::MatrixXd> dpfg_dx;
    std::vector<std::shared_ptr<Type>> dpfg_dx_order;
    UpdaterHelper::get_feature_jacobian_representation(state, feat, dpfg_dlambda, dpfg_dx, dpfg_dx_order);
    
    auto& feature = feat;
     
    Eigen::Vector3d p_FinG = feature.p_FinG;
    // Loop through each camera for this feature
    for (auto const &pair : feature.timestamps) {

      // If extrinsics calib
      if (state->_options.do_calib_camera_pose) {
        PRINT_ERROR("[SchurVINS] Calibration of camera pose not implemented yet\n");
        return;
      }

      // If doing calibration intrinsics
      if (state->_options.do_calib_camera_intrinsics) {
        PRINT_ERROR("[SchurVINS] Calibration of camera intrinsics not implemented yet\n");
        return;
      }

      // Our calibration between the IMU and CAMi frames
      std::shared_ptr<Vec> distortion = state->_cam_intrinsics.at(pair.first);
      std::shared_ptr<PoseJPL> calibration = state->_calib_IMUtoCAM.at(pair.first);
      Eigen::Matrix3d R_ItoC = calibration->Rot();
      Eigen::Vector3d p_IinC = calibration->pos();
      feature.W[pair.first].clear();
      feature.W[pair.first].resize(feature.timestamps[pair.first].size(), Eigen::Matrix<double, 6, 3>::Zero());
      feature.res[pair.first].clear();
      feature.res[pair.first].resize(feature.timestamps[pair.first].size(), Eigen::Vector2d::Zero());
  
      // Loop through all measurements for this specific camera
      for (size_t m = 0; m < feature.timestamps[pair.first].size(); m++) {
        // Get current IMU clone state
        std::shared_ptr<PoseJPL> clone_Ii = state->_clones_IMU.at(feature.timestamps[pair.first].at(m));
        
        Eigen::Matrix3d R_GtoIi = clone_Ii->Rot();
        Eigen::Vector3d p_IiinG = clone_Ii->pos();
  
        // Get current feature in the IMU
        Eigen::Vector3d p_FinIi = R_GtoIi * (p_FinG - p_IiinG);
  
        // Project the current feature into the current frame of reference
        Eigen::Vector3d p_FinCi = R_ItoC * p_FinIi + p_IinC;
        Eigen::Vector2d uv_norm;
        uv_norm << p_FinCi(0) / p_FinCi(2), p_FinCi(1) / p_FinCi(2);
  
        // Distort the normalized coordinates (radtan or fisheye)
        Eigen::Vector2d uv_dist;
        uv_dist = state->_cam_intrinsics_cameras.at(pair.first)->distort_d(uv_norm);
  
        // Our residual
        Eigen::Vector2d uv_m;
        uv_m << (double)feature.uvs[pair.first].at(m)(0), (double)feature.uvs[pair.first].at(m)(1);
        r = uv_m - uv_dist;

        const double r_l2 = r.squaredNorm();
        const double obs_invdev = 0.25;
        const double huberA = 1.5;
        double huberB = huberA * huberA;
        double huber_scale = 1.0;
        if (r_l2 > huberB) {
            const double radius = sqrt(r_l2);
            double rho1 = std::max(std::numeric_limits<double>::min(), huberA / radius);
            huber_scale = sqrt(rho1);
            r *= huber_scale;
        }
        r *= obs_invdev;
        //=========================================================================
        //=========================================================================
  
        // If we are doing first estimate Jacobians, then overwrite with the first estimates
        if (state->_options.do_fej) {
          PRINT_ERROR("FEJ not implemented In SchurVINS yet\n");
          return;
        }
  
        // Compute Jacobians in respect to normalized image coordinates and possibly the camera intrinsics
        Eigen::MatrixXd dz_dzn, dz_dzeta;
        state->_cam_intrinsics_cameras.at(pair.first)->compute_distort_jacobian(uv_norm, dz_dzn, dz_dzeta);
  
        // Normalized coordinates in respect to projection function
        Eigen::MatrixXd dzn_dpfc = Eigen::MatrixXd::Zero(2, 3);
        const double pFc2 = (p_FinCi(2) * p_FinCi(2));
        dzn_dpfc(0, 0) = 1 / p_FinCi(2);
        dzn_dpfc(1, 1) = 1 / p_FinCi(2);
        dzn_dpfc(0, 2) = -p_FinCi(0) / pFc2;
        dzn_dpfc(1, 2) = -p_FinCi(1) / pFc2;
        dzn_dpfc *= obs_invdev * huber_scale;
  
        // Derivative of p_FinCi in respect to p_FinIi
        Eigen::MatrixXd dpfc_dpfg = R_ItoC * R_GtoIi;
  
        // Derivative of p_FinCi in respect to camera clone state
        Eigen::MatrixXd dpfc_dclone = Eigen::MatrixXd::Zero(3, 6);
        dpfc_dclone.block(0, 0, 3, 3).noalias() = R_ItoC * skew_x(p_FinIi);
        dpfc_dclone.block(0, 3, 3, 3) = -dpfc_dpfg;
  
        //=========================================================================
        //=========================================================================
  
        // Precompute some matrices
        Eigen::MatrixXd dz_dpfc = dz_dzn * dzn_dpfc;
  
        // CHAINRULE: get the total feature Jacobian
        jx.noalias() = dz_dpfc * dpfc_dclone;
        jf.noalias() = dz_dpfc * dpfc_dpfg;

        Amtx.block(map_hx[clone_Ii], map_hx[clone_Ii], 6,6).noalias() += jx.transpose() * jx;
        Bvct.segment(map_hx[clone_Ii], 6).noalias() += jx.transpose() * r;

        feature.V.noalias() += jf.transpose() * jf;
        feature.gv.noalias() += jf.transpose() * r;
        feature.W[pair.first].at(m) = jx.transpose() * jf;
  
        // Derivative of p_FinCi in respect to camera calibration (R_ItoC, p_IinC)
        if (state->_options.do_calib_camera_pose) {
          PRINT_ERROR("Camera extrinsics calibration not implemented in SchurVINS yet\n");
          return;
        }
  
        // Derivative of measurement in respect to distortion parameters
        if (state->_options.do_calib_camera_intrinsics) {
          PRINT_ERROR("Camera intrinsics calibration not implemented in SchurVINS yet\n");
          return;
        }
  
      }
    }
    landmarkVec.push_back(feature);
    it2++;
  }

  auto it3 = landmarkVec.begin();
  while(it3 != landmarkVec.end()) {
    auto &feature = *it3;
    feature.stack_W_map.clear();
    for (auto const &pair : feature.timestamps) {
      for (size_t i = 0; i < feature.W[pair.first].size(); i++) {
        std::shared_ptr<PoseJPL> clone_Ii = state->_clones_IMU.at(feature.timestamps[pair.first].at(i));
        int stateIdx = clone_Ii->id();
        auto iter = feature.stack_W_map.find(stateIdx);
        if (iter == feature.stack_W_map.end()) {
          feature.stack_W_map.emplace(stateIdx, feature.W[pair.first].at(i));
        } else {
          PRINT_INFO("Stacking Extra W for Landmark %d, state %d\n", feature.featid, stateIdx);
          iter->second += feature.W[pair.first].at(i);
        }
      }
    }
    it3++;
  }

  // Loop 3D Landmark
  for(size_t i = 0; i < landmarkVec.size(); i++) {
    auto &feature = landmarkVec[i];
    feature.V_inv = feature.V.inverse();
    // Loop each camera
    for (auto const &pair : feature.timestamps) {
      for(size_t m = 0; m < feature.timestamps[pair.first].size(); m++) {
        std::shared_ptr<PoseJPL> clone_Im = state->_clones_IMU.at(feature.timestamps[pair.first].at(m));
        if (feature.stack_W_map.find(clone_Im->id()) == feature.stack_W_map.end()) {
          PRINT_ERROR(RED "StateHelper::SV_EKFUpdate() - stack_W_map does not contain clone_Im!\n" RESET);
          std::exit(EXIT_FAILURE);
        }
        Eigen::Matrix<double, 6, 3> WVinv = feature.stack_W_map.at(clone_Im->id()) * feature.V_inv;
        for(size_t n = m; n < feature.timestamps[pair.first].size(); n++) {
          std::shared_ptr<PoseJPL> clone_In = state->_clones_IMU.at(feature.timestamps[pair.first].at(n));
          Eigen::Matrix<double, 6, 6> schurComplemnt = WVinv * feature.stack_W_map.at(clone_In->id()).transpose();
          Amtx.block(map_hx[clone_Im], map_hx[clone_In], 6,6).noalias() -= schurComplemnt;
          // sym
          if(m != n) {
            Amtx.block(map_hx[clone_In], map_hx[clone_Im], 6,6).noalias() -= schurComplemnt.transpose();
          }
        }
        Bvct.segment(map_hx[clone_Im], 6).noalias() -= WVinv * feature.gv;
      }
    }
  }

   rT3 = boost::posix_time::microsec_clock::local_time();
 
   // We have appended all features to our Hx_big, res_big
   // Delete it so we do not reuse information
   for (size_t f = 0; f < feature_vec.size(); f++) {
     feature_vec[f]->to_delete = true;
   }
 
   rT4 = boost::posix_time::microsec_clock::local_time();
 
   // Our noise is isotropic, so make it here after our compression
  //  Eigen::MatrixXd R_big = _options.sigma_pix_sq * Amtx;
   Eigen::MatrixXd R_big =  _options.sigma_pix_sq * Eigen::MatrixXd::Identity(Amtx.rows(), Amtx.cols());

   // check if matrix is empty
  if (R_big.rows() == 0 || R_big.cols() == 0) {
    PRINT_ERROR("[SchurVINS] R_big is empty!\n");
    return;
  }

   StateHelper::SV_EKFUpdate(state, Hx_order, Amtx, Bvct, R_big);

   // 6. With all good features update the state
  //  StateHelper::EKFUpdate(state, Hx_order_big, Hx_big, res_big, R_big);
   rT5 = boost::posix_time::microsec_clock::local_time();
 
   // Debug print timing information
  //  PRINT_ALL("[MSCKF-UP]: %.4f seconds to clean\n", (rT1 - rT0).total_microseconds() * 1e-6);
  //  PRINT_ALL("[MSCKF-UP]: %.4f seconds to triangulate\n", (rT2 - rT1).total_microseconds() * 1e-6);
  //  PRINT_ALL("[MSCKF-UP]: %.4f seconds create system (%d features)\n", (rT3 - rT2).total_microseconds() * 1e-6, (int)feature_vec.size());
  //  PRINT_ALL("[MSCKF-UP]: %.4f seconds compress system\n", (rT4 - rT3).total_microseconds() * 1e-6);
  //  PRINT_ALL("[MSCKF-UP]: %.4f seconds update state (%d size)\n", (rT5 - rT4).total_microseconds() * 1e-6, (int)res_big.rows());
  //  PRINT_ALL("[MSCKF-UP]: %.4f seconds total\n", (rT5 - rT1).total_microseconds() * 1e-6);
 }
