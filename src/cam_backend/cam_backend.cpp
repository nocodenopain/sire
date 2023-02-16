#include "sire/cam_backend/cam_backend.hpp"

#include <cmath>

#include <hpp/fcl/broadphase/broadphase_dynamic_AABB_tree.h>
#include <hpp/fcl/distance.h>
#include <hpp/fcl/math/transform.h>
#include <hpp/fcl/mesh_loader/assimp.h>
#include <hpp/fcl/mesh_loader/loader.h>
#include <hpp/fcl/shape/geometric_shapes.h>

#include <aris/core/serialization.hpp>
#include <aris/dynamic/model.hpp>
#include <aris/server/control_server.hpp>

#include "sire/collision/collision_filter.hpp"
#include "sire/middleware/sire_middleware.hpp"

namespace sire::cam_backend {
auto mapAngleToSymRange(double angle, double range) -> double {
  bool is_negative = false;
  range = std::abs(range);
  if (angle < 0) {
    is_negative = true;
    angle = -angle;
  }
  double temp = std::fmod((angle - range), 2 * range) - range;
  return is_negative ? -temp : temp;
}

auto vectorCross(double* in_1, double* in_2, double* out) -> void {
  out[0] = in_1[1] * in_2[2] - in_2[1] * in_1[2];
  out[1] = -in_1[0] * in_2[2] + in_2[0] * in_1[2];
  out[2] = in_1[0] * in_2[1] - in_2[0] * in_1[1];
  return;
}

auto vectorNormalize(aris::Size n, double* in) -> void {
  double temp = aris::dynamic::s_norm(n, in);
  for (aris::Size i = 0; i < n; ++i) {
    in[i] /= temp;
  }
  return;
}
auto xyz2pm(double* x, double* y, double* z, double* out) -> void {
  out[0] = x[0], out[1] = y[0], out[2] = z[0];
  out[4] = x[1], out[5] = y[1], out[6] = z[1];
  out[8] = x[2], out[9] = y[2], out[10] = z[2];
  out[15] = 1;
  return;
}

auto tiltAngle2pm(double side_tilt_angle, double forward_tilt_angle,
                  double* out) -> void {
  std::array<double, 3> x_vec{1, 0, 0};
  std::array<double, 3> y_vec{0, 1, 0};
  std::array<double, 3> z_vec{0, 0, 1};
  bool side_othogonal, forward_othogonal;
  bool side_negative = side_tilt_angle < 0;
  bool forward_negative = forward_tilt_angle < 0;
  if (aris::dynamic::s_is_equal(std::abs(side_tilt_angle), aris::PI / 2.0,
                                1e-2)) {
    side_tilt_angle = (side_negative) ? -1.57079632679 : 1.57079632679;
    side_othogonal = true;
  }
  if (aris::dynamic::s_is_equal(std::abs(forward_tilt_angle), aris::PI / 2.0,
                                1e-2)) {
    forward_tilt_angle = (forward_negative) ? -1.57079632679 : 1.57079632679;
    forward_othogonal = true;
  }
  // 如果侧倾和前倾角都垂直，就保留侧倾，不管前倾角
  if (side_othogonal && forward_othogonal) {
    y_vec = {0, 0, (side_negative) ? -1.0 : 1.0};
    z_vec = {0, (side_negative) ? 1.0 : -1.0, 0};
  } else if (side_othogonal && !forward_othogonal) {
    y_vec = {0, 0, (side_negative) ? -1.0 : 1.0};
    if (side_negative) {
      x_vec = {std::cos(forward_tilt_angle), -std::sin(forward_tilt_angle), 0};
      z_vec = {std::sin(forward_tilt_angle), std::cos(forward_tilt_angle), 0};
    } else {
      x_vec = {std::cos(forward_tilt_angle), std::sin(forward_tilt_angle), 0};
      z_vec = {std::sin(forward_tilt_angle), -std::cos(forward_tilt_angle), 0};
    }
  } else if (!side_othogonal && forward_othogonal) {
    x_vec = {0, 0, (forward_negative) ? 1.0 : -1.0};
    if (forward_negative) {
      y_vec = {-std::sin(forward_tilt_angle), std::cos(forward_tilt_angle), 0};
      z_vec = {-std::cos(forward_tilt_angle), -std::sin(forward_tilt_angle), 0};
    } else {
      y_vec = {std::sin(forward_tilt_angle), std::cos(forward_tilt_angle), 0};
      z_vec = {std::cos(forward_tilt_angle), -std::sin(forward_tilt_angle), 0};
    }
  } else {
    z_vec = {std::tan(forward_tilt_angle), std::tan(side_tilt_angle), 1};
    x_vec = {1, 0 - tan(forward_tilt_angle)};
    vectorCross(z_vec.data(), x_vec.data(), y_vec.data());
    vectorNormalize(3, z_vec.data());
    vectorNormalize(3, x_vec.data());
    vectorNormalize(3, y_vec.data());
  }
  xyz2pm(x_vec.data(), y_vec.data(), z_vec.data(), out);
}

struct CamBackend::Imp {
  vector<bool> collision_result_;
  vector<set<std::pair<collision::geometry::GeometryId,
                       collision::geometry::GeometryId>>>
      collided_objects_result_;

  shared_ptr<aris::dynamic::Model> robot_model_ptr_;
  unique_ptr<collision::CollisionEngine> collision_engine_ptr_;
};

auto CamBackend::cptCollisionByEEPose(
    double* ee_pe, collision::CollidedObjectsCallback& callback) -> void {
  // aris generalMotion末端默认表示是Euler321 ZYX
  imp_->robot_model_ptr_->setOutputPos(ee_pe);
  if (imp_->robot_model_ptr_->inverseKinematics()) return;
  imp_->robot_model_ptr_->forwardKinematics();
  aris::Size partSize = imp_->robot_model_ptr_->partPool().size();
  vector<double> part_pq(partSize * 7);
  for (aris::Size i = 0; i < partSize; ++i) {
    imp_->robot_model_ptr_->partPool().at(i).getPq(part_pq.data() + i * 7);
  }
  imp_->collision_engine_ptr_->updateLocation(part_pq.data());
  imp_->collision_engine_ptr_->hasCollisions(callback);
}

auto CamBackend::cptEEPose(WobjToolInstallMethod install_method, int cpt_option,
                           double angle, double* tool_path_point_pm,
                           double tool_axis_angle, double side_tilt_angle,
                           double forward_tilt_angle, double* target_ee_pe)
    -> void {
  tool_axis_angle = mapAngleToSymRange(tool_axis_angle, aris::PI);
  side_tilt_angle = mapAngleToSymRange(side_tilt_angle, aris::PI / 2.0);
  forward_tilt_angle = mapAngleToSymRange(forward_tilt_angle, aris::PI / 2.0);
  if (install_method == WobjToolInstallMethod::EX_WOBJ_HAND_TOOL) {
    // AxisA6 collision map
    if (cpt_option == 0) {
      tool_axis_angle = angle - aris::PI;
    } else if (cpt_option == 1) {
      // 1. 根据前倾旋转加工点坐标系（normal tangent)
      angle = angle - aris::PI / 2.0;
      angle = mapAngleToSymRange(angle, aris::PI / 2.0);
      side_tilt_angle = angle;
    }
  } else if (install_method == WobjToolInstallMethod::HAND_WOBJ_EX_TOOL) {
    if (cpt_option == 0) {
    } else if (cpt_option == 1) {
    }
  }

  // 最终需要知道的3个角度，前倾、侧倾、axisA6的转动角度，就可以计算末端位置
  // 1. 根据前倾侧倾旋转加工点坐标系（normal tangent)
  double tilt_pm[16];
  tiltAngle2pm(side_tilt_angle, forward_tilt_angle, tilt_pm);
  double ee_pm[16];
  aris::dynamic::s_pm_dot_pm(tool_path_point_pm, tilt_pm, ee_pm);
  // 放弃使用欧拉角计算前倾侧倾的方法，因为没有办法同时保证前倾角和侧倾角不变
  // double tilt_re[3]{0.0, forward_tilt_angle / 180.0 * aris::PI,
  //                   side_tilt_angle / 180.0 * aris::PI};  // 321
  // double tilt_pm[16];
  // aris::dynamic::s_re2pm(tilt_re, tilt_pm, "321");
  // double ee_pm[16];
  // aris::dynamic::s_pm_dot_pm(tilt_pm, tool_path_point_pm, ee_pm);

  // 2. 得到刀具空间位姿
  ee_pm[1] = -ee_pm[1];
  ee_pm[2] = -ee_pm[2];
  ee_pm[5] = -ee_pm[5];
  ee_pm[6] = -ee_pm[6];
  ee_pm[8] = -ee_pm[8];
  ee_pm[9] = -ee_pm[9];
  // 3. 旋转对应的axisa6的角度
  double re[3]{tool_axis_angle, 0.0, 0.0};  // 313
  double rotate_ee_z_pm[16];
  aris::dynamic::s_re2pm(re, rotate_ee_z_pm);
  double target_ee_pm[16];
  aris::dynamic::s_pm_dot_pm(ee_pm, rotate_ee_z_pm, target_ee_pm);
  aris::dynamic::s_pm2pe(target_ee_pm, target_ee_pe, "321");
}

// initial CAM backend by two config file
auto CamBackend::init(string model_xml_path, string collision_xml_path)
    -> void {
  auto config_path =
      std::filesystem::absolute(".");  // 获取当前可执行文件所在的路径
  const string model_config_name = "cam_model.xml";
  auto model_config_path = config_path / model_config_name;
  imp_->robot_model_ptr_.reset(new aris::dynamic::Model());
  aris::core::fromXmlFile(&(imp_->robot_model_ptr_), model_config_path);

  const string collision_config_name = "collision_calculator.xml";
  auto collision_config_path = config_path / collision_config_name;
  imp_->collision_engine_ptr_.reset(new collision::CollisionEngine());
  aris::core::fromXmlFile(&(imp_->collision_engine_ptr_),
                          collision_config_path);

  imp_->robot_model_ptr_->init();
  imp_->collision_engine_ptr_->init();
}

// initial CAM backend by control server
auto CamBackend::init() -> void {
  imp_->robot_model_ptr_.reset(&dynamic_cast<aris::dynamic::Model&>(
      aris::server::ControlServer::instance().model()));
  imp_->collision_engine_ptr_->init();
}

// 未考虑周全的问题：
// 1. 多冗余轴情况：七轴机械臂会有几个冗余轴？两个，但是怎么表示怎么计算？
// 2. Option的设置怎么更优雅
//
// TODO:
// 1. 外部轴
// 3. Option怎么更优雅
//
//  tool_z_vec在不设置侧倾的时候与normal一致，但是设置侧倾之后就不一样了，需要传入给AxisA6的旋转使用
//  单位 m
void CamBackend::cptCollisionMap(WobjToolInstallMethod install_method,
                                 int cpt_option, aris::Size resolution,
                                 aris::Size pSize, double* points,
                                 double* tool_axis_angles,
                                 double* side_tilt_angles,
                                 double* forward_tilt_angles, double* normal,
                                 double* tangent) {
  imp_->collision_result_.resize(resolution * pSize);
  imp_->collided_objects_result_.resize(resolution * pSize);
  double step_angle;
  if (cpt_option == 0) {
    step_angle = 2.0 * aris::PI / resolution;
  } else {
    step_angle = aris::PI / resolution;
  }
  for (aris::Size i = 0; i < resolution; ++i) {
    for (aris::Size j = 0; j < pSize; ++j) {
      // 1. 加工点坐标系( normal tangent)
      double* point = points + j * 3;
      double* forward_vec = tangent + j * 3;
      double* normal_vec = normal + j * 3;
      double tool_axis_angle = tool_axis_angles[j];
      double side_tilt_angle = side_tilt_angles[j];
      double forward_tilt_angle = forward_tilt_angles[j];
      double y_vec[3] = {
          normal_vec[1] * forward_vec[2] - forward_vec[1] * normal_vec[2],
          -normal_vec[0] * forward_vec[2] + forward_vec[0] * normal_vec[2],
          normal_vec[0] * forward_vec[1] - forward_vec[0] * normal_vec[1]};
      double tool_point_pm[16] = {forward_vec[0],
                                  y_vec[0],
                                  normal_vec[0],
                                  point[0],
                                  forward_vec[1],
                                  y_vec[1],
                                  normal_vec[1],
                                  point[1],
                                  forward_vec[2],
                                  y_vec[2],
                                  normal_vec[2],
                                  point[2],
                                  0.0,
                                  0.0,
                                  0.0,
                                  1.0};
      double target_angle = step_angle * i;
      double target_ee_pe[6];
      cptEEPose(install_method, cpt_option, target_angle, tool_point_pm,
                tool_axis_angle, side_tilt_angle, forward_tilt_angle,
                target_ee_pe);
      // 5. 设置末端位姿并反解
      collision::CollidedObjectsCallback callback(
          &imp_->collision_engine_ptr_->collisionFilter());
      cptCollisionByEEPose(target_ee_pe, callback);
      if (callback.collidedObjectMap().size() != 0) {
        imp_->collision_result_[i * pSize + j] = true;
        imp_->collided_objects_result_[i * pSize + j] =
            callback.collidedObjectMap();
      }
    }
  }
  return;
}

void CamBackend::cptCollisionMap(WobjToolInstallMethod install_method,
                                 int cpt_option, aris::Size resolution,
                                 aris::Size pSize, double* points_pm,
                                 double* tool_axis_angles,
                                 double* side_tilt_angles,
                                 double* forward_tilt_angles) {
  imp_->collision_result_.resize(resolution * pSize);
  imp_->collided_objects_result_.resize(resolution * pSize);
  double step_angle;
  if (cpt_option == 0) {
    step_angle = 2.0 * aris::PI / resolution;
  } else {
    step_angle = aris::PI / resolution;
  }
  for (aris::Size i = 0; i < resolution; ++i) {
    for (aris::Size j = 0; j < pSize; ++j) {
      // 1. 加工点坐标系( normal tangent)
      double forward_tilt_angle = forward_tilt_angles[j];
      double side_tilt_angle = side_tilt_angles[j];
      double tool_axis_angle = side_tilt_angles[j];
      double* point_pm = points_pm + j * 16;
      double target_angle = step_angle * i;
      double target_ee_pe[6];
      cptEEPose(install_method, cpt_option, target_angle, point_pm,
                tool_axis_angle, side_tilt_angle, forward_tilt_angle,
                target_ee_pe);
      // 5. 设置末端位姿并反解
      collision::CollidedObjectsCallback callback(
          &imp_->collision_engine_ptr_->collisionFilter());
      cptCollisionByEEPose(target_ee_pe, callback);
      if (callback.collidedObjectMap().size() != 0) {
        imp_->collision_result_[i * pSize + j] = true;
        imp_->collided_objects_result_[i * pSize + j] =
            callback.collidedObjectMap();
      }
    }
  }
  return;
}

auto CamBackend::getCollisionEngine() -> collision::CollisionEngine& {
  return *imp_->collision_engine_ptr_;
}
auto CamBackend::resetCollisionEngine(
    collision::CollisionEngine* engine) -> void {
  imp_->collision_engine_ptr_.reset(engine);
}

auto CamBackend::getCollisionMapResult() -> const vector<bool>& {
  return imp_->collision_result_;
}
auto CamBackend::getCollidedObjectsResult()
    -> const vector<set<std::pair<collision::geometry::GeometryId,
                                  collision::geometry::GeometryId>>>& {
  return imp_->collided_objects_result_;
}

CamBackend::CamBackend() : imp_(new Imp) {}
CamBackend::~CamBackend() = default;

ARIS_REGISTRATION {
  aris::core::class_<CamBackend>("SireCamBackend")
      .inherit<core::SireModuleBase>()
      .prop("collision_calculator", &CamBackend::resetCollisionEngine,
            &CamBackend::getCollisionEngine);
}
}  // namespace sire::cam_backend