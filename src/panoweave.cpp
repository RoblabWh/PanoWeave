#include "panoweave.hpp"

#include <fstream>
#include <cereal/archives/json.hpp>
#include <spdlog/spdlog.h>
#include <opencv2/core/affine.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/ocl/core.hpp>
#include <opencv2/gapi/ocl/imgproc.hpp>

#include "basalt/serialization/headers_serialization.h"
#include "basalt/camera/generic_camera.hpp"


namespace PanoWeave
{
    using CvAffine3T = cv::Affine3<ScalarT>;

    PanoWeave::PanoWeave(const std::string &filepath)
    {
        spdlog::trace("PanoWeave::PanoWeave(const std::string &)");
        this->loadCalibration(filepath);
    }

    PanoWeave::PanoWeave(const std::string &filepath, ScalarT fov_x, ScalarT fov_y, ScalarT depth)
        : fov_x(fov_x), fov_y(fov_y), depth_static(depth)
    {
        spdlog::trace("PanoWeave::PanoWeave(const std::string &, ScalarT, ScalarT, ScalarT)");
        this->loadCalibration(filepath);
        this->buildInternals();
    }

    PanoWeave::PanoWeave(const std::string &filepath, ScalarT fov_x, ScalarT fov_y, const cv::Mat &depth)
        : fov_x(fov_x), fov_y(fov_y), depth_dynamic(depth)
    {
        spdlog::trace("PanoWeave::PanoWeave(const std::string &, ScalarT, ScalarT, const cv::Mat &)");
        this->loadCalibration(filepath);
        this->buildInternals();
    }

    void PanoWeave::loadCalibration(const std::string &filepath)
    {
        spdlog::trace("PanoWeave::loadCalibration(const std::string &)");
        std::ifstream file(filepath);
        cereal::JSONInputArchive archive(file);
        basalt::Calibration<double> calib;
        archive(calib);
        this->calib = calib.cast<ScalarT>();

        this->buildVignettes();

        this->rebuild = true;
    }

    void PanoWeave::setDepth(ScalarT depth)
    {
        spdlog::trace("PanoWeave::setDepth(ScalarT)");
        this->depth_static = depth;
        this->depth_dynamic.release();
        this->rebuild = true;
    }

    void PanoWeave::setDepth(const cv::Mat &depth)
    {
        spdlog::trace("PanoWeave::setDepth(const cv::Mat &)");
        this->depth_static = 0.0;
        this->depth_dynamic = depth;
        this->rebuild = true;
    }

    void PanoWeave::setFov(ScalarT fov_x, ScalarT fov_y)
    {
        spdlog::trace("PanoWeave::setFov(ScalarT, ScalarT)");
        this->fov_x = fov_x;
        this->fov_y = fov_y;
        this->rebuild = true;
    }

    ScalarT PanoWeave::fovX() const
    {
        return this->fov_x;
    }
    ScalarT PanoWeave::fovX(ScalarT fov)
    {
        this->setFov(fov, this->fov_y);
        return fov;
    }
    ScalarT PanoWeave::fovY() const
    {
        return this->fov_y;
    }
    ScalarT PanoWeave::fovY(ScalarT fov)
    {
        this->setFov(this->fov_x, fov);
        return fov;
    }

    void PanoWeave::setResolution(const cv::Size &resolution)
    {
        spdlog::trace("PanoWeave::setResolution(const cv::Size &)");
        this->res = resolution;
        this->rebuild = true;
    }

    void PanoWeave::setResolution(int width, int height)
    {
        spdlog::trace("PanoWeave::setResolution(int, int)");
        this->setResolution(cv::Size(width, height));
    }

    int PanoWeave::width() const
    {
        return this->res.width;
    }
    int PanoWeave::width(int width)
    {
        this->setResolution(width, this->res.height);
        return width;
    }
    int PanoWeave::height() const
    {
        return this->res.height;
    }
    int PanoWeave::height(int height)
    {
        this->setResolution(this->res.width, height);
        return height;
    }

    void PanoWeave::setVignetteThreshold(ScalarT threshold)
    {
        this->vign_thresh = threshold;
        this->rebuild = true;
    }

    ScalarT PanoWeave::vignetteThreshold() const
    {
        return this->vign_thresh;
    }
    ScalarT PanoWeave::vignetteThreshold(ScalarT thr)
    {
        this->setVignetteThreshold(thr);
        return thr;
    }



    void PanoWeave::weave(const std::vector<cv::Mat> &images, cv::Mat &pano)
    {
        spdlog::trace("PanoWeave::weave(const std::vector<cv::Mat> &, cv::Mat &)");
        if (this->buildInternals())
        {
            pano.create(this->res, CV_8UC3);
            std::vector<cv::Mat> wrapper(1);
            wrapper[0] = pano;
            this->graph(images, wrapper);
        }
        else
        {
            spdlog::warn("weave(): Unable to stitch image, failed building internals");
        }
    }

    void PanoWeave::weave(const std::vector<cv::Mat> &images, ScalarT depth, cv::Mat &pano)
    {
        spdlog::trace("PanoWeave::weave(const std::vector<cv::Mat> &, ScalarT, cv::Mat &)");
        this->setDepth(depth);
        this->weave(images, pano);
    }

    void PanoWeave::weave(const std::vector<cv::Mat> &images, const cv::Mat &depth, cv::Mat &pano)
    {
        spdlog::trace("PanoWeave::weave(const std::vector<cv::Mat> &, const cv::Mat &, cv::Mat &)");
        this->setDepth(depth);
        this->weave(images, pano);
    }


    bool PanoWeave::buildInternals()
    {
        spdlog::trace("PanoWeave::buildInternals()");
        if (!this->rebuild)
            return true;

        if (!this->buildMaps())
        {
            spdlog::warn("buildInternals(): failed building maps");
            return false;
        }
        //TODO do smarter stuff here if this stays (dont always rebuild maps and vignettes if only one has changed)
        this->buildVignettes();
        // if (!this->buildVignettes())
        // {
        //     spdlog::warn("buildInternals(): failed building vignettes");
        //     return false;
        // }

        this->buildGraph();
        return true;
    }

    void createSphericalPoints3(const cv::Size &res, ScalarT fov_x, ScalarT fov_y, cv::Mat &rays)
    {
        spdlog::trace("createSphericalPoints3(const cv::Size &, ScalarT, ScalarT, cv::Mat &)");
        // rays.create(res, CvMatT(3));
        CV_Assert(rays.type() == CvMatT(3) && rays.size() == res);

        const ScalarT width_2 = res.width / 2.0;
        const ScalarT height_2 = res.height / 2.0;

        rays.forEach<CvPoint3T>([&](CvPoint3T &point, const int *pos) -> void
                                {
                                    const int u = pos[1];
                                    const int v = pos[0];

                                    const ScalarT x = (u - width_2) / res.width * fov_x + M_PI_2;
                                    const ScalarT y = (v - height_2) / res.height * fov_y;

                                    // Calculate X, Y, and Z components of spherical rays
                                    point.x = -std::cos(y) * std::cos(x); // x
                                    point.y = std::sin(y);                // y
                                    point.z = std::cos(y) * std::sin(x);  // z
                                });
    }

    void transformSphericalPoints3(const cv::Mat &points_in, const CvAffine3T &transform, cv::Mat &points_out)
    {
        spdlog::trace("transformSphericalPoints3(const cv::Mat &, const CvAffine3T &, cv::Mat &)");
        if (&points_in == &points_out)
        {
            points_out.forEach<CvPoint3T>([&](CvPoint3T &point, const int *pos) -> void
                                          { point = transform * point; });
        }
        else
        {
            // points_out.create(points_in.size(), points_in.type());
            CV_Assert(points_out.type() == points_in.type() && points_out.size() == points_in.size());
            points_out.forEach<CvPoint3T>([&](CvPoint3T &point, const int *pos) -> void
                                          { point = transform * points_in.at<CvPoint3T>(pos); });
        }
    }

    template <typename T>
    void applyDepth(const cv::Mat &in, const T &depth, cv::Mat &out)
    {
        spdlog::error("applyDepth(): Called with invalid depth type.");
        CV_Assert(false);
    }
    template <>
    void applyDepth<ScalarT>(const cv::Mat &in, const ScalarT &depth, cv::Mat &out)
    {
        spdlog::trace("applyDepth(const cv::Mat &, const ScalarT &, cv::Mat &)");
        // CV_Assert(false);
        cv::multiply(in, depth, out);
    }
    template <>
    void applyDepth<cv::Mat>(const cv::Mat &in, const cv::Mat &depth, cv::Mat &out)
    {
        spdlog::trace("applyDepth(const cv::Mat &, const cv::Mat &, cv::Mat &)");
        CV_Assert(in.size() == depth.size());
        CV_Assert(in.type() == CvMatT(3));
        CV_Assert(depth.type() == CvMatT(1));
        if (&in != &out)
        {
            // out.create(in.size(), in.type());
            CV_Assert(in.size() == out.size() && in.type() == out.type());
        }
        out.forEach<CvPoint3T>([&](CvPoint3T &point, const int *pos) -> void
                               { point = in.at<CvPoint3T>(pos) * depth.at<ScalarT>(pos); });
    }

    template <typename T>
    void buildMapsVariable(EigenAlignedCvMat<3> &rays, const basalt::Calibration<ScalarT> calibration, const T &depth, std::vector<EigenAlignedCvMat<2>> &maps)
    {
        spdlog::trace("buildMapsVariable(EigenAlignedCvMat<3> &, const basalt::Calibration<ScalarT>, const T &, std::vector<EigenAlignedCvMat<2>> &)");
        maps.clear();
        const uint8_t num_cams = calibration.intrinsics.size();
        maps.reserve(num_cams);

        applyDepth(rays, depth, rays);

        for (uint8_t cam_idx = 0; cam_idx < num_cams; ++cam_idx)
        {
            std::vector<bool> useless_return_data;
            maps.emplace_back(static_cast<cv::Mat>(rays).size());
            cv::Mat &map = maps[cam_idx];
            //TODO fix? (T_i_c)
            calibration.intrinsics[cam_idx].project(rays, calibration.T_i_c[cam_idx].matrix().inverse(), maps[cam_idx], useless_return_data);

            //TODO implement the loop for projection myself and dodge this?
            map.forEach<CvPoint2T>([&](auto &val, auto pos) -> void {
                if (!useless_return_data[pos[0] * map.cols + pos[1]])
                {
                    val.x = 0;
                    val.y = 0;
                }
            });
        }
    }

    bool PanoWeave::buildMaps()
    {
        spdlog::trace("PanoWeave::buildMaps()");
        if (!this->rebuild)
        {
            return true;
        }

        if (this->res.empty())
        {
            spdlog::warn("buildMap(): resolution is empty");
            return false;
        }

        EigenAlignedCvMat<3> rays(this->res);
        createSphericalPoints3(this->res, this->fov_x, this->fov_y, rays);

        // TODO global transform?
        CvAffine3T global_transform;
        transformSphericalPoints3(rays, global_transform, rays);

        if (this->depth_static > 0.0)
        {
            buildMapsVariable(rays, this->calib, this->depth_static, this->maps);
        }
        else if (!this->depth_dynamic.empty() && this->depth_dynamic.type() == CvMatT(1))
        {
            buildMapsVariable(rays, this->calib, this->depth_dynamic, this->maps);
        }
        else
        {
            spdlog::warn("buildMap(): depth is empty");
            return false;
        }

        this->rebuild = false;
        return true;
    }

    void PanoWeave::buildGraph()
    {
        spdlog::trace("PanoWeave::buildGraph()");
        const uint8_t num_cams = this->calib.resolution.size();

        std::vector<cv::Mat> inputs;
        inputs.reserve(num_cams);
        for (const auto &res : this->calib.resolution)
        {
            inputs.emplace_back(res[1], res[0], CV_8UC3);
        }
        std::vector<cv::Mat> output;
        output.reserve(1);
        output.emplace_back(this->res, CV_8UC3);

        cv::GMetaArgs input_descr(num_cams);
        for (uint8_t cam_idx = 0; cam_idx < num_cams; ++cam_idx)
        {
            input_descr[cam_idx] = cv::descr_of(inputs[cam_idx]);
        }
        cv::GMetaArgs output_descr(1);
        output_descr[0] = cv::descr_of(output[0]);


        std::vector<cv::GMat> g_inputs(num_cams);
        std::vector<cv::GMat> pano(1);

        cv::GMat pano_b(cv::Mat::zeros(this->res, CvMatT(1)));
        cv::GMat pano_g(cv::Mat::zeros(this->res, CvMatT(1)));
        cv::GMat pano_r(cv::Mat::zeros(this->res, CvMatT(1)));
        cv::GMat count(cv::Mat::zeros(this->res, CvMatT(1)));


        for (uint8_t cam_idx = 0; cam_idx < num_cams; ++cam_idx)
        {
            auto input_split = cv::gapi::split3(g_inputs[cam_idx]);
            cv::GMat input_b = cv::gapi::convertTo(std::get<0>(input_split), CvMatT(1));
            cv::GMat input_g = cv::gapi::convertTo(std::get<1>(input_split), CvMatT(1));
            cv::GMat input_r = cv::gapi::convertTo(std::get<2>(input_split), CvMatT(1));
            cv::GMat input_corr_b = cv::gapi::div(input_b, cv::GMat(this->vigns[cam_idx]), 1.0);
            cv::GMat input_corr_g = cv::gapi::div(input_g, cv::GMat(this->vigns[cam_idx]), 1.0);
            cv::GMat input_corr_r = cv::gapi::div(input_r, cv::GMat(this->vigns[cam_idx]), 1.0);

            cv::GMat pano_part_b = cv::gapi::remap(input_corr_b, this->maps[cam_idx], cv::Mat(), cv::INTER_NEAREST);
            cv::GMat pano_part_g = cv::gapi::remap(input_corr_g, this->maps[cam_idx], cv::Mat(), cv::INTER_NEAREST);
            cv::GMat pano_part_r = cv::gapi::remap(input_corr_r, this->maps[cam_idx], cv::Mat(), cv::INTER_NEAREST);
            cv::GMat mask_part = cv::gapi::remap(cv::GMat(this->masks[cam_idx]), this->maps[cam_idx], cv::Mat(), cv::INTER_NEAREST);

            pano_b = cv::gapi::add(pano_b, pano_part_b, CvMatT(1));
            pano_g = cv::gapi::add(pano_g, pano_part_g, CvMatT(1));
            pano_r = cv::gapi::add(pano_r, pano_part_r, CvMatT(1));

            cv::GMat count_part = cv::gapi::mask(cv::GMat(cv::Mat::ones(this->res, CvMatT(1))), mask_part);
            count = cv::gapi::add(count, count_part, CvMatT(1));
        }

        pano_b = cv::gapi::div(pano_b, count, 1.0);
        pano_g = cv::gapi::div(pano_g, count, 1.0);
        pano_r = cv::gapi::div(pano_r, count, 1.0);

        pano_b = cv::gapi::convertTo(pano_b, CV_8UC1, 1.0, 0.5);
        pano_g = cv::gapi::convertTo(pano_g, CV_8UC1, 1.0, 0.5);
        pano_r = cv::gapi::convertTo(pano_r, CV_8UC1, 1.0, 0.5);

        pano[0] = cv::gapi::merge3(pano_b, pano_g, pano_r);

        cv::GComputation comp(g_inputs, pano);

        cv::GKernelPackage ocl_kernels = cv::gapi::combine(cv::gapi::core::ocl::kernels(), cv::gapi::imgproc::ocl::kernels());
        this->graph = comp.compile(std::move(input_descr), cv::compile_args(ocl_kernels));
    }

    void buildVignette(const basalt::RdSpline<1, 4, ScalarT> &vign_data, const ScalarT *oc, const cv::Size &res, ScalarT threshold, cv::Mat &vign, cv::Mat &mask)
    {
        spdlog::trace("buildVignette(const basalt::RdSpline<1, 4, ScalarT> &, const ScalarT *, const cv::Size &, ScalarT, cv::Mat &, cv::Mat &)");
        mask.create(res, CV_8UC1);
        CV_Assert(vign.size() == res);

        vign.forEach<ScalarT>([&](ScalarT &val, const int *pos) -> void
                            {
                                const int64_t loc = (EigenPoint2T(pos[1], pos[0]) - Eigen::Map<const EigenPoint2T>(oc)).norm() * 1e9;

                                val = vign_data.evaluate(loc)[0];
                                if (val < threshold)
                                    val = std::numeric_limits<ScalarT>::max();
                                else if (val > 1.0)
                                    val = 1.0;
                            });

        vign.convertTo(mask, CV_8UC1, 255, 0.5);
    }

    void PanoWeave::buildVignettes()
    {
        spdlog::trace("PanoWeave::buildVignettes()");
        this->vigns.clear();
        this->masks.clear();
        const uint8_t num_cams = this->calib.resolution.size();
        this->vigns.reserve(num_cams);
        this->masks.reserve(num_cams);

        for (uint8_t cam_idx = 0; cam_idx < num_cams; ++cam_idx)
        {
            //TODO test with aspect ratio != 1
            const Eigen::Vector2i &res_ = this->calib.resolution[cam_idx];
            const cv::Size res(res_[0], res_[1]);

            vigns.emplace_back(res);
            masks.emplace_back(res, CV_8UC1);

            buildVignette(this->calib.vignette[cam_idx], this->calib.intrinsics[cam_idx].getParam().data() + 2, res, this->vign_thresh, this->vigns[cam_idx], this->masks[cam_idx]);
        }
    }

}
