#include "panoweave.hpp"

#include <fstream>
#include <cereal/archives/json.hpp>
#include <spdlog/spdlog.h>
#include <opencv2/core/affine.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>

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
        : depth_static(depth), fov_x(fov_x), fov_y(fov_y)
    {
        spdlog::trace("PanoWeave::PanoWeave(const std::string &, ScalarT, ScalarT, ScalarT)");
        this->loadCalibration(filepath);
        this->buildInternals();
    }

    PanoWeave::PanoWeave(const std::string &filepath, ScalarT fov_x, ScalarT fov_y, const cv::Mat &depth)
        : depth_dynamic(depth), fov_x(fov_x), fov_y(fov_y)
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

        this->build_maps = true;
        this->build_vigns = true;
    }

    void PanoWeave::setDepth(ScalarT depth)
    {
        spdlog::trace("PanoWeave::setDepth(ScalarT)");
        this->depth_static = depth;
        this->depth_dynamic.release();
        this->build_maps = true;
    }

    void PanoWeave::setDepth(const cv::Mat &depth)
    {
        spdlog::trace("PanoWeave::setDepth(const cv::Mat &)");
        this->depth_static = 0.0;
        this->depth_dynamic = depth;
        this->build_maps = true;
    }

    void PanoWeave::setFov(ScalarT fov_x, ScalarT fov_y)
    {
        spdlog::trace("PanoWeave::setFov(ScalarT, ScalarT)");
        this->fov_x = fov_x;
        this->fov_y = fov_y;
        this->build_maps = true;
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
        this->build_maps = true;
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
        this->build_vigns = true;
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


    bool correctResponse(const cv::Mat &in, const Eigen::VectorXf &inv_resp, cv::Mat &out)
    {
        const int cn = in.channels();
        out.create(in.size(), CvMatT(cn));

        if (cn == 1)
            out.forEach<ScalarT>([&](auto &val, const int *pos)
                                 { val = inv_resp[in.at<uchar>(pos)]; });
        else if (cn == 3)
            out.forEach<cv::Vec<ScalarT, 3>>([&](auto &val, const int *pos)
                                             { const auto &src = in.at<cv::Vec3b>(pos);
                                    val[0] = inv_resp[src[0]];
                                    val[1] = inv_resp[src[1]];
                                    val[2] = inv_resp[src[2]]; });
        else if (cn == 4)
            out.forEach<cv::Vec<ScalarT, 4>>([&](auto &val, const int *pos)
                                             { const auto &src = in.at<cv::Vec4b>(pos);
                                    val[0] = inv_resp[src[0]];
                                    val[1] = inv_resp[src[1]];
                                    val[2] = inv_resp[src[2]];
                                    val[4] = src[4]; });
        else // TODO this type of error handling needed?
        {
            spdlog::error("Can not correct response for {} channel images", cn);
            return false;
        }
        return true;
    }

    void PanoWeave::weave(const std::vector<cv::Mat> &images, cv::Mat &pano)
    {
        spdlog::trace("PanoWeave::weave(const std::vector<cv::Mat> &, cv::Mat &)");

        if (images.front().channels() != this->channels)
        {
            this->channels = images.front().channels();
            this->build_mirrors = true;
        }

        if (this->buildInternals())
        {
            const int cn = images.front().channels();
            pano = cv::Mat::zeros(this->res, CvMatT(cn));

            for (uint8_t i = 0; i < this->calib.intrinsics.size(); ++i)
            {
                const cv::Mat &img = images[i];

                cv::Mat undist;
                correctResponse(img, this->calib.response[i], undist);

                cv::multiply(undist, this->vigns[i], undist);

                cv::Mat remapped;
                cv::remap(undist, remapped, static_cast<cv::Mat>(this->maps[i]), cv::noArray(), cv::INTER_NEAREST);

                cv::add(remapped, pano, pano);
            }
            cv::multiply(pano, this->mask, pano);
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

        if (!this->buildMaps())
        {
            spdlog::warn("buildInternals(): failed building maps");
            return false;
        }
        this->buildVignettes();
        this->buildMask();
        this->buildMirrors();

        return true;
    }

    void createSphericalPoints3(const cv::Size &res, ScalarT fov_x, ScalarT fov_y, cv::Mat &rays)
    {
        spdlog::trace("createSphericalPoints3(const cv::Size &, ScalarT, ScalarT, cv::Mat &)");
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
                                    point.x = std::cos(y) * std::sin(x);  // x
                                    point.y = -std::cos(y) * std::cos(x); // y
                                    point.z = std::sin(y);                // z
                                });
    }

    void transformSphericalPoints3(const cv::Mat &points_in, const CvAffine3T &transform, cv::Mat &points_out)
    {
        spdlog::trace("transformSphericalPoints3(const cv::Mat &, const CvAffine3T &, cv::Mat &)");
        if (&points_in == &points_out)
        {
            points_out.forEach<CvPoint3T>([&](CvPoint3T &point, const int *pos) -> void
                                          { (void) pos;
                                             point = transform * point; });
        }
        else
        {
            CV_Assert(points_out.type() == points_in.type() && points_out.size() == points_in.size());
            points_out.forEach<CvPoint3T>([&](CvPoint3T &point, const int *pos) -> void
                                          { point = transform * points_in.at<CvPoint3T>(pos); });
        }
    }

    template <typename T>
    void applyDepth(const cv::Mat &in, const T &depth, cv::Mat &out)
    {
        (void)in;
        (void)depth;
        (void)out;
        spdlog::error("applyDepth(): Called with invalid depth type.");
        CV_Assert(false);
    }
    template <>
    void applyDepth<ScalarT>(const cv::Mat &in, const ScalarT &depth, cv::Mat &out)
    {
        spdlog::trace("applyDepth(const cv::Mat &, const ScalarT &, cv::Mat &)");
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
            CV_Assert(in.size() == out.size() && in.type() == out.type());
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
            std::vector<bool> success;
            maps.emplace_back(static_cast<cv::Mat>(rays).size());
            cv::Mat &map = maps[cam_idx];

            calibration.intrinsics[cam_idx].project(rays, calibration.T_i_c[cam_idx].matrix().inverse(), maps[cam_idx], success);

            // TODO implement the loop for projection myself and dodge this?
            map.forEach<CvPoint2T>([&](auto &val, auto pos) -> void
                                   {
                if (!success[pos[0] * map.cols + pos[1]])
                {
                    val.x = 0;
                    val.y = 0;
                } });
        }
    }

    bool PanoWeave::buildMaps()
    {
        spdlog::trace("PanoWeave::buildMaps()");
        if (!this->build_maps)
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

        this->build_mask = true;
        this->build_maps = false;
        return true;
    }

    void PanoWeave::buildVignettes()
    {
        spdlog::trace("PanoWeave::buildVignettes()");
        if (!this->build_vigns)
            return;

        auto vmaps = this->calib.vignette_maps();

        this->vigns_base.resize(vmaps.size());
        for (uint8_t cam_idx = 0; cam_idx < vmaps.size(); ++cam_idx)
        {
            auto vmap = vmaps[cam_idx];
            auto vign = cv::Mat(vmap.rows(), vmap.cols(), CvMatT(1), vmap.data());

            // TODO maybe dont copy
            vign.copyTo(this->vigns_base[cam_idx]);
        }

        this->build_mask = true;
        this->build_mirrors = true;
        this->build_vigns = false;
    }

    void PanoWeave::buildMask()
    {
        if (!this->build_mask)
            return;

        // build rescale mask from new maps and vignettes
        this->mask_base = cv::Mat::zeros(this->res, CvMatT(1));
        for (uint8_t cam_idx = 0; cam_idx < this->calib.intrinsics.size(); ++cam_idx)
        {
            cv::Mat mask_part;
            cv::remap(this->vigns_base[cam_idx] > 0, mask_part, static_cast<cv::Mat>(this->maps[cam_idx]), cv::noArray(), cv::INTER_NEAREST);
            this->mask_base += mask_part;
        }
        this->mask_base.forEach<ScalarT>([&](auto &val, auto pos)
                                         {
                                            (void) pos;
                                            if (val > 0) val = 1.0 / val; });

        this->build_mirrors = true;
        this->build_mask = false;
    }

    void mirrorChannels(const cv::Mat &in, int channels, cv::Mat &out)
    {
        auto comp = new cv::Mat[channels];
        for (uint8_t c = 0; c < channels; ++c)
            comp[c] = in;
        cv::merge(comp, channels, out);
        delete[] comp;
    }

    void PanoWeave::buildMirrors()
    {
        if (!this->build_mirrors)
            return;

        this->vigns.resize(this->vigns_base.size());
        for (uint8_t cam_idx = 0; cam_idx < this->calib.intrinsics.size(); ++cam_idx)
        {
            mirrorChannels(this->vigns_base[cam_idx], this->channels, this->vigns[cam_idx]);
        }
        mirrorChannels(this->mask_base, this->channels, this->mask);

        this->build_mirrors = false;
    }

}
