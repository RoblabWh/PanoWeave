#include "panoweave.hpp"
#include "ocl_kernels.hpp"

#include <fstream>
#include <cereal/archives/json.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/cfg/env.h>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc.hpp>

#include "basalt/serialization/headers_serialization.h"
#include "basalt/camera/generic_camera.hpp"

namespace PanoWeave
{

    Stitcher::Stitcher()
    {
        spdlog::trace("Stitcher::Stitcher()");
        spdlog::cfg::load_env_levels();
    }

    Stitcher::Stitcher(const std::string &filepath) : Stitcher()
    {
        spdlog::trace("Stitcher::Stitcher(const std::string &)");
        this->loadCalibration(filepath);
    }

    Stitcher::Stitcher(const std::string &filepath, ScalarT depth) : Stitcher()
    {
        spdlog::trace("Stitcher::Stitcher(const std::string &, ScalarT, ScalarT, ScalarT)");
        this->loadCalibration(filepath);
        this->depth_static = depth;
        this->buildInternals();
    }

    Stitcher::Stitcher(const std::string &filepath, const cv::Mat &depth) : Stitcher()
    {
        spdlog::trace("Stitcher::Stitcher(const std::string &, ScalarT, ScalarT, const cv::Mat &)");
        this->loadCalibration(filepath);
        depth.copyTo(this->depth_dynamic);
        this->buildInternals();
    }

    void Stitcher::loadCalibration(const std::string &filepath)
    {
        spdlog::trace("Stitcher::loadCalibration(const std::string &)");
        std::ifstream file(filepath);
        cereal::JSONInputArchive archive(file);
        basalt::Calibration<double> calib;
        archive(calib);
        this->calib = calib.cast<ScalarT>();

        this->response.resize(this->calib.response.size());
        for (uint8_t i = 0; i < this->calib.response.size(); ++i)
            cv::Mat({static_cast<int>(this->calib.response[i].size())}, CvMatT(1), this->calib.response[i].data()).copyTo(this->response[i]);

        this->build_maps = true;
        this->build_vigns = true;
    }

    void Stitcher::setDepth(ScalarT depth)
    {
        spdlog::trace("Stitcher::setDepth(ScalarT)");
        this->depth_static = depth;
        this->depth_dynamic.release();
        this->build_maps = true;
    }

    void Stitcher::setDepth(const cv::Mat &depth)
    {
        spdlog::trace("Stitcher::setDepth(const cv::Mat &)");
        this->depth_static = 0.0;
        depth.copyTo(this->depth_dynamic);
        this->build_maps = true;
    }

    CvFovT Stitcher::fov() const
    {
        spdlog::trace("Stitcher::fov() const");
        return this->fov_;
    }
    CvFovT Stitcher::fov(ScalarT fov_x, ScalarT fov_y)
    {
        spdlog::trace("Stitcher::fov(ScalarT, ScalarT)");
        return this->fov({fov_x, fov_y});
    }
    CvFovT Stitcher::fov(CvFovT fov)
    {
        spdlog::trace("Stitcher::fov(CvFovT)");
        this->fov_ = fov;
        this->build_maps = true;
        return fov;
    }
    ScalarT Stitcher::fovX() const
    {
        spdlog::trace("Stitcher::fovX() const");
        return this->fov_.width;
    }
    ScalarT Stitcher::fovX(ScalarT fov)
    {
        spdlog::trace("Stitcher::fovX(ScalarT)");
        this->fov(fov, this->fov_.height);
        return fov;
    }
    ScalarT Stitcher::fovY() const
    {
        spdlog::trace("Stitcher::fovY() const");
        return this->fov_.height;
    }
    ScalarT Stitcher::fovY(ScalarT fov)
    {
        spdlog::trace("Stitcher::fovY(ScalarT)");
        this->fov(this->fov_.width, fov);
        return fov;
    }

    cv::Size Stitcher::resolution() const
    {
        spdlog::trace("Stitcher::resolution() const");
        return this->res;
    }
    cv::Size Stitcher::resolution(const cv::Size &resolution)
    {
        spdlog::trace("Stitcher::resolution(const cv::Size &)");
        this->res = resolution;
        this->build_maps = true;
        return resolution;
    }
    cv::Size Stitcher::resolution(int width, int height)
    {
        spdlog::trace("Stitcher::resolution(int, int)");
        return this->resolution({width, height});
    }
    int Stitcher::width() const
    {
        spdlog::trace("Stitcher::width() const");
        return this->res.width;
    }
    int Stitcher::width(int width)
    {
        spdlog::trace("Stitcher::width(int)");
        this->resolution(width, this->res.height);
        return width;
    }
    int Stitcher::height() const
    {
        spdlog::trace("Stitcher::height() const");
        return this->res.height;
    }
    int Stitcher::height(int height)
    {
        spdlog::trace("Stitcher::height(int)");
        this->resolution(this->res.width, height);
        return height;
    }

    ScalarT Stitcher::vignetteThreshold() const
    {
        spdlog::trace("Stitcher::vignetteThreshold() const");
        return this->vign_thresh;
    }
    ScalarT Stitcher::vignetteThreshold(ScalarT threshold)
    {
        spdlog::trace("Stitcher::vignetteThreshold(ScalarT)");
        this->vign_thresh = threshold;
        this->build_vigns = true;
        return threshold;
    }

    bool Stitcher::useMaskAsVignette() const
    {
        spdlog::trace("Stitcher::useMaskAsVignette() const");
        return this->use_mask_as_vign;
    }
    bool Stitcher::useMaskAsVignette(bool use)
    {
        spdlog::trace("Stitcher::useMaskAsVignette(bool)");
        this->use_mask_as_vign = use;
        this->build_mask = this->build_mask || use;
        this->build_vigns = this->build_vigns || !use;
        return use;
    }

    CvAffine3T Stitcher::transform() const
    {
        spdlog::trace("Stitcher::transform() const");
        return this->tf;
    }
    CvAffine3T Stitcher::transform(const CvAffine3T &transform)
    {
        spdlog::trace("Stitcher::transform(const CvAffine3T &)");
        this->tf = transform;
        this->build_maps = true;
        return transform;
    }
    CvAffine3T Stitcher::transform(const CvAffine3T::Mat3 &rotation, const CvAffine3T::Vec3 &translation)
    {
        spdlog::trace("Stitcher::transform(const CvAffine3T::Mat3 &, const CvAffine3T::Vec3 &)");
        return this->transform(CvAffine3T(rotation, translation));
    }
    CvAffine3T Stitcher::transform(const CvAffine3T::Vec3 &rotation, const CvAffine3T::Vec3 &translation)
    {
        spdlog::trace("Stitcher::transform(const CvAffine3T::Vec3 &, const CvAffine3T::Vec3 &)");
        return this->transform(CvAffine3T(rotation, translation));
    }
    CvAffine3T Stitcher::transform(const CvAffine3T::Mat4 &affine)
    {
        spdlog::trace("Stitcher::transform(const CvAffine3T::Mat4 &)");
        return this->transform(CvAffine3T(affine));
    }

    bool ocl_correctResponse(cv::InputArray _src, cv::InputArray _inv_resp, cv::OutputArray _dst)
    {
        spdlog::trace("ocl_correctResponse(cv::InputArray, cv::InputArray, cv::OutputArray)");
        const int rows_per_wi = cv::ocl::Device::getDefault().isIntel() ? 4 : 1;

        const int tp = _src.type();
        if (tp != CV_8UC1 && tp != CV_8UC3 && tp != CV_8UC4)
            return false;

        cv::UMat src = _src.getUMat();
        cv::UMat inv_resp = _inv_resp.getUMat();
        cv::UMat dst = _dst.getUMat();

        cv::ocl::Kernel k("correct_response", cv::ocl::PanoWeave::correct_response_oclsrc, cv::format("-DCN=%d", _src.channels()));
        k.args(
            cv::ocl::KernelArg::ReadOnlyNoSize(src),
            cv::ocl::KernelArg::ReadOnlyNoSize(inv_resp),
            cv::ocl::KernelArg::WriteOnly(dst));

        size_t global_threads[2] = {(size_t)dst.cols, ((size_t)dst.rows + rows_per_wi - 1) / rows_per_wi};
        return k.run(2, global_threads, nullptr, false);
    }

    void correctResponse(cv::InputArray _src, cv::InputArray _inv_resp, cv::OutputArray _dst)
    {
        spdlog::trace("correctResponse(cv::InputArray, cv::InputArray, cv::OutputArray)");
        const int cn = _src.channels();
        _dst.create(_src.size(), CvMatT(cn));

        if (cv::ocl::haveOpenCL() && ocl_correctResponse(_src, _inv_resp, _dst))
            return;

        cv::Mat src = _src.getMat();
        cv::Mat dst = _dst.getMat();
        cv::Mat inv_resp = _inv_resp.getMat();

        if (cn == 1)
            dst.forEach<ScalarT>([&](auto &val, const int *pos)
                                 { val = inv_resp.at<ScalarT>(src.at<uchar>(pos)); });
        else if (cn == 3)
            dst.forEach<cv::Vec<ScalarT, 3>>([&](auto &dval, const int *pos)
                                             { const auto &sval = src.at<cv::Vec3b>(pos);
                                                dval[0] = inv_resp.at<ScalarT>(sval[0]);
                                                dval[1] = inv_resp.at<ScalarT>(sval[1]);
                                                dval[2] = inv_resp.at<ScalarT>(sval[2]); });
        else if (cn == 4)
            dst.forEach<cv::Vec<ScalarT, 4>>([&](auto &dval, const int *pos)
                                             { const auto &sval = src.at<cv::Vec4b>(pos);
                                                dval[0] = inv_resp.at<ScalarT>(sval[0]);
                                                dval[1] = inv_resp.at<ScalarT>(sval[1]);
                                                dval[2] = inv_resp.at<ScalarT>(sval[2]);
                                                dval[3] = sval[3]; });
        else
        {
            spdlog::error("Can not correct response for {} channel images", cn);
            CV_Assert(false);
        }
    }

    void Stitcher::stitch(const std::vector<cv::Mat> &images, cv::Mat &pano)
    {
        spdlog::trace("Stitcher::stitch(const std::vector<cv::Mat> &, cv::Mat &)");
        this->stitch(images, std::vector<ScalarT>(images.size(), 1.0), pano);
    }
    void Stitcher::stitch(const std::vector<cv::Mat> &images, const std::vector<ScalarT> &exposure, cv::Mat &pano)
    {
        spdlog::trace("Stitcher::stitch(const std::vector<cv::Mat> &, const std::vector<ScalarT> &, cv::Mat &)");

        if (images.front().channels() != this->channels)
        {
            this->channels = images.front().channels();
            this->build_mirrors = true;
        }

        if (this->buildInternals())
        {
            std::vector<ScalarT> exp_sort = exposure;
            std::sort(exp_sort.begin(), exp_sort.end());
            ScalarT target_exposure = exp_sort[exp_sort.size() / 2];

            cv::UMat pano_l = cv::UMat::zeros(this->res, CvMatT(this->channels));

            for (uint8_t i = 0; i < images.size(); ++i)
            {
                cv::Mat img = images[i];

                cv::UMat undist;
                if (this->response.empty())
                    img.convertTo(undist, CvMatT(this->channels));
                else
                    correctResponse(img, this->response[i], undist);
                cv::multiply(undist, this->vigns[i], undist);
                cv::multiply(undist, target_exposure / exposure[i], undist);

                cv::UMat remapped;
                cv::remap(undist, remapped, this->maps_dev[i], cv::noArray(), cv::INTER_NEAREST);

                cv::add(remapped, pano_l, pano_l);
            }
            cv::divide(pano_l, this->mask, pano_l);
            pano_l.convertTo(pano, CV_8UC(this->channels));
        }
        else
        {
            spdlog::warn("stitch(): Unable to stitch image, failed building internals");
        }
    }

    void Stitcher::stitch(const std::vector<cv::Mat> &images, const std::vector<ScalarT> &exposure, ScalarT depth, cv::Mat &pano)
    {
        spdlog::trace("Stitcher::stitch(const std::vector<cv::Mat> &, const std::vector<ScalarT> &, ScalarT, cv::Mat &)");
        this->setDepth(depth);
        this->stitch(images, exposure, pano);
    }

    void Stitcher::stitch(const std::vector<cv::Mat> &images, const std::vector<ScalarT> &exposure, const cv::Mat &depth, cv::Mat &pano)
    {
        spdlog::trace("Stitcher::stitch(const std::vector<cv::Mat> &, const std::vector<ScalarT> &, const cv::Mat &, cv::Mat &)");
        this->setDepth(depth);
        this->stitch(images, exposure, pano);
    }

    bool Stitcher::buildInternals()
    {
        spdlog::trace("Stitcher::buildInternals()");

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

    void createSphericalPoints3(const cv::Size &res, const CvFovT &fov, cv::Mat &rays)
    {
        spdlog::trace("createSphericalPoints3(const cv::Size &, ScalarT, ScalarT, cv::Mat &)");
        CV_Assert(rays.type() == CvMatT(3) && rays.size() == res);

        const ScalarT width_2 = res.width / 2.0;
        const ScalarT height_2 = res.height / 2.0;

        rays.forEach<CvPoint3T>([&](CvPoint3T &point, const int *pos) -> void
                                {
                                    const int u = pos[1];
                                    const int v = pos[0];

                                    const ScalarT x = (u - width_2) / res.width * fov.width + M_PI_2;
                                    const ScalarT y = (v - height_2) / res.height * fov.height;

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

    bool Stitcher::buildMaps()
    {
        spdlog::trace("Stitcher::buildMaps()");
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
        createSphericalPoints3(this->res, this->fov_, rays);
        transformSphericalPoints3(rays, this->tf, rays);

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

        this->maps_dev.resize(this->maps.size());
        for (uint8_t i = 0; i < this->maps.size(); ++i)
            static_cast<cv::Mat>(this->maps[i]).copyTo(this->maps_dev[i]);

        this->build_mask = true;
        this->build_maps = false;
        return true;
    }

    void Stitcher::buildVignettes()
    {
        spdlog::trace("Stitcher::buildVignettes()");
        if (!this->build_vigns)
            return;

        auto vmaps = this->calib.vignette_maps(this->vign_thresh);

        this->vigns_base.resize(vmaps.size());
        for (uint8_t cam_idx = 0; cam_idx < vmaps.size(); ++cam_idx)
        {
            auto vmap = vmaps[cam_idx];
            auto vign = cv::Mat(vmap.rows(), vmap.cols(), CvMatT(1), vmap.data());
            vign.copyTo(this->vigns_base[cam_idx]);
        }

        this->build_mask = true;
        this->build_mirrors = true;
        this->build_vigns = false;
    }

    void Stitcher::buildMask()
    {
        spdlog::trace("Stitcher::buildMask()");
        if (!this->build_mask)
            return;

        // build rescale mask from new maps and vignettes
        this->mask_base = cv::UMat::zeros(this->res, CvMatT(1));
        for (uint8_t cam_idx = 0; cam_idx < this->calib.intrinsics.size(); ++cam_idx)
        {
            cv::UMat mask_part, mask_bin;
            cv::compare(this->vigns_base[cam_idx], 0, mask_bin, cv::CMP_GT);
            cv::divide(mask_bin, 255, mask_bin);
            cv::remap(mask_bin, mask_part, this->maps_dev[cam_idx], cv::noArray(), cv::INTER_NEAREST);
            cv::add(this->mask_base, mask_part, this->mask_base, cv::noArray(), CvMatT(1));
            if (this->use_mask_as_vign)
                mask_bin.convertTo(this->vigns_base[cam_idx], CvMatT(1));
        }

        this->build_mirrors = true;
        this->build_mask = false;
    }

    void mirrorChannels(const cv::UMat &in, int channels, cv::UMat &out)
    {
        spdlog::trace("mirrorChannels(const cv::UMat &, int, cv::UMat &)");
        auto comp = std::vector<cv::UMat>(channels);
        for (uint8_t c = 0; c < channels; ++c)
            comp[c] = in;
        cv::merge(comp, out);
    }

    void Stitcher::buildMirrors()
    {
        spdlog::trace("Stitcher::buildMirrors()");
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
