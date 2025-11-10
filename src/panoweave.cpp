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

#ifdef USE_CUDA
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#endif

namespace panoweave
{
    using CvPoint3T = cv::Point3_<ScalarT>;

#ifdef USE_CUDA
    struct DeviceData
    {
            std::vector<cv::cuda::GpuMat> mapx, mapy;
            std::vector<cv::cuda::GpuMat> vigns;
            std::vector<cv::cuda::GpuMat> response;
            cv::cuda::GpuMat mask;
            cv::cuda::Stream stream;
    };

    void cuda_correctResponse(cv::InputArray src, cv::InputArray inv_resp, cv::OutputArray dst, cv::cuda::Stream &stream = cv::cuda::Stream::Null());
#else
    struct DeviceData {};
#endif

    Stitcher::Stitcher()
    {
        spdlog::trace("Stitcher::Stitcher()");
        spdlog::cfg::load_env_levels();
#ifdef USE_CUDA
        this->dev = std::make_unique<DeviceData>();
#endif
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

    Stitcher::~Stitcher() = default;

    void Stitcher::loadCalibration(const std::string &filepath)
    {
        spdlog::trace("Stitcher::loadCalibration(const std::string &)");
        std::ifstream file(filepath);
        cereal::JSONInputArchive archive(file);
        archive(this->calib);

#ifdef USE_CUDA
        this->dev->response.resize(this->calib.response.size());
#endif
        this->response.resize(this->calib.response.size());
        for (uint8_t i = 0; i < this->calib.response.size(); ++i)
        {
            cv::Mat({static_cast<int>(this->calib.response[i].size())}, CvMatT(1), this->calib.response[i].data()).copyTo(this->response[i]);
#ifdef USE_CUDA
            this->dev->response[i].upload(this->response[i]);
#endif
        }

        this->build_maps = true;
        this->build_vigns = true;
    }

    const basalt::Calibration<ScalarT> &Stitcher::calibration()
    {
        spdlog::trace("Stitcher::calibration()");
        return this->calib;
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

#ifndef USE_CUDA
            static cv::UMat undist, remapped, stitched;

            stitched.create(this->res, CvMatT(this->channels));
            stitched.setTo(cv::Scalar::all(0));
            for (uint8_t i = 0; i < images.size(); ++i)
            {
                cv::Mat img = images[i];

                if (this->response.empty())
                    img.convertTo(undist, CvMatT(this->channels));
                else
                    correctResponse(img, this->response[i], undist);
                cv::multiply(undist, this->vigns[i], undist);
                cv::multiply(undist, target_exposure / exposure[i], undist);

                cv::remap(undist, remapped, this->maps[i], cv::noArray(), cv::INTER_NEAREST);

                cv::add(remapped, stitched, stitched);
            }
            cv::divide(stitched, this->mask, stitched);
            stitched.convertTo(pano, CV_8UC(this->channels));
#else
            static cv::cuda::GpuMat img, undist, remapped, stitched;

            stitched.create(this->res, CvMatT(this->channels));
            stitched.setTo(cv::Scalar::all(0), this->dev->stream);
            for (uint8_t i = 0; i < images.size(); ++i)
            {
                img.upload(images[i], this->dev->stream);
                if (this->response.empty())
                    img.convertTo(undist, CvMatT(this->channels), this->dev->stream);
                else
                    cuda_correctResponse(img, this->dev->response[i], undist, this->dev->stream);
                cv::cuda::multiply(undist, this->dev->vigns[i], undist, 1.0, -1, this->dev->stream);
                cv::cuda::multiply(undist, cv::Scalar::all(target_exposure / exposure[i]), undist, 1.0, -1, this->dev->stream);
                cv::cuda::remap(undist, remapped, this->dev->mapx[i], this->dev->mapy[i], cv::INTER_NEAREST, 0, cv::Scalar(), this->dev->stream);

                cv::cuda::add(remapped, stitched, stitched, cv::noArray(), -1, this->dev->stream);
            }
            cv::cuda::divide(stitched, this->dev->mask, stitched, 1.0, -1, this->dev->stream);
            stitched.convertTo(stitched, CV_8UC(this->channels), this->dev->stream);
            stitched.download(pano, this->dev->stream);
            this->dev->stream.waitForCompletion();
#endif
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

    void createSphericalPoints(const cv::Size &res, const CvFovT &fov, cv::Mat &points)
    {
        spdlog::trace("createSphericalPoints(const cv::Size &, ScalarT, ScalarT, cv::Mat &)");
        points.create(res, CvMatT(3));
        points.forEach<CvPoint3T>([&](auto &point, const auto &pos) -> void
                                {
                                    const ScalarT u = pos[1];
                                    const ScalarT v = pos[0];

                                    const ScalarT x = u / res.width * fov.width - M_PI_2;
                                    const ScalarT y = v / res.height * fov.height;

                                    // Calculate X, Y, and Z components of spherical rays
                                    point.x = std::sin(y) * std::sin(x);
                                    point.y = std::sin(y) * std::cos(x);
                                    point.z = std::cos(y);
                                });
    }

    void transformPoints(const cv::Mat &in, const CvAffine3T &transform, cv::Mat &out)
    {
        spdlog::trace("transformPoints(const cv::Mat &, const CvAffine3T &, cv::Mat &)");
        out.create(in.size(), in.type());
        out.forEach<CvPoint3T>([&](auto &point, const auto &pos) -> void
                                { point = transform * in.at<CvPoint3T>(pos); });
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
        out.create(in.size(), in.type());
        out.forEach<CvPoint3T>([&](auto &point, const auto &pos) -> void
                               { point = in.at<CvPoint3T>(pos) * depth.at<ScalarT>(pos); });
    }

    void buildMappingTables(const cv::Mat &points, const basalt::Calibration<ScalarT> calibration, std::vector<cv::Mat> &maps)
    {
        spdlog::trace("buildMappingTables(const cv::Mat &, const basalt::Calibration<ScalarT>, std::vector<cv::Mat> &)");
        maps.clear();
        const uint8_t num_cams = calibration.intrinsics.size();
        maps.reserve(num_cams);

        for (uint8_t cam_idx = 0; cam_idx < num_cams; ++cam_idx)
        {
            maps.emplace_back(points.size(), CvMatT(2));
            std::visit([&](const auto& intr)
            {
                maps[cam_idx].forEach<cv::Vec<ScalarT, 2>>([&](auto &mapping, const auto pos) -> void
                {
                    auto p2d = Eigen::Map<Eigen::Vector<ScalarT, 2>>(mapping.val);
                    auto p3d = Eigen::Map<const Eigen::Vector<ScalarT, 3>>(points.at<cv::Vec<ScalarT, 3>>(pos).val);
                    if (!intr.project(calibration.T_i_c[cam_idx].inverse() * p3d, p2d))
                    {
                        p2d.setZero();
                    }
                });
            },
            calibration.intrinsics[cam_idx].variant);
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
            spdlog::warn("buildMaps(): resolution is empty");
            return false;
        }

        cv::Mat points;
        createSphericalPoints(this->res, this->fov_, points);
        transformPoints(points, this->tf, points);

        if (this->depth_static > 0.0)
        {
            applyDepth(points, this->depth_static, points);
        }
        else if (!this->depth_dynamic.empty() && this->depth_dynamic.type() == CvMatT(1))
        {
            applyDepth(points, this->depth_dynamic, points);
        }
        else
        {
            spdlog::warn("buildMaps(): depth is empty");
            return false;
        }

        std::vector<cv::Mat> maps;
        buildMappingTables(points, this->calib, maps);

#ifdef USE_CUDA
        this->dev->mapx.resize(maps.size());
        this->dev->mapy.resize(maps.size());
#endif
        this->maps.resize(maps.size());
        for (uint8_t i = 0; i < maps.size(); ++i)
        {
            static_cast<cv::Mat>(maps[i]).copyTo(this->maps[i]);
#ifdef USE_CUDA
            std::vector<cv::UMat> maps_xy;
            cv::split(this->maps[i], maps_xy);
            this->dev->mapx[i].upload(maps_xy[0]);
            this->dev->mapy[i].upload(maps_xy[1]);
#endif
        }

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
#ifndef USE_CUDA
            cv::compare(this->vigns_base[cam_idx], 0, mask_bin, cv::CMP_GT);
            cv::divide(mask_bin, 255, mask_bin);
            cv::remap(mask_bin, mask_part, this->maps[cam_idx], cv::noArray(), cv::INTER_NEAREST);
#else
            cv::cuda::GpuMat mask_part_cudev, mask_bin_cudev;
            cv::cuda::compare(this->vigns_base[cam_idx], cv::Scalar::all(0), mask_bin_cudev, cv::CMP_GT);
            cv::cuda::divide(mask_bin_cudev, cv::Scalar::all(255), mask_bin_cudev);
            cv::cuda::remap(mask_bin_cudev, mask_part_cudev, this->dev->mapx[cam_idx], this->dev->mapy[cam_idx], cv::INTER_NEAREST);
            mask_part_cudev.download(mask_part);
            mask_bin_cudev.download(mask_bin);
#endif
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

#ifdef USE_CUDA
        this->dev->vigns.resize(this->vigns_base.size());
#endif
        this->vigns.resize(this->vigns_base.size());
        for (uint8_t cam_idx = 0; cam_idx < this->calib.intrinsics.size(); ++cam_idx)
        {
            mirrorChannels(this->vigns_base[cam_idx], this->channels, this->vigns[cam_idx]);
#ifdef USE_CUDA
            this->dev->vigns[cam_idx].upload(this->vigns[cam_idx]);
#endif
        }
        mirrorChannels(this->mask_base, this->channels, this->mask);
#ifdef USE_CUDA
        this->dev->mask.upload(this->mask);
#endif

        this->build_mirrors = false;
    }

}
