#pragma once
#include <opencv2/core.hpp>
#include <opencv2/core/affine.hpp>
#include "basalt/calibration/calibration.hpp"

namespace panoweave
{
    #define CvMatT CV_32FC
    using ScalarT = float;
    using CvFovT = cv::Size_<ScalarT>;
    using CvAffine3T = cv::Affine3<ScalarT>;

    class Stitcher
    {
    public:
        Stitcher();
        Stitcher(const std::string &calibration_filepath);
        Stitcher(const std::string &calibration_filepath, ScalarT depth);
        Stitcher(const std::string &calibration_filepath, const cv::Mat &depth);
        ~Stitcher() = default;

        void loadCalibration(const std::string &calibration_filepath);
        const basalt::Calibration<ScalarT> &calibration();
        void setDepth(ScalarT depth);
        void setDepth(const cv::Mat &depth);

        cv::Size resolution() const;
        cv::Size resolution(const cv::Size &resolution);
        cv::Size resolution(int width, int height);
        int width() const;
        int width(int width);
        int height() const;
        int height(int height);

        CvFovT fov() const;
        CvFovT fov(ScalarT fov_x, ScalarT fov_y);
        CvFovT fov(CvFovT fov);
        ScalarT fovX() const;
        ScalarT fovX(ScalarT fov);
        ScalarT fovY() const;
        ScalarT fovY(ScalarT fov);

        ScalarT vignetteThreshold() const;
        ScalarT vignetteThreshold(ScalarT threshold);

        bool useMaskAsVignette() const;
        bool useMaskAsVignette(bool use);

        CvAffine3T transform() const;
        CvAffine3T transform(const CvAffine3T &transform);
        CvAffine3T transform(const CvAffine3T::Mat4 &affine);
        CvAffine3T transform(const CvAffine3T::Mat3 &rotation,
                             const CvAffine3T::Vec3 &translation = CvAffine3T::Vec3(0, 0, 0));
        CvAffine3T transform(const CvAffine3T::Vec3 &rotation,
                             const CvAffine3T::Vec3 &translation = CvAffine3T::Vec3(0, 0, 0));

        void stitch(const std::vector<cv::Mat> &images, cv::Mat &pano);
        void stitch(const std::vector<cv::Mat> &images, const std::vector<ScalarT> &exposure, cv::Mat &pano);
        void stitch(const std::vector<cv::Mat> &images, const std::vector<ScalarT> &exposure, ScalarT depth, cv::Mat &pano);
        void stitch(const std::vector<cv::Mat> &images, const std::vector<ScalarT> &exposure, const cv::Mat &depth, cv::Mat &pano);

    private:
        bool buildInternals();
        bool buildMaps();
        void buildVignettes();
        void buildMask();
        void buildMirrors();

        bool build_maps = true, build_vigns = true, build_mask = true, build_mirrors = true;
        bool use_mask_as_vign = false;
        int channels = 0;
        cv::Size res;
        std::vector<cv::UMat> maps;
        ScalarT vign_thresh = 0.5;
        cv::Mat depth_dynamic;
        ScalarT depth_static = 0.0;
        CvFovT fov_ = {M_PI * 2.0, M_PI};
        CvAffine3T tf;
        basalt::Calibration<ScalarT> calib;
        std::vector<cv::UMat> vigns, vigns_base;
        std::vector<cv::UMat> response;
        cv::UMat mask, mask_base;
    };

}
