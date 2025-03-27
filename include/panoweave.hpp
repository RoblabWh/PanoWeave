#pragma once
#include <opencv2/core.hpp>
#include <opencv2/gapi.hpp>
#include "basalt/calibration/calibration.hpp"

namespace PanoWeave
{
    using ScalarT = float;
    #define CvMatT CV_32FC
    using CvPoint2T = cv::Point_<ScalarT>;
    using CvPoint3T = cv::Point3_<ScalarT>;
    using EigenPoint2T = Eigen::Matrix<ScalarT, 2, 1>;
    using EigenPoint3T = Eigen::Matrix<ScalarT, 3, 1>;
    using EigenAlVec2T = Eigen::aligned_vector<EigenPoint2T>;
    using EigenAlVec3T = Eigen::aligned_vector<EigenPoint3T>;

    template<size_t T>
    class EigenAlignedCvMat
    {
    public:
        using EigenT = Eigen::aligned_vector<Eigen::Matrix<ScalarT, T, 1>>;

        EigenAlignedCvMat(const cv::Size &res)
            : eigen_mat(res.area()), cv_mat(res, CvMatT(T), this->eigen_mat.data()) {}

        operator cv::Mat &()
        {
            return this->cv_mat;
        }
        operator Eigen::aligned_vector<Eigen::Matrix<ScalarT, T, 1>> &()
        {
            return this->eigen_mat;
        }

    private:
        EigenT eigen_mat;
        cv::Mat cv_mat;
    };

    class PanoWeave
    {
    public:
        PanoWeave() = default;
        PanoWeave(const std::string &calibration_filepath);
        PanoWeave(const std::string &calibration_filepath, ScalarT fov_x, ScalarT fov_y, ScalarT depth);
        PanoWeave(const std::string &calibration_filepath, ScalarT fov_x, ScalarT fov_y, const cv::Mat &depth);
        ~PanoWeave() = default;

        void setDepth(ScalarT depth);
        void setDepth(const cv::Mat &depth);
        void setFov(ScalarT fov_x, ScalarT fov_y);
        void setResolution(const cv::Size &resolution);
        void setResolution(int width, int height);
        void setVignetteThreshold(ScalarT threshold);

        ScalarT fovX() const;
        ScalarT fovX(ScalarT fov);
        ScalarT fovY() const;
        ScalarT fovY(ScalarT fov);
        int width() const;
        int width(int width);
        int height() const;
        int height(int height);

        ScalarT vignetteThreshold() const;
        ScalarT vignetteThreshold(ScalarT thr);

        void weave(const std::vector<cv::Mat> &images, cv::Mat &pano);
        void weave(const std::vector<cv::Mat> &images, ScalarT depth, cv::Mat &pano);
        void weave(const std::vector<cv::Mat> &images, const cv::Mat &depth, cv::Mat &pano);

    private:
        void loadCalibration(const std::string &calibration_filepath);
        bool buildInternals();
        bool buildMaps();
        void buildVignettes();
        void buildMask();
        void buildMirrors();

        bool build_maps = true, build_vigns = true, build_mask = true, build_mirrors = true;
        int channels = 0;
        cv::Size res;
        std::vector<EigenAlignedCvMat<2>> maps;
        ScalarT vign_thresh = 0.5;
        cv::Mat depth_dynamic;
        ScalarT depth_static = 0.0;
        ScalarT fov_x = M_PI * 2.0, fov_y = M_PI;
        basalt::Calibration<ScalarT> calib;
        std::vector<cv::Mat> vigns, vigns_base;
        cv::Mat mask, mask_base;
    };

}
