#pragma once
#include <opencv2/core.hpp>
#include <opencv2/core/affine.hpp>
#include "basalt/calibration/calibration.hpp"

namespace PanoWeave
{
    using ScalarT = float;
    #define CvMatT CV_32FC
    using CvFovT = cv::Size_<ScalarT>;
    using CvPoint2T = cv::Point_<ScalarT>;
    using CvPoint3T = cv::Point3_<ScalarT>;
    using CvAffine3T = cv::Affine3<ScalarT>;
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

        operator Eigen::aligned_vector<Eigen::Matrix<ScalarT, T, 1>> &()
        {
            return this->eigen_mat;
        }
        operator cv::Mat &()
        {
            return this->cv_mat;
        }

    private:
        EigenT eigen_mat;
        cv::Mat cv_mat;
    };

    class PanoWeave
    {
    public:
        PanoWeave();
        PanoWeave(const std::string &calibration_filepath);
        PanoWeave(const std::string &calibration_filepath, ScalarT depth);
        PanoWeave(const std::string &calibration_filepath, const cv::Mat &depth);
        ~PanoWeave() = default;

        void loadCalibration(const std::string &calibration_filepath);
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

        CvAffine3T transform() const;
        CvAffine3T transform(const CvAffine3T &transform);
        CvAffine3T transform(const CvAffine3T::Mat4 &affine);
        CvAffine3T transform(const CvAffine3T::Mat3 &rotation,
                             const CvAffine3T::Vec3 &translation = CvAffine3T::Vec3(0, 0, 0));
        CvAffine3T transform(const CvAffine3T::Vec3 &rotation,
                             const CvAffine3T::Vec3 &translation = CvAffine3T::Vec3(0, 0, 0));

        void weave(const std::vector<cv::Mat> &images, cv::Mat &pano);
        void weave(const std::vector<cv::Mat> &images, ScalarT depth, cv::Mat &pano);
        void weave(const std::vector<cv::Mat> &images, const cv::Mat &depth, cv::Mat &pano);

    private:
        bool buildInternals();
        bool buildMaps();
        void buildVignettes();
        void buildMask();
        void buildMirrors();

        bool build_maps = true, build_vigns = true, build_mask = true, build_mirrors = true;
        int channels = 0;
        cv::Size res;
        std::vector<EigenAlignedCvMat<2>> maps;
        std::vector<cv::UMat> maps_dev;
        ScalarT vign_thresh = 0.5;
        cv::UMat depth_dynamic;
        ScalarT depth_static = 0.0;
        CvFovT fov_ = {M_PI * 2.0, M_PI};
        CvAffine3T tf;
        basalt::Calibration<ScalarT> calib;
        std::vector<cv::UMat> vigns, vigns_base;
        std::vector<cv::UMat> response;
        cv::UMat mask, mask_base;
    };

}
