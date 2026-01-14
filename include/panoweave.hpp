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

    /**
     * @brief Device-specific data structure.
     *
     * Holds device-specific resources for GPU acceleration (e.g., CUDA), if needed.
     */
    struct DeviceData;

    /**
     * @class Stitcher
     * @brief Stitches multiple camera images into a single spherical panorama using Basalt calibration data.
     *
     * The Stitcher supports setting resolution, field of view, depth, and transformation of the spherical camera.
     * Depth values can be provided as a uniform scalar or a per-pixel depth map to cover simple or complex use cases.
     * Exposure times of the input images can optionally be used to adjust the input images to a common exposure.
     * Images with one (mono), three (BGR, RGB), or four (BGRA, RGBA) channels are supported without extra configuration.
     *
     * @note Internal data structures are built lazily, i.e., only when required at the beginning of stitching. This avoids unnecessary computation when multiple parameters are changed before stitching.
     *
     * Example usage:
     * @code
     * panoweave::Stitcher stitcher("calibration.json");
     * stitcher.resolution(2160, 1080);
     * stitcher.setDepth(5.0f);
     * cv::Mat pano;
     * stitcher.stitch(images, exposures, pano);
     * @endcode
     */
    class Stitcher
    {
    public:

        /**
         * @brief Constructs an empty Stitcher object.
         *
         * Initializes an empty Stitcher object. Calibration data must be loaded,
         * resolution set, and depth defined before stitching operations can be performed.
         */
        Stitcher();

        /**
         * @brief Constructs a Stitcher object with the specified calibration file.
         *
         * Loads camera calibration data from the given file path.
         *
         * @param calibration_filepath Path to the camera calibration data file.
         *
         * @throws std::runtime_error If the file cannot be opened or the data is invalid.
         *
         * @see loadCalibration
         */
        Stitcher(const std::string &calibration_filepath);

        /**
         * @brief Constructs a Stitcher object with the specified calibration file and uniform depth.
         *
         * Loads camera calibration and sets a uniform depth value.
         *
         * @param calibration_filepath Path to the camera calibration data file.
         * @param depth Uniform depth value.
         *
         * @throws std::runtime_error If the file cannot be opened or the data is invalid.
         *
         * @see loadCalibration, setDepth
         */
        Stitcher(const std::string &calibration_filepath, ScalarT depth);

        /**
         * @brief Constructs a Stitcher object with the specified calibration file and per-point depth.
         *
         * Loads camera calibration and sets a per-point depth map.
         *
         * @param calibration_filepath Path to the camera calibration data file.
         * @param depth Per-point depth map.
         *
         * @throws std::runtime_error If the file cannot be opened or the data is invalid.
         *
         * @see loadCalibration, setDepth
         */
        Stitcher(const std::string &calibration_filepath, const cv::Mat &depth);

        /**
         * @brief Destructor for the Stitcher class.
         *
         * Uses the default destructor.
         */
        ~Stitcher();


        /**
         * @brief Loads camera calibration data from the specified file.
         *
         * Reads calibration parameters from a Basalt calibration.json file.
         *
         * @note Requires rebuilding of all internal data structures.
         *
         * @param calibration_filepath Path to the camera calibration data file.
         *
         * @throws std::runtime_error If the file cannot be opened or the data is invalid.
         */
        void loadCalibration(const std::string &calibration_filepath);

        /**
         * @brief Returns a read-only reference to the calibration data.
         *
         * @return Calibration data used for stitching.
         */
        const basalt::Calibration<ScalarT> &calibration() const;

        /**
         * @brief Sets a uniform depth for the panorama.
         *
         * Uses the same depth value for all points in the panorama.
         *
         * @note Requires rebuilding the mapping tables.
         *
         * @param depth Uniform depth value.
         */
        void setDepth(ScalarT depth);

        /**
         * @brief Sets a per-point depth map for the panorama.
         *
         * The depth map provides depth information for each point in the panorama.
         * Its resolution must match the output resolution of the panorama.
         *
         * @note Requires rebuilding the mapping tables.
         *
         * @param depth Per-point depth map.
         */
        void setDepth(const cv::Mat &depth);


        /**
         * @brief Get the current output resolution for the panorama.
         *
         * @return The current resolution.
         */
        cv::Size resolution() const;

        /**
         * @brief Set the output resolution for the panorama.
         * @note Requires rebuilding the mapping tables.
         *
         * @param resolution Desired resolution.
         * @return The set resolution.
         */
        cv::Size resolution(const cv::Size &resolution);

        /**
         * @brief Set the output resolution for the panorama.
         *
         * Overload accepting width and height as separate parameters.
         *
         * @note Requires rebuilding the mapping tables.
         *
         * @param width Desired width.
         * @param height Desired height.
         * @return The set resolution.
         */
        cv::Size resolution(int width, int height);

        /**
         * @brief Get the current output width for the panorama.
         *
         * @return The current width.
         */
        int width() const;

        /**
         * @brief Set the output width for the panorama.
         * @note Requires rebuilding the mapping tables.
         *
         * @param width Desired width.
         * @return The set width.
         */
        int width(int width);

        /**
         * @brief Get the current output height for the panorama.
         *
         * @return The current height.
         */
        int height() const;

        /**
         * @brief Set the output height for the panorama.
         * @note Requires rebuilding the mapping tables.
         *
         * @param height Desired height.
         * @return The set height.
         */
        int height(int height);


        /**
         * @brief Get the current field of view (FOV) for the panorama.
         *
         * @return The current FOV.
         */
        CvFovT fov() const;

        /**
         * @brief Set the field of view (FOV) for the panorama.
         *
         * The FOV must be in the range (0, 2*Pi] for horizontal and (0, Pi] for vertical.
         *
         * @note Requires rebuilding the mapping tables.
         *
         * @param fov Desired FOV.
         * @return The set FOV.
         */
        CvFovT fov(CvFovT fov);

        /**
         * @brief Set the field of view (FOV) for the panorama.
         *
         * Overload accepting horizontal and vertical FOV as separate parameters.
         *
         * @note Requires rebuilding the mapping tables.
         *
         * @param fov_x Desired horizontal FOV.
         * @param fov_y Desired vertical FOV.
         * @return The set FOV.
         */
        CvFovT fov(ScalarT fov_x, ScalarT fov_y);

        /**
         * @brief Get the current horizontal field of view (FOV) for the panorama.
         *
         * @return The current horizontal FOV.
         */
        ScalarT fovX() const;

        /**
         * @brief Set the horizontal field of view (FOV) for the panorama.
         *
         * The horizontal FOV must be in the range (0, 2*Pi].
         *
         * @note Requires rebuilding the mapping tables.
         *
         * @param fov Desired horizontal FOV.
         * @return The set horizontal FOV.
         */
        ScalarT fovX(ScalarT fov);

        /**
         * @brief Get the current vertical field of view (FOV) for the panorama.
         *
         * @return The current vertical FOV.
         */
        ScalarT fovY() const;

        /**
         * @brief Set the vertical field of view (FOV) for the panorama.
         *
         * The vertical FOV must be in the range (0, Pi].
         *
         * @note Requires rebuilding the mapping tables.
         *
         * @param fov Desired vertical FOV.
         * @return The set vertical FOV.
         */
        ScalarT fovY(ScalarT fov);


        /**
         * @brief Get the current vignette threshold.
         *
         * @return The current vignette threshold.
         */
        ScalarT vignetteThreshold() const;

        /**
         * @brief Set the vignette threshold used to generate vignette maps.
         *
         * The vignette maps are used to compensate for lens vignetting in input images.
         * Values should be in the range [0.0, 1.0].
         * A lower threshold means pixels farther from the center are considered valid.
         *
         * @note Requires rebuilding the vignette maps.
         *
         * @param threshold Desired vignette threshold.
         * @return The set vignette threshold.
         */
        ScalarT vignetteThreshold(ScalarT threshold);


        /**
         * @brief Gets whether the mask is used as vignette.
         *
         * @return True if the mask is used as vignette, false otherwise.
         */
        bool useMaskAsVignette() const;

        /**
         * @brief Sets whether to use the mask as vignette.
         *
         * If set to true, the mask will replace the vignette maps.
         * This is useful when vignette correction is not desired (for example, if vignetting calibration quality is insufficient),
         * and the mask should define valid pixel regions instead.
         *
         * @note Requires rebuilding the mask or vignette maps depending on the value.
         *
         * @param use True to use the mask as vignette, false otherwise.
         * @return The set value.
         */
        bool useMaskAsVignette(bool use);


        /**
         * @brief Get the current transformation from body to calibration coordinate system.
         *
         * @return The current transformation.
         */
        CvAffine3T transform() const;

        /**
         * @brief Set the transformation from body to calibration coordinate system.
         *
         * This transformation is applied to spherical points before projecting them into camera views.
         * The Basalt calibration coordinate system is either the IMU or the first camera coordinate system,
         * depending on the calibration type.
         *
         * @note Requires rebuilding the mapping tables.
         *
         * @param transform Desired transformation.
         * @return The set transformation.
         */
        CvAffine3T transform(const CvAffine3T &transform);

        /**
         * @brief Set the transformation using an affine matrix.
         *
         * @note Requires rebuilding the mapping tables.
         *
         * @param affine Desired affine transformation matrix.
         * @return The set transformation.
         */
        CvAffine3T transform(const CvAffine3T::Mat4 &affine);

        /**
         * @brief Set the transformation using rotation matrix and translation vector.
         *
         * @note Requires rebuilding the mapping tables.
         *
         * @param rotation Desired rotation matrix.
         * @param translation Desired translation vector.
         * @return The set transformation.
         */
        CvAffine3T transform(const CvAffine3T::Mat3 &rotation, const CvAffine3T::Vec3 &translation = CvAffine3T::Vec3(0, 0, 0));

        /**
         * @brief Set the transformation using a rotation vector and translation vector.
         *
         * @note Requires rebuilding the mapping tables.
         *
         * @param rotation Desired rotation vector (axis-angle, Rodrigues).
         * @param translation Desired translation vector.
         * @return The set transformation.
         */
        CvAffine3T transform(const CvAffine3T::Vec3 &rotation, const CvAffine3T::Vec3 &translation = CvAffine3T::Vec3(0, 0, 0));


        /**
         * @brief Stitches the provided input images into a single panoramic image.
         *
         * Exposure values are used to correct the brightness of each input image and adjust
         * the overall exposure of the panorama. Internal data structures are built or rebuilt
         * as necessary before stitching.
         *
         * @param images Vector of input images to be stitched.
         * @param exposure Vector of exposure values corresponding to each input image.
         * @param pano Output panoramic image.
         */
        void stitch(const std::vector<cv::Mat> &images, const std::vector<ScalarT> &exposure, cv::Mat &pano);

        /**
         * @brief Stitches the provided input images into a single panoramic image.
         *
         * Overload for stitching without exposure correction.
         *
         * @param images Vector of input images to be stitched.
         * @param pano Output panoramic image.
         */
        void stitch(const std::vector<cv::Mat> &images, cv::Mat &pano);

        /**
         * @brief Stitches the provided input images into a single panoramic image.
         *
         * Overload for stitching with a uniform depth value.
         *
         * @warning Triggers rebuild of mapping tables, making it significantly slower than
         * other stitch() overloads.
         *
         * @param images Vector of input images to be stitched.
         * @param exposure Vector of exposure values corresponding to each input image.
         * @param depth Uniform depth value to be used for all points.
         * @param pano Output panoramic image.
         *
         * @see setDepth
         */
        void stitch(const std::vector<cv::Mat> &images, const std::vector<ScalarT> &exposure, ScalarT depth, cv::Mat &pano);

        /**
         * @brief Stitches the provided input images into a single panoramic image.
         *
         * Overload for stitching with a per-point depth map.
         *
         * @warning Triggers rebuild of mapping tables, making it significantly slower than
         * other stitch() overloads.
         *
         * @param images Vector of input images to be stitched.
         * @param exposure Vector of exposure values corresponding to each input image.
         * @param depth Per-point depth map to be used for all points.
         * @param pano Output panoramic image.
         *
         * @see setDepth
         */
        void stitch(const std::vector<cv::Mat> &images, const std::vector<ScalarT> &exposure, const cv::Mat &depth, cv::Mat &pano);


        /**
         * @brief Builds all necessary internal data structures required for stitching.
         *
         * Calls the individual build functions for maps, vignettes, mask, and mirrors as needed.
         *
         * @note This function is called internally before stitching if any internal data structures need to be rebuilt.
         *
         * @return True if all internal data structures were built successfully, false otherwise.
         */
        bool buildInternals();

    private:

        /**
         * @brief Builds the lookup tables and initial proximity-based weights.
         *
         * Generates a spherical point cloud based on the current resolution and field of view,
         * transforms the points using the current transformation, and builds the mapping tables
         * and initial proximity-based weights in the calibration.
         *
         * @note Triggers mask rebuild.
         *
         * @return true If the mapping tables were built successfully.
         * @return false If there was an error building the mapping tables.
         */
        bool buildMaps();

        /**
         * @brief Builds the vignette maps.
         *
         * Generates vignette maps based on the current vignette threshold and calibration data.
         *
         * @note Triggers mask and mirror rebuild.
         */
        void buildVignettes();

        /**
         * @brief Builds the masks and finalizes the weights.
         *
         * Generates masks from vignette maps and applies them to normalize and combine the proximity-based weights.
         * Updates the vignette maps if the mask is used as vignette.
         *
         * @note Triggers mirror rebuild.
         */
        void buildMask();

        /**
         * @brief Builds the vignette map and weight mirrors for multi-channel images.
         *
         * Generates multi-channel versions of the vignette maps and weights to support multi-channel images.
         */
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
        std::vector<cv::UMat> weights, weights_base, weights_raw;

        // Additional resources
        std::unique_ptr<DeviceData> dev;
    };

}
