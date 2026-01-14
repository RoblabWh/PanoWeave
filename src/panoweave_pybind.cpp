#include "panoweave.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace panoweave
{
    template <typename T>
    using ndarray = pybind11::array_t<T, pybind11::array::c_style | pybind11::array::forcecast>;
    using PyImage = ndarray<uint8_t>;
    using PyDepth = ndarray<ScalarT>;

    class PyStitcher : public Stitcher
    {
    public:
        PyStitcher() : Stitcher() {}
        PyStitcher(const std::string &calibration_filepath)
            : Stitcher(calibration_filepath) {}
        PyStitcher(const std::string &calibration_filepath, ScalarT depth)
            : Stitcher(calibration_filepath, depth) {}
        PyStitcher(const std::string &calibration_filepath, PyDepth depth)
            : Stitcher(calibration_filepath, cv::Mat(depth.shape(0), depth.shape(1), CvMatT(1), depth.mutable_data())) {}

        void setDepth(PyDepth depth)
        {
            Stitcher::setDepth(cv::Mat(depth.shape(0), depth.shape(1), CvMatT(1), depth.mutable_data()));
        }

        std::tuple<int, int> resolution() const
        {
            return std::make_tuple(this->width(), this->height());
        }
        std::tuple<int, int> resolution(const std::tuple<int, int> &resolution)
        {
            Stitcher::resolution(std::get<0>(resolution), std::get<1>(resolution));
            return resolution;
        }
        std::tuple<int, int> resolution(int width, int height)
        {
            Stitcher::resolution(width, height);
            return this->resolution();
        }

        std::tuple<ScalarT, ScalarT> fov() const
        {
            return std::make_tuple(this->fovX(), this->fovY());
        }
        std::tuple<ScalarT, ScalarT> fov(ScalarT fov_x, ScalarT fov_y)
        {
            Stitcher::fov(fov_x, fov_y);
            return this->fov();
        }
        std::tuple<ScalarT, ScalarT> fov(const std::tuple<ScalarT, ScalarT> &fov)
        {
            Stitcher::fov(std::get<0>(fov), std::get<1>(fov));
            return fov;
        }

        ndarray<ScalarT> transform() const
        {
            ndarray<ScalarT> tf({4UL, 4UL});
            std::memcpy(tf.mutable_data(), Stitcher::transform().matrix.val, sizeof(ScalarT) * 16);
            return tf;
        }
        ndarray<ScalarT> transform(const ndarray<ScalarT> &matrix,
                                   const std::optional<std::tuple<ScalarT, ScalarT, ScalarT>> &translation)
        {
            if (matrix.shape(0) == 3 && matrix.shape(1) == 3)
            {
                CvAffine3T::Mat3 rmat;
                std::memcpy(rmat.val, matrix.data(), sizeof(ScalarT) * 9);
                if (translation.has_value())
                    Stitcher::transform(rmat, CvAffine3T::Vec3(std::get<0>(*translation), std::get<1>(*translation), std::get<2>(*translation)));
                else
                    Stitcher::transform(rmat);
            }
            else if (matrix.shape(0) == 3 && matrix.shape(1) == 4)
            {
                CvAffine3T::Mat3 rot_mat;
                for (uint i = 0; i < 3; ++i)
                    std::memcpy(rot_mat.val + 3 * i, matrix.data() + 3 * i, sizeof(ScalarT) * 3);
                Stitcher::transform(rot_mat, CvAffine3T::Vec3(matrix.at(0, 3), matrix.at(1, 3), matrix.at(2, 3)));
            }
            else if (matrix.shape(0) == 4 && matrix.shape(1) == 4)
            {
                CvAffine3T::Mat4 affine;
                std::memcpy(affine.val, matrix.data(), sizeof(ScalarT) * 16);
                Stitcher::transform(affine);
            }
            else
                throw std::runtime_error("Invalid matrix shape for transform. Expected 3x3, 3x4, or 4x4 matrix.");
            return this->transform();
        }
        ndarray<ScalarT> transform(const std::tuple<ScalarT, ScalarT, ScalarT> &rotation,
                                   const std::optional<std::tuple<ScalarT, ScalarT, ScalarT>> &translation)
        {
            if (translation.has_value())
                Stitcher::transform(CvAffine3T::Vec3(std::get<0>(rotation), std::get<1>(rotation), std::get<2>(rotation)),
                                     CvAffine3T::Vec3(std::get<0>(*translation), std::get<1>(*translation), std::get<2>(*translation)));
            else
                Stitcher::transform(CvAffine3T::Vec3(std::get<0>(rotation), std::get<1>(rotation), std::get<2>(rotation)));
            return this->transform();
        }

        PyImage stitch(std::vector<PyImage> &images)
        {
            return this->stitch(images, std::vector<ScalarT>(images.size(), 1.0));
        }
        PyImage stitch(std::vector<PyImage> &images, const std::vector<ScalarT> &exposure)
        {
            std::vector<cv::Mat> cv_images;
            cv_images.reserve(images.size());
            for (auto &img : images)
            {
                cv_images.emplace_back(img.shape(0), img.shape(1), CV_8UC(img.shape(2)), img.mutable_data());
            }

            auto pano = PyImage({static_cast<size_t>(Stitcher::height()), static_cast<size_t>(Stitcher::width()), static_cast<size_t>(images[0].shape(2))});
            auto cv_pano = cv::Mat(pano.shape(0), pano.shape(1), CV_8UC(pano.shape(2)), pano.mutable_data());

            Stitcher::stitch(cv_images, exposure, cv_pano);
            return pano;
        }
        PyImage stitch(std::vector<PyImage> &images, const std::vector<ScalarT> &exposure, ScalarT depth)
        {
            Stitcher::setDepth(depth);
            return this->stitch(images, exposure);
        }
        PyImage stitch(std::vector<PyImage> &images, const std::vector<ScalarT> &exposure, PyDepth depth)
        {
            this->setDepth(depth);
            return this->stitch(images, exposure);
        }
    };

    PYBIND11_MODULE(panoweave, m)
    {
        m.doc() = "PanoWeave Python Bindings";

        pybind11::class_<PyStitcher>(m, "Stitcher")
            .def(pybind11::init<>(),
                R"pbdoc(
                    Constructs an empty Stitcher object.

                    Initializes an empty Stitcher object. Calibration data must be loaded,
                    resolution set, and depth defined before stitching operations can be performed.

                    Returns:
                        Stitcher: An uninitialized Stitcher object.
                )pbdoc")
            .def(pybind11::init<const std::string &>(), pybind11::arg("calibration_filepath"),
                R"pbdoc(
                    Constructs a Stitcher object with the specified calibration file.

                    Loads camera calibration data from the given file path.

                    Args:
                        calibration_filepath (str): Path to the camera calibration data file.

                    Raises:
                        RuntimeError: If the file cannot be opened or the data is invalid.

                    See Also:
                        loadCalibration
                )pbdoc")
            .def(pybind11::init<const std::string &, ScalarT>(), pybind11::arg("calibration_filepath"), pybind11::arg("depth"),
                R"pbdoc(
                    Constructs a Stitcher object with the specified calibration file and uniform depth.

                    Loads camera calibration and sets a uniform depth value.

                    Args:
                        calibration_filepath (str): Path to the camera calibration data file.
                        depth (float): Uniform depth value.

                    Raises:
                        RuntimeError: If the file cannot be opened or the data is invalid.

                    See Also:
                        loadCalibration, setDepth
                )pbdoc")
            .def(pybind11::init<const std::string &, PyDepth>(), pybind11::arg("calibration_filepath"), pybind11::arg("depth"),
                R"pbdoc(
                    Constructs a Stitcher object with the specified calibration file and per-point depth.

                    Loads camera calibration and sets a per-point depth map.

                    Args:
                        calibration_filepath (str): Path to the camera calibration data file.
                        depth (numpy.ndarray): Per-point depth map.

                    Raises:
                        RuntimeError: If the file cannot be opened or the data is invalid.

                    See Also:
                        loadCalibration, setDepth
                )pbdoc")

            .def("loadCalibration", static_cast<void (Stitcher::*)(const std::string &)>(&Stitcher::loadCalibration), pybind11::arg("calibration_filepath"),
                R"pbdoc(
                    Loads camera calibration data from the specified file.

                    Reads calibration parameters from a Basalt calibration.json file.

                    Note:
                        Requires rebuilding of all internal data structures.

                    Args:
                        calibration_filepath (str): Path to the camera calibration data file.

                    Raises:
                        RuntimeError: If the file cannot be opened or the data is invalid.
                )pbdoc")
            .def("setDepth", static_cast<void (Stitcher::*)(ScalarT)>(&Stitcher::setDepth), pybind11::arg("depth"),
                R"pbdoc(
                    Sets a uniform depth for the panorama.

                    Uses the same depth value for all points in the panorama.

                    Note:
                        Requires rebuilding the mapping tables.

                    Args:
                        depth (float): Uniform depth value.
                )pbdoc")
            .def("setDepth", static_cast<void (PyStitcher::*)(PyDepth)>(&PyStitcher::setDepth), pybind11::arg("depth"),
                R"pbdoc(
                    Sets a per-point depth map for the panorama.

                    The depth map provides depth information for each point in the panorama. Its resolution must match the output resolution of the panorama.

                    Note:
                        Requires rebuilding the mapping tables.

                    Args:
                        depth: Per-point depth map.
                )pbdoc")

            .def("resolution", static_cast<std::tuple<int, int> (PyStitcher::*)() const>(&PyStitcher::resolution),
                R"pbdoc(
                    Get the current output resolution for the panorama.

                    Returns:
                        tuple[int,int]: The current resolution.
                )pbdoc")
            .def("resolution", static_cast<std::tuple<int, int> (PyStitcher::*)(const std::tuple<int, int> &)>(&PyStitcher::resolution), pybind11::arg("resolution"),
                R"pbdoc(
                    Set the output resolution for the panorama.

                    Note:
                        Requires rebuilding the mapping tables.

                    Args:
                        resolution (tuple[int,int]): Desired resolution as (width, height).

                    Returns:
                        tuple[int,int]: The set resolution.
                )pbdoc")
            .def("resolution", static_cast<std::tuple<int, int> (PyStitcher::*)(int, int)>(&PyStitcher::resolution), pybind11::arg("width"), pybind11::arg("height"),
                R"pbdoc(
                    Set the output resolution for the panorama.

                    Overload accepting width and height as separate parameters.

                    Note:
                        Requires rebuilding the mapping tables.

                    Args:
                        width (int): Desired width.
                        height (int): Desired height.

                    Returns:
                        tuple[int,int]: The set resolution.
                )pbdoc")
            .def("width", static_cast<int (Stitcher::*)() const>(&Stitcher::width),
                R"pbdoc(
                    Get the current output width for the panorama.

                    Returns:
                        int: The current width.
                )pbdoc")
            .def("width", static_cast<int (Stitcher::*)(int)>(&Stitcher::width), pybind11::arg("width"),
                R"pbdoc(
                    Set the output width for the panorama.

                    Note:
                        Requires rebuilding the mapping tables.

                    Args:
                        width (int): Desired width.

                    Returns:
                        int: The set width.
                )pbdoc")
            .def("height", static_cast<int (Stitcher::*)() const>(&Stitcher::height),
                R"pbdoc(
                    Get the current output height for the panorama.

                    Returns:
                        int: The current height.
                )pbdoc")
            .def("height", static_cast<int (Stitcher::*)(int)>(&Stitcher::height), pybind11::arg("height"),
                R"pbdoc(
                    Set the output height for the panorama.

                    Note:
                        Requires rebuilding the mapping tables.

                    Args:
                        height (int): Desired height.

                    Returns:
                        int: The set height.
                )pbdoc")

            .def("fov", static_cast<std::tuple<ScalarT, ScalarT> (PyStitcher::*)() const>(&PyStitcher::fov),
                R"pbdoc(
                    Get the current field of view (FOV) for the panorama.

                    Returns:
                        tuple[float,float]: The current FOV.
                )pbdoc")
            .def("fov", static_cast<std::tuple<ScalarT, ScalarT> (PyStitcher::*)(const std::tuple<ScalarT, ScalarT> &)>(&PyStitcher::fov), pybind11::arg("fov"),
                R"pbdoc(
                    Set the field of view (FOV) for the panorama.

                    The FOV must be in the range (0, 2*Pi] for horizontal and (0, Pi] for vertical.

                    Note:
                        Requires rebuilding the mapping tables.

                    Args:
                        fov (tuple[float,float]): Desired FOV.

                    Returns:
                        tuple[float,float]: The set FOV.
                )pbdoc")
            .def("fov", static_cast<std::tuple<ScalarT, ScalarT> (PyStitcher::*)(ScalarT, ScalarT)>(&PyStitcher::fov), pybind11::arg("fov_x"), pybind11::arg("fov_y"),
                R"pbdoc(
                    Set the field of view (FOV) for the panorama.

                    Overload accepting horizontal and vertical FOV as separate parameters.

                    Note:
                        Requires rebuilding the mapping tables.

                    Args:
                        fov_x (float): Desired horizontal FOV.
                        fov_y (float): Desired vertical FOV.

                    Returns:
                        tuple[float,float]: The set FOV.
                )pbdoc")
            .def("fovX", static_cast<ScalarT (Stitcher::*)() const>(&Stitcher::fovX),
                R"pbdoc(
                    Get the current horizontal field of view (FOV) for the panorama.

                    Returns:
                        float: The current horizontal FOV.
                )pbdoc")
            .def("fovX", static_cast<ScalarT (Stitcher::*)(ScalarT)>(&Stitcher::fovX), pybind11::arg("fov"),
                R"pbdoc(
                    Set the horizontal field of view (FOV) for the panorama.

                    The horizontal FOV must be in the range (0, 2*Pi].

                    Note:
                        Requires rebuilding the mapping tables.

                    Args:
                        fov (float): Desired horizontal FOV.

                    Returns:
                        float: The set horizontal FOV.
                )pbdoc")
            .def("fovY", static_cast<ScalarT (Stitcher::*)() const>(&Stitcher::fovY),
                R"pbdoc(
                    Get the current vertical field of view (FOV) for the panorama.

                    Returns:
                        float: The current vertical FOV.
                )pbdoc")
            .def("fovY", static_cast<ScalarT (Stitcher::*)(ScalarT)>(&Stitcher::fovY), pybind11::arg("fov"),
                R"pbdoc(
                    Set the vertical field of view (FOV) for the panorama.

                    The vertical FOV must be in the range (0, Pi].

                    Note:
                        Requires rebuilding the mapping tables.

                    Args:
                        fov (float): Desired vertical FOV.

                    Returns:
                        float: The set vertical FOV.
                )pbdoc")

            .def("vignetteThreshold", static_cast<ScalarT (Stitcher::*)() const>(&Stitcher::vignetteThreshold),
                R"pbdoc(
                    Get the current vignette threshold.

                    Returns:
                        float: The current vignette threshold.
                )pbdoc")
            .def("vignetteThreshold", static_cast<ScalarT (Stitcher::*)(ScalarT)>(&Stitcher::vignetteThreshold), pybind11::arg("threshold"),
                R"pbdoc(
                    Set the vignette threshold used to generate vignette maps.

                    The vignette maps are used to compensate for lens vignetting in input images.
                    Values should be in the range [0.0, 1.0]. A lower threshold means pixels farther from the center are considered valid.

                    Note:
                        Requires rebuilding the vignette maps.

                    Args:
                        threshold (float): Desired vignette threshold.

                    Returns:
                        float: The set vignette threshold.
                )pbdoc")

            .def("useMaskAsVignette", static_cast<bool (Stitcher::*)() const>(&Stitcher::useMaskAsVignette),
                R"pbdoc(
                    Gets whether the mask is used as vignette.

                    Returns:
                        bool: True if the mask is used as vignette, false otherwise.
                )pbdoc")
            .def("useMaskAsVignette", static_cast<bool (Stitcher::*)(bool)>(&Stitcher::useMaskAsVignette), pybind11::arg("use"),
                R"pbdoc(
                    Sets whether to use the mask as vignette.

                    If set to true, the mask will replace the vignette maps.
                    This is useful when vignette correction is not desired (for example, if vignetting calibration quality is insufficient),
                    and the mask should define valid pixel regions instead.

                    Note:
                        Requires rebuilding the mask or vignette maps depending on the value.

                    Args:
                        use (bool): True to use the mask as vignette, false otherwise.

                    Returns:
                        bool: The set value.
                )pbdoc")

            .def("transform", static_cast<ndarray<ScalarT> (PyStitcher::*)() const>(&PyStitcher::transform),
                R"pbdoc(
                    Get the current transformation from body to calibration coordinate system.

                    Returns:
                        numpy.ndarray: The current transformation as a 4x4 matrix.
                )pbdoc")
            .def("transform", static_cast<ndarray<ScalarT> (PyStitcher::*)(const ndarray<ScalarT> &, const std::optional<std::tuple<ScalarT, ScalarT, ScalarT>> &)>(&PyStitcher::transform),
                 pybind11::arg("matrix"), pybind11::arg("translation") = pybind11::none(),
                R"pbdoc(
                    Set the transformation from body to calibration coordinate system.

                    This transformation is applied to spherical points before projecting them into camera views.
                    The Basalt calibration coordinate system is either the IMU or the first camera coordinate system,
                    depending on the calibration type.

                    Note:
                        Requires rebuilding the mapping tables.

                    Args:
                        matrix (numpy.ndarray): Desired transformation matrix (3x3, 3x4, or 4x4).
                        translation (tuple[float,float,float], optional): Desired translation vector.

                    Raises:
                        RuntimeError: If the matrix shape is invalid.

                    Returns:
                        numpy.ndarray: The set transformation as a 4x4 matrix.
                )pbdoc")
            .def("transform", static_cast<ndarray<ScalarT> (PyStitcher::*)(const std::tuple<ScalarT, ScalarT, ScalarT> &, const std::optional<std::tuple<ScalarT, ScalarT, ScalarT>> &)>(&PyStitcher::transform),
                 pybind11::arg("rotation"), pybind11::arg("translation") = pybind11::none(),
                R"pbdoc(
                    Set the transformation using a rotation vector and translation vector.

                    Note:
                        Requires rebuilding the mapping tables.

                    Args:
                        rotation (tuple[float,float,float]): Desired rotation vector (axis-angle, Rodrigues).
                        translation (tuple[float,float,float], optional): Desired translation vector.

                    Returns:
                        numpy.ndarray: The set transformation as a 4x4 matrix.
                )pbdoc")

            .def("stitch", static_cast<PyImage (PyStitcher::*)(std::vector<PyImage> &, const std::vector<ScalarT> &)>(&PyStitcher::stitch),
                 pybind11::arg("images"), pybind11::arg("exposure"),
                R"pbdoc(
                    Stitches the provided input images into a single panoramic image.

                    Exposure values are used to correct the brightness of each input image and adjust the overall exposure of the panorama.
                    Internal data structures are built or rebuilt as necessary before stitching.

                    Args:
                        images (list[numpy.ndarray]): Vector of input images to be stitched.
                        exposure (list[float]): Vector of exposure values corresponding to each input image.

                    Returns:
                        numpy.ndarray: Output panoramic image.
                )pbdoc")
            .def("stitch", static_cast<PyImage (PyStitcher::*)(std::vector<PyImage> &)>(&PyStitcher::stitch), pybind11::arg("images"),
                R"pbdoc(
                    Stitches the provided input images into a single panoramic image.

                    Overload for stitching without exposure correction.

                    Args:
                        images (list[numpy.ndarray]): Vector of input images to be stitched.

                    Returns:
                        numpy.ndarray: Output panoramic image.
                )pbdoc")
            .def("stitch", static_cast<PyImage (PyStitcher::*)(std::vector<PyImage> &, const std::vector<ScalarT> &, ScalarT)>(&PyStitcher::stitch),
                 pybind11::arg("images"), pybind11::arg("exposure"), pybind11::arg("depth"),
                R"pbdoc(
                    Stitches the provided input images into a single panoramic image.

                    Overload for stitching with a uniform depth value.

                    Warning:
                        Triggers rebuild of mapping tables, making it significantly slower than other stitch() overloads.

                    Args:
                        images (list[numpy.ndarray]): Vector of input images to be stitched.
                        exposure (list[float]): Vector of exposure values corresponding to each input image.
                        depth (float): Uniform depth value to be used for all points.

                    Returns:
                        numpy.ndarray: Output panoramic image.

                    See Also:
                        setDepth
                )pbdoc")
            .def("stitch", static_cast<PyImage (PyStitcher::*)(std::vector<PyImage> &, const std::vector<ScalarT> &, PyDepth)>(&PyStitcher::stitch),
                 pybind11::arg("images"), pybind11::arg("exposure"), pybind11::arg("depth"),
                R"pbdoc(
                    Stitches the provided input images into a single panoramic image.

                    Overload for stitching with a per-point depth map.

                    Warning:
                        Triggers rebuild of mapping tables, making it significantly slower than other stitch() overloads.

                    Args:
                        images (list[numpy.ndarray]): Vector of input images to be stitched.
                        exposure (list[float]): Vector of exposure values corresponding to each input image.
                        depth (numpy.ndarray): Per-point depth map to be used for all points.

                    Returns:
                        numpy.ndarray: Output panoramic image.

                    See Also:
                        setDepth
                )pbdoc")

            .doc() = R"pbdoc(
                    Stitches multiple camera images into a single spherical panorama using Basalt calibration data.

                    The Stitcher class supports setting resolution, field of view, depth, and transformation of the spherical camera.
                    Depth values can be provided as a uniform scalar or a per-pixel depth map to cover simple or complex use cases.
                    Exposure times of the input images can optionally be used to adjust the input images to a common exposure.
                    Images with one (mono), three (BGR, RGB), or four (BGRA, RGBA) channels are supported without extra configuration.

                    Note:
                        Internal data structures are built lazily, i.e., only when required at the beginning of stitching. This avoids unnecessary computation when multiple parameters are changed before stitching.

                    Example:
                        stitcher = panoweave.Stitcher("calibration.json")
                        stitcher.resolution(2160, 1080)
                        stitcher.setDepth(5.0)
                        pano = stitcher.stitch(images, exposures)
                )pbdoc";
    }

}
