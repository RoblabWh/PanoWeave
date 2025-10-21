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
            .def(pybind11::init<>())
            .def(pybind11::init<const std::string &>(), pybind11::arg("calibration_filepath"))
            .def(pybind11::init<const std::string &, ScalarT>(), pybind11::arg("calibration_filepath"), pybind11::arg("depth"))
            .def(pybind11::init<const std::string &, PyDepth>(), pybind11::arg("calibration_filepath"), pybind11::arg("depth"))

            .def("loadCalibration", static_cast<void (Stitcher::*)(const std::string &)>(&Stitcher::loadCalibration))
            .def("setDepth", static_cast<void (Stitcher::*)(ScalarT)>(&Stitcher::setDepth))
            .def("setDepth", static_cast<void (PyStitcher::*)(PyDepth)>(&PyStitcher::setDepth))

            .def("resolution", static_cast<std::tuple<int, int> (PyStitcher::*)() const>(&PyStitcher::resolution))
            .def("resolution", static_cast<std::tuple<int, int> (PyStitcher::*)(const std::tuple<int, int> &)>(&PyStitcher::resolution))
            .def("resolution", static_cast<std::tuple<int, int> (PyStitcher::*)(int, int)>(&PyStitcher::resolution))
            .def("width", static_cast<int (Stitcher::*)() const>(&Stitcher::width))
            .def("width", static_cast<int (Stitcher::*)(int)>(&Stitcher::width))
            .def("height", static_cast<int (Stitcher::*)() const>(&Stitcher::height))
            .def("height", static_cast<int (Stitcher::*)(int)>(&Stitcher::height))

            .def("fov", static_cast<std::tuple<ScalarT, ScalarT> (PyStitcher::*)() const>(&PyStitcher::fov))
            .def("fov", static_cast<std::tuple<ScalarT, ScalarT> (PyStitcher::*)(ScalarT, ScalarT)>(&PyStitcher::fov))
            .def("fov", static_cast<std::tuple<ScalarT, ScalarT> (PyStitcher::*)(const std::tuple<ScalarT, ScalarT> &)>(&PyStitcher::fov))
            .def("fovX", static_cast<ScalarT (Stitcher::*)() const>(&Stitcher::fovX))
            .def("fovX", static_cast<ScalarT (Stitcher::*)(ScalarT)>(&Stitcher::fovX))
            .def("fovY", static_cast<ScalarT (Stitcher::*)() const>(&Stitcher::fovY))
            .def("fovY", static_cast<ScalarT (Stitcher::*)(ScalarT)>(&Stitcher::fovY))

            .def("vignetteThreshold", static_cast<ScalarT (Stitcher::*)() const>(&Stitcher::vignetteThreshold))
            .def("vignetteThreshold", static_cast<ScalarT (Stitcher::*)(ScalarT)>(&Stitcher::vignetteThreshold))

            .def("useMaskAsVignette", static_cast<bool (Stitcher::*)() const>(&Stitcher::useMaskAsVignette))
            .def("useMaskAsVignette", static_cast<bool (Stitcher::*)(bool)>(&Stitcher::useMaskAsVignette))

            .def("transform", static_cast<ndarray<ScalarT> (PyStitcher::*)() const>(&PyStitcher::transform))
            .def("transform", static_cast<ndarray<ScalarT> (PyStitcher::*)(const ndarray<ScalarT> &, const std::optional<std::tuple<ScalarT, ScalarT, ScalarT>> &)>(&PyStitcher::transform),
                 pybind11::arg("matrix"), pybind11::arg("translation") = pybind11::none())
            .def("transform", static_cast<ndarray<ScalarT> (PyStitcher::*)(const std::tuple<ScalarT, ScalarT, ScalarT> &, const std::optional<std::tuple<ScalarT, ScalarT, ScalarT>> &)>(&PyStitcher::transform),
                 pybind11::arg("rotation"), pybind11::arg("translation") = pybind11::none())

            .def("stitch", static_cast<PyImage (PyStitcher::*)(std::vector<PyImage> &)>(&PyStitcher::stitch))
            .def("stitch", static_cast<PyImage (PyStitcher::*)(std::vector<PyImage> &, const std::vector<ScalarT> &)>(&PyStitcher::stitch))
            .def("stitch", static_cast<PyImage (PyStitcher::*)(std::vector<PyImage> &, const std::vector<ScalarT> &, ScalarT)>(&PyStitcher::stitch))
            .def("stitch", static_cast<PyImage (PyStitcher::*)(std::vector<PyImage> &, const std::vector<ScalarT> &, PyDepth)>(&PyStitcher::stitch));
    }

}
