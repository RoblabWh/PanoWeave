#include <panoweave.hpp>

#include <opencv2/core/eigen.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <argparse/argparse.hpp>

bool readInputs(std::vector<cv::VideoCapture> &caps, std::vector<cv::Mat> &imgs)
{
    imgs.resize(caps.size());
    bool ret = true;
    for (uint8_t i = 0; i < caps.size() && ret; ++i)
    {
        ret = caps[i].read(imgs[i]);
    }
    return ret;
}

struct CallbackData {
    std::shared_ptr<panoweave::Stitcher> stitcher;
    Eigen::Vector3f rpy;
};

cv::Affine3f::Mat3 rpyToMat(const Eigen::Vector3f &rpy)
{
        cv::Affine3f::Mat3 rotmat;
        cv::eigen2cv((
                      Eigen::AngleAxisf(rpy.x(), Eigen::Vector3f::UnitX())
                    * Eigen::AngleAxisf(rpy.y(), Eigen::Vector3f::UnitY())
                    * Eigen::AngleAxisf(rpy.z(), Eigen::Vector3f::UnitZ())
                ).toRotationMatrix(), rotmat);
        return rotmat;
}

int main(int argc, char **argv)
{
    argparse::ArgumentParser argp("Video Stitcher Example");
    argp.add_argument("inputs")
        .nargs(argparse::nargs_pattern::at_least_one)
        .append()
        .help("path to input videos");
    argp.add_argument("--calibration", "--calib", "-c")
        .help("path to basalt calibration.json for camera calibration");
    try
    {
        argp.parse_args(argc, argv);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << argp;
        return 1;
    }
    std::string calib_path = argp.get<std::string>("calibration");
    std::vector<std::string> input_paths = argp.get<std::vector<std::string>>("inputs");

    auto stitcher = std::make_shared<panoweave::Stitcher>(calib_path);
    stitcher->resolution(3200, 1600);
    stitcher->setDepth(1);

    uint8_t num_cams = input_paths.size();
    std::vector<cv::VideoCapture> caps;
    caps.resize(num_cams);

    for (uint8_t i = 0; i < num_cams; ++i)
    {
        if (!caps[i].open(input_paths[i]))
        {
            std::cerr << "Unable to open video file" << std::endl;
            return 1;
        }
    }

    std::vector<cv::Mat> imgs;
    imgs.resize(num_cams);

    auto cb_data = new CallbackData{stitcher, Eigen::Vector3f::Zero()};
    for (uint8_t i = 0; i < num_cams; ++i)
    {
        cv::namedWindow("Input " + std::to_string(i), cv::WINDOW_NORMAL);
    }
    cv::namedWindow("Output", cv::WINDOW_NORMAL);
    cv::createTrackbar("Resolution X (px)", "Output", nullptr, 4000, [](int pos, void *stitcher) -> void
                       { reinterpret_cast<panoweave::Stitcher *>(stitcher)->width(pos > 0 ? pos : 1); }, stitcher.get());
    cv::createTrackbar("Resolution Y (px)", "Output", nullptr, 2000, [](int pos, void *stitcher) -> void
                       { reinterpret_cast<panoweave::Stitcher *>(stitcher)->height(pos > 0 ? pos : 1); }, stitcher.get());
    cv::createTrackbar("FOV X (deg)", "Output", nullptr, 360, [](int pos, void *stitcher) -> void
                       { reinterpret_cast<panoweave::Stitcher *>(stitcher)->fovX(pos * M_PI / 180); }, stitcher.get());
    cv::createTrackbar("FOV Y (deg)", "Output", nullptr, 180, [](int pos, void *stitcher) -> void
                       { reinterpret_cast<panoweave::Stitcher *>(stitcher)->fovY(pos * M_PI / 180); }, stitcher.get());
    cv::createTrackbar("Depth (cm)", "Output", nullptr, 1000, [](int pos, void *stitcher) -> void
                       { reinterpret_cast<panoweave::Stitcher *>(stitcher)->setDepth(pos > 0 ? pos / 100. : 0.1); }, stitcher.get());
    cv::createTrackbar("Vignette Threshold (%)", "Output", nullptr, 100, [](int pos, void *stitcher) -> void
                       { reinterpret_cast<panoweave::Stitcher *>(stitcher)->vignetteThreshold(pos / 100.); }, stitcher.get());
    cv::createTrackbar("Rotation X (deg)", "Output", nullptr, 360, [](int pos, void *data) -> void
                       { auto cb = reinterpret_cast<CallbackData *>(data);
                         cb->rpy.x() = pos / 180. * M_PI;
                         cb->stitcher->transform(rpyToMat(cb->rpy)); }, cb_data);
    cv::createTrackbar("Rotation Y (deg)", "Output", nullptr, 360, [](int pos, void *data) -> void
                       { auto cb = reinterpret_cast<CallbackData *>(data);
                         cb->rpy.y() = pos / 180. * M_PI;
                         cb->stitcher->transform(rpyToMat(cb->rpy)); }, cb_data);
    cv::createTrackbar("Rotation Z (deg)", "Output", nullptr, 360, [](int pos, void *data) -> void
                       { auto cb = reinterpret_cast<CallbackData *>(data);
                         cb->rpy.z() = pos / 180. * M_PI;
                         cb->stitcher->transform(rpyToMat(cb->rpy)); }, cb_data);

    while (readInputs(caps, imgs))
    {
        for (uint8_t i = 0; i < num_cams; ++i)
        {
            cv::imshow("Input " + std::to_string(i), imgs[i]);
        }

        cv::Mat pano;
        stitcher->stitch(imgs, pano);
        cv::imshow("Output", pano);
        if (cv::waitKey(1) == 27)
            break;
    }
}
