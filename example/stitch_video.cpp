#include <panoweave.hpp>

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

int main(int argc, char **argv)
{
    argparse::ArgumentParser argp("Video Stitcher Example");
    argp.add_argument("output")
        .help("Path to output video");
    argp.add_argument("calibration")
        .help("Path to basalt calibration.json for camera calibration");
    argp.add_argument("inputs")
        .nargs(argparse::nargs_pattern::at_least_one)
        .append()
        .help("Path to input videos");
    try
    {
        argp.parse_args(argc, argv);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << argp;
        return 1;
    }
    std::string calib_path = argp.get<std::string>("calibration");
    std::vector<std::string> input_paths = argp.get<std::vector<std::string>>("inputs");
    std::string output_path = argp.get<std::string>("output");

    auto stitcher = std::make_unique<PanoWeave::PanoWeave>(calib_path);
    stitcher->setResolution(4800, 2400);
    stitcher->setDepth(1);

    bool ret = true;

    uint8_t num_cams = 2; //TODO FIX THIS
    std::vector<cv::VideoCapture> caps;
    caps.resize(num_cams);

    for (uint8_t i = 0; i < num_cams && ret; ++i)
    {
        ret = caps[i].open(input_paths[i]);
    }
    if (!ret)
    {
        std::cerr << "Unable to open video file" << std::endl;
        return 1;
    }

    std::vector<cv::Mat> imgs;
    imgs.resize(num_cams);

    for (uint8_t i = 0; i < num_cams; ++i)
    {
        cv::namedWindow("Input " + std::to_string(i), cv::WINDOW_NORMAL);
    }
    cv::namedWindow("Output", cv::WINDOW_NORMAL);
    cv::createTrackbar("ResX", "Output", nullptr, 4000, [](int pos, void *stitcher) -> void { static_cast<PanoWeave::PanoWeave*>(stitcher)->width(pos); }, stitcher.get());
    cv::createTrackbar("ResY", "Output", nullptr, 2000, [](int pos, void *stitcher) -> void { static_cast<PanoWeave::PanoWeave*>(stitcher)->height(pos); }, stitcher.get());
    cv::createTrackbar("FovX", "Output", nullptr, 360, [](int pos, void *stitcher) -> void { static_cast<PanoWeave::PanoWeave*>(stitcher)->fovX(pos * M_PI / 180); }, stitcher.get());
    cv::createTrackbar("FovY", "Output", nullptr, 180, [](int pos, void *stitcher) -> void { static_cast<PanoWeave::PanoWeave*>(stitcher)->fovY(pos * M_PI / 180); }, stitcher.get());
    cv::createTrackbar("Depth", "Output", nullptr, 100, [](int pos, void *stitcher) -> void { static_cast<PanoWeave::PanoWeave*>(stitcher)->setDepth(pos > 0 ? pos : 0.1); }, stitcher.get());
    cv::createTrackbar("VignThresh", "Output", nullptr, 100, [](int pos, void *stitcher) -> void { static_cast<PanoWeave::PanoWeave*>(stitcher)->vignetteThreshold(pos / 100.); }, stitcher.get());

    while (readInputs(caps, imgs))
    {
        for (uint8_t i = 0; i < num_cams; ++i)
        {
            cv::imshow("Input " + std::to_string(i), imgs[i]);
        }

        cv::Mat pano;
        auto start = std::chrono::steady_clock::now();
        stitcher->weave(imgs, pano);
        auto end = std::chrono::steady_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count() << std::endl;
        cv::imshow("Output", pano);

        cv::waitKey(1);
    }
}
