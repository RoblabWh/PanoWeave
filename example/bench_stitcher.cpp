#include <panoweave.hpp>

#include <opencv2/core/ocl.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <argparse/argparse.hpp>

bool readInputs(const std::vector<std::string> &inputs, std::vector<cv::Mat> &imgs)
{
    imgs.reserve(inputs.size());
    for (const auto &input : inputs)
    {
        cv::VideoCapture cap(input);
        if (!cap.isOpened())
        {
            std::cerr << "Unable to open video file: " << input << std::endl;
            return false;
        }
        cv::Mat img;
        if (!cap.read(img))
        {
            std::cerr << "Failed to read frame from: " << input << std::endl;
            return false;
        }
        imgs.push_back(img);
    }
    return true;
}

int main(int argc, char **argv)
{
    argparse::ArgumentParser argp("Video Stitcher Example");
    argp.add_argument("inputs")
        .nargs(argparse::nargs_pattern::at_least_one)
        .append()
        .help("Path to input videos");
    argp.add_argument("--calibration", "--calib", "-c")
        .help("Path to basalt calibration.json for camera calibration");
    argp.add_argument("--width", "-w")
        .scan<'u', size_t>()
        .default_value<size_t>(3200)
        .help("Output resolution width in pixels");
    argp.add_argument("--height", "-h")
        .scan<'u', size_t>()
        .default_value<size_t>(1600)
        .help("Output resolution height in pixels");
    argp.add_argument("--iterations", "-i")
        .scan<'u', size_t>()
        .default_value<size_t>(1000)
        .help("Number of iterations to run");
    argp.add_argument("--skip-first")
        .default_value(false)
        .implicit_value(true)
        .help("Skip the first iteration (removes lazy loading time)");
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
    size_t width = argp.get<size_t>("width");
    size_t height = argp.get<size_t>("height");
    size_t iter = argp.get<size_t>("iterations");
    bool skip_first = argp.get<bool>("skip-first");

    std::cout << "OpenCL Support: " << std::boolalpha << cv::ocl::haveOpenCL() << std::endl;
    if (cv::ocl::haveOpenCL())
    {
        const auto device = cv::ocl::Device::getDefault();
        std::cout << "Device: " << device.name() << '\n'
                  << "  Available: " << device.available() << '\n'
                  << "  Compiler:  " << device.compilerAvailable() << '\n'
                  << "  Linker:    " << device.linkerAvailable() << '\n'
                  << std::endl;
    }

    auto stitcher = std::make_unique<panoweave::Stitcher>(calib_path);
    stitcher->resolution(width, height);
    stitcher->setDepth(1);

    std::vector<cv::Mat> imgs;
    if (!readInputs(input_paths, imgs))
    {
        std::cerr << "Failed to read input images." << std::endl;
        return 1;
    }

    for (uint8_t i = 0; i < imgs.size(); ++i)
    {
        cv::namedWindow("Input " + std::to_string(i), cv::WINDOW_NORMAL);
        cv::imshow("Input " + std::to_string(i), imgs[i]);
    }
    cv::namedWindow("Output", cv::WINDOW_NORMAL);

    cv::Mat pano;
    if (skip_first)
    {
        stitcher->stitch(imgs, pano);
    }

    std::cout << "Resolution: " << width << 'x' << height << '\n'
              << "Iterations: " << iter << std::endl;
    std::list<std::chrono::nanoseconds> durations;
    std::chrono::nanoseconds duration = std::chrono::nanoseconds(0);
    for (uint64_t i = 0; i < iter; ++i)
    {
        const auto start = std::chrono::steady_clock::now();
        stitcher->stitch(imgs, pano);
        const auto end = std::chrono::steady_clock::now();
        const auto d = end - start;
        duration += d;
        durations.push_back(d);

        cv::imshow("Output", pano);
        if (cv::waitKey(1) == 27)
            break;
    }

    const auto avg_duration = std::chrono::duration_cast<std::chrono::duration<double, std::nano>>(duration) / durations.size();
    double variance = 0.0;
    for (const auto &duration : durations)
    {
        const auto diff = std::chrono::duration_cast<std::chrono::duration<double>>(duration - avg_duration).count();
        variance += diff * diff;
    }
    variance /= durations.size();
    const double mean = std::chrono::duration_cast<std::chrono::duration<double>>(avg_duration).count();

    std::cout << "Benchmark:\n"
                 "  Samples: " << durations.size() << "\n"
                 "  Mean:    " << mean << "s\t" << 1. / mean << "Hz\n"
                 "  StdDev:  " << std::sqrt(variance) << "s" << std::endl;
}
