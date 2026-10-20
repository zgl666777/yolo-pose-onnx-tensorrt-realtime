#include "poseguard/engine.hpp"
#include "poseguard/geometry.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <filesystem>

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "Usage: pose_camera engine [camera|video] [imgsz=640] [--headless]\n"; return 2; }
    try {
        int size = 640;
        bool headless = false;
        for (int i=3; i<argc; ++i) {
            if (std::string(argv[i]) == "--headless") headless = true;
            else size = std::stoi(argv[i]);
        }
        poseguard::Engine engine(argv[1], size);
        cv::VideoCapture cap;
        const std::string source = argc > 2 ? argv[2] : "0";
        if (std::all_of(source.begin(), source.end(), [](unsigned char c) { return std::isdigit(c); }))
            cap.open(std::stoi(source));
        else cap.open(source);
        if (!cap.isOpened()) throw std::runtime_error("Cannot open camera/video");
        std::vector<float> input(static_cast<std::size_t>(3) * size * size), output(engine.output_size());
        std::size_t frames = 0;
        double gpu_total_ms = 0.0;
        cv::Mat frame;
        bool user_stopped = false;
        while (cap.read(frame)) {
            const auto start = std::chrono::steady_clock::now();
            const auto meta = poseguard::preprocess(frame, size, input);
            auto times = engine.infer(input.data(), input.size(), output.data(), output.size());
            ++frames;
            gpu_total_ms += times[1];
            const auto keep = poseguard::decode(output, meta);
            constexpr int edges[][2] = {{0,1},{0,2},{1,3},{2,4},{5,6},{5,7},{7,9},{6,8},{8,10},
                                        {5,11},{6,12},{11,12},{11,13},{13,15},{12,14},{14,16}};
            for (const auto& pose : keep) {
                cv::rectangle(frame, pose.box, {90, 220, 110}, 2);
                for (const auto& edge : edges) {
                    const auto a = pose.points[edge[0]], b = pose.points[edge[1]];
                    if (std::min(a.z, b.z) >= 0.3f)
                        cv::line(frame, cv::Point(cvRound(a.x), cvRound(a.y)), cv::Point(cvRound(b.x), cvRound(b.y)),
                                 {240, 200, 80}, 2, cv::LINE_AA);
                }
                for (const auto& point : pose.points) if (point.z >= 0.3f)
                    cv::circle(frame, {cvRound(point.x), cvRound(point.y)}, 3, {230, 230, 230}, -1);
            }
            const float elapsed = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();
            cv::putText(frame, "GPU " + std::to_string(times[1]) + " ms / frame " + std::to_string(elapsed) + " ms",
                         {20, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.55, {240, 240, 240}, 1);
            if (!headless) {
                cv::imshow("PoseGuard C++ Pose", frame);
                const auto key = cv::waitKey(1) & 0xff;
                if (key == 27 || key == 'q') { user_stopped = true; break; }
            }
        }
        if (!headless) cv::destroyAllWindows();
        std::cout << "frames=" << frames << " mean_gpu_ms=" << gpu_total_ms / std::max<std::size_t>(frames, 1) << '\n';
        if (frames == 0) throw std::runtime_error("Source produced no frames");
        if (!user_stopped && !std::filesystem::is_regular_file(std::filesystem::u8path(source)))
            throw std::runtime_error("Live source stopped; use poseguard run --backend cpp for supervised capture recovery");
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
