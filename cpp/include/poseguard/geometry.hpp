#pragma once
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace poseguard {
struct Pose { cv::Rect2f box; float score; std::array<cv::Point3f, 17> points; };
struct Letterbox { float scale; int left, top, width, height, size; };

inline Letterbox preprocess(const cv::Mat& frame, int size, std::vector<float>& input) {
    const double scale = std::min(static_cast<double>(size) / frame.cols, static_cast<double>(size) / frame.rows);
    const int w = static_cast<int>(std::nearbyint(frame.cols * scale));
    const int h = static_cast<int>(std::nearbyint(frame.rows * scale));
    const int left = static_cast<int>(std::round((size - w) / 2.0 - .1));
    const int top = static_cast<int>(std::round((size - h) / 2.0 - .1));
    cv::Mat resized, padded;
    cv::resize(frame, resized, {w, h});
    cv::copyMakeBorder(resized, padded, top, size - h - top, left, size - w - left,
                       cv::BORDER_CONSTANT, {114, 114, 114});
    input.resize(static_cast<std::size_t>(3) * size * size);
    for (int y = 0; y < size; ++y) for (int x = 0; x < size; ++x) {
        const auto pixel = padded.at<cv::Vec3b>(y, x);
        for (int c = 0; c < 3; ++c)
            input[static_cast<std::size_t>(c) * size * size + y * size + x] = pixel[2 - c] / 255.0f;
    }
    return {static_cast<float>(scale), left, top, frame.cols, frame.rows, size};
}

inline float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float inter = std::max(0.0f, std::min(a.x+a.width, b.x+b.width)-std::max(a.x,b.x)) *
                        std::max(0.0f, std::min(a.y+a.height,b.y+b.height)-std::max(a.y,b.y));
    return inter / std::max(a.area() + b.area() - inter, 1e-9f);
}

inline std::vector<Pose> decode(const std::vector<float>& output, const Letterbox& meta,
                                float confidence=.35f, float threshold=.45f) {
    if (output.size() % 56) throw std::runtime_error("Expected 56-channel pose output");
    const std::size_t n = output.size() / 56;
    std::vector<Pose> poses;
    for (std::size_t i=0; i<n; ++i) {
        bool finite = true;
        for (int c=0; c<56; ++c) finite = finite && std::isfinite(output[c*n+i]);
        if (!finite || output[4*n+i] < confidence || output[2*n+i] <= 0 || output[3*n+i] <= 0) continue;
        Pose p;
        p.score = output[4*n+i];
        const float x1=output[i]-output[2*n+i]/2, y1=output[n+i]-output[3*n+i]/2;
        const float x2=output[i]+output[2*n+i]/2, y2=output[n+i]+output[3*n+i]/2;
        p.box = {x1,y1,x2-x1,y2-y1};
        for (int k=0; k<17; ++k) p.points[k] = {output[(5+3*k)*n+i],output[(6+3*k)*n+i],output[(7+3*k)*n+i]};
        poses.push_back(p);
    }
    std::stable_sort(poses.begin(),poses.end(),[](const Pose& a,const Pose& b){return a.score>b.score;});
    if (poses.size()>3000) poses.resize(3000);
    std::vector<Pose> selected;
    for (const auto& pose: poses) {
        if (selected.size()==100) break;
        if (std::none_of(selected.begin(),selected.end(),[&](const Pose& prev){return iou(prev.box,pose.box)>threshold;}))
            selected.push_back(pose);
    }
    // Restore and clip only after model-space NMS, including degenerate border boxes.
    std::vector<Pose> result;
    for (auto pose: selected) {
        auto x = [&](float v){return std::clamp((v-meta.left)/meta.scale,0.0f,static_cast<float>(meta.width));};
        auto y = [&](float v){return std::clamp((v-meta.top)/meta.scale,0.0f,static_cast<float>(meta.height));};
        const float x1=x(pose.box.x), y1=y(pose.box.y), x2=x(pose.box.x+pose.box.width), y2=y(pose.box.y+pose.box.height);
        pose.box = {x1,y1,x2-x1,y2-y1};
        for (auto& point: pose.points) point={x(point.x),y(point.y),std::clamp(point.z,0.0f,1.0f)};
        if (x2>x1 && y2>y1) result.push_back(pose);
    }
    return result;
}
}
