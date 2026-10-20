#include "poseguard/geometry.hpp"
#include <fstream>
#include <iostream>
#include <filesystem>

int main(int argc, char** argv) {
    try {
        if (argc < 4) throw std::runtime_error("geometry_check preprocess|decode input output [parameters]");
        std::ifstream stream(std::filesystem::u8path(argv[2]), std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
        std::vector<float> result;
        if (std::string(argv[1]) == "preprocess") {
            auto image = cv::imdecode(bytes, cv::IMREAD_COLOR);
            if (image.empty()) throw std::runtime_error("Unreadable image");
            poseguard::preprocess(image, std::stoi(argv[4]), result);
        } else {
            if (argc != 9 || bytes.size()%sizeof(float)) throw std::runtime_error("Invalid decode arguments");
            std::vector<float> raw(bytes.size()/sizeof(float));
            std::memcpy(raw.data(),bytes.data(),bytes.size());
            poseguard::Letterbox meta{std::stof(argv[4]),std::stoi(argv[5]),std::stoi(argv[6]),std::stoi(argv[7]),std::stoi(argv[8]),640};
            for (const auto& p: poseguard::decode(raw,meta)) {
                result.insert(result.end(),{p.box.x,p.box.y,p.box.x+p.box.width,p.box.y+p.box.height,p.score});
                for (const auto& k: p.points) result.insert(result.end(),{k.x,k.y,k.z});
            }
        }
        std::ofstream destination(std::filesystem::u8path(argv[3]),std::ios::binary);
        destination.write(reinterpret_cast<const char*>(result.data()),result.size()*sizeof(float));
        if (!destination) throw std::runtime_error("Cannot write output");
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
