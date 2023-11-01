#pragma once
#include <ncnn/net.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <stdio.h>
#include <string>
#include <float.h>
#include <vector>
#include <fstream>

#include "common.hpp"

#define MAX_STRIDE 64

class Yolo {
public:
    Yolo();
    ~Yolo();
    int load(const std::string& bin, const std::string& param);
    int detect(const cv::Mat& bgr, std::vector<Object>& objects);
    int detect_dynamic(const cv::Mat& bgr, std::vector<Object>& objects);
    void draw_objects(cv::Mat& bgr, const std::vector<Object>& objects);
    void video(cv::VideoCapture capture);
    void image(cv::Mat in, std::string outputPath);
    void get_class_names(std::string data);
    void get_blob_name(std::string in, std::string out, std::string out0, std::string out1, std::string out2);

    bool dynamic = false;
    bool save = false;
    int target_size = 640;
    float prob_threshold = 0.25;
    float nms_threshold = 0.45;

private:
    ncnn::Net net;
    std::vector<std::string> class_names;
    int class_count = 80;
    std::string in_blob;
    std::string out_blob;
    std::string out0_blob;
    std::string out1_blob;
    std::string out2_blob;
};