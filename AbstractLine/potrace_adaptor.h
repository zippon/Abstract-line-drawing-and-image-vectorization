//
//  potrace_adaptor.h
//  potrace_test
//
//  Created by Zhipeng Wu on 5/29/13.
//  Copyright (c) 2013 Zhipeng Wu. All rights reserved.
//

#ifndef potrace_test_potrace_adaptor_h
#define potrace_test_potrace_adaptor_h

#include <iostream>
#include <string>
#include <cstdlib>
#include <opencv2/opencv.hpp>

using std::string;
using std::cout;
using std::endl;
using std::system;

const string kTempFolder = "/var/folders/_m/bhc8k8f971dbd_qkl27bh3yw0000gn/T/";

static inline string GetTitle(const string& filename) {
  size_t found1 = filename.find_last_of("/");
  size_t found2 = filename.find_last_of(".");
  if ((string::npos == found1) || (string::npos == found1)) {
    cout << "File title parsing error..." << endl;
    return "";
  }
  return filename.substr(found1 + 1, found2 - found1 - 1);
}

// Given an input image path, return the vectorized version.
static inline cv::Mat ExecPotrace(const string& file_path) {
  string file_title = GetTitle(file_path);
  string bmp_path = kTempFolder + file_title + ".bmp";
  string pgm_path = kTempFolder + file_title + ".pgm";
  cv::Mat input_img = cv::imread(file_path, 0);
  if (input_img.empty()) {
    cout << "Input image error...";
    return cv::Mat();
  }
  cv::imwrite(bmp_path, input_img);
  string potrace_arg = "/usr/local/bin/potrace -g -z minority -t 100.000000 -a 1.300000 -O \
  0.200000 -u 10.000000 -k 0.500000 -o" + pgm_path + " " + bmp_path;
  system(potrace_arg.c_str());
  cv::Mat output_img = cv::imread(pgm_path, 0);
  return output_img;
}

// Given an input image, return the vectorized version.
static inline cv::Mat ExecPotrace(const cv::Mat& input_img) {
  string bmp_path = kTempFolder + "temp_potrace.bmp";
  string pgm_path = kTempFolder + "temp_potrace.pgm";
  cv::imwrite(bmp_path, input_img);
  string potrace_arg = "/usr/local/bin/potrace -g -z minority -t 100.000000 -a 1.300000 -O \
  0.200000 -u 10.000000 -k 0.500000 -o" + pgm_path + " " + bmp_path;
  system(potrace_arg.c_str());
  cv::Mat output_img = cv::imread(pgm_path, 0);
  return output_img;
}

// Given an input image, save the vectorized version (.svg).
static inline void ExecPotraceAndSaveSVG(const cv::Mat& input_img,
                                         const string& svg_path) {
  string bmp_path = kTempFolder + "temp_potrace.bmp";
  cv::imwrite(bmp_path, input_img);
  string potrace_arg = "/usr/local/bin/potrace -s -z minority -t 100.000000 -a 1.300000 -O \
  0.200000 -u 10.000000 -k 0.500000 -o" + svg_path + " " + bmp_path;
  system(potrace_arg.c_str());
}

// Given an input image path, save the vectorized version (.svg).
static inline void ExecPotraceAndSaveSVG(const string& file_path,
                                         const string& svg_path) {
  string file_title = GetTitle(file_path);
  string bmp_path = kTempFolder + file_title + ".bmp";
  cv::Mat input_img = cv::imread(file_path, 0);
  if (input_img.empty()) {
    cout << "Input image error...";
    return;
  }
  cv::imwrite(bmp_path, input_img);
  string potrace_arg = "/usr/local/bin/potrace -s -z minority -t 100.000000 -a 1.300000 -O \
  0.200000 -u 10.000000 -k 0.500000 -o" + svg_path + " " + bmp_path;
  system(potrace_arg.c_str());
}

#endif
