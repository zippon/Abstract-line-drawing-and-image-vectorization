//
//  thing.h
//  AbstractLine
//
//  Created by Zhipeng Wu on 5/28/13.
//  Copyright (c) 2013 Zhipeng Wu. All rights reserved.
//
// http://opencv-code.com/quick-tips/implementation-of-thinning-algorithm-in-opencv/
// http://opencv-code.com/quick-tips/implementation-of-guo-hall-thinning-algorithm/


#ifndef AbstractLine_thing_h
#define AbstractLine_thing_h

#include <opencv2/opencv.hpp>

// Pixels:
// 1: foregrounf
// 0: background
static cv::Mat ZhangThinningIteration (const cv::Mat& im, int iter) {
  assert(CV_8UC1 == im.type());
  cv::Mat marker = cv::Mat::zeros(im.size(), CV_8UC1);
  for (int i = 1; i < im.rows-1; i++) {
    for (int j = 1; j < im.cols-1; j++) {
      uchar p2 = im.at<uchar>(i-1, j);
      uchar p3 = im.at<uchar>(i-1, j+1);
      uchar p4 = im.at<uchar>(i, j+1);
      uchar p5 = im.at<uchar>(i+1, j+1);
      uchar p6 = im.at<uchar>(i+1, j);
      uchar p7 = im.at<uchar>(i+1, j-1);
      uchar p8 = im.at<uchar>(i, j-1);
      uchar p9 = im.at<uchar>(i-1, j-1);
      
      int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
      (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
      (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
      (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
      int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
      int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
      int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);
      
      if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
        marker.at<uchar>(i,j) = 1;
    }
  }
  return im & ~marker;
}

static cv::Mat ZhangThinning(const cv::Mat& im) {
  cv::Mat prev = cv::Mat::zeros(im.size(), CV_8UC1);
  if (im.channels() != 1)
    return prev;
  cv::Mat img, temp, diff;
  im.convertTo(img, CV_8UC1);
  img /= 255;
  do {
    temp = ZhangThinningIteration(img, 0);
    img = ZhangThinningIteration(temp, 1);
    cv::absdiff(img, prev, diff);
    img.copyTo(prev);
  } while (cv::countNonZero(diff) > 0);
  return img * 255;
}


#endif
