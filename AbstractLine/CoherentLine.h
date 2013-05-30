//
//  CoherentLine.h
//  AbstractLine
//
//  Created by Zhipeng Wu on 5/22/13.
//  Copyright (c) 2013 Zhipeng Wu. All rights reserved.
//
//  Kyprianidis, J. E., & Döllner, J. (2008).
//  Image Abstraction by Structure Adaptive Filtering.
//  In: Proc. EG UK Theory and Practice of Computer Graphics, pp. 51–58.
//
//  H. Kang, S. Lee, C. Chui.
//  "Coherent Line Drawing".
//  Proc. ACM Symposium on Non-photorealistic Animation and Rendering,
//  pp. 43-50, San Diego, CA, August, 2007.

#ifndef __AbstractLine__CoherentLine__
#define __AbstractLine__CoherentLine__

#include "thing.h"
#include <time.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <functional>
#include <numeric>
#include <sstream>
#include <map>
#include <numeric>
#include <iterator>
#include <queue>
#include <opencv2/opencv.hpp>

using std::map;
using std::pair;
using std::string;
using std::vector;
using std::cout;
using std::endl;

class CoherentLine {
public:
  CoherentLine(const string& img_path) {
    srand (static_cast<unsigned int>(time(NULL)));
    image_ = cv::imread(img_path, 1);
    cv::cvtColor(image_, gray_, CV_BGR2GRAY);
    rows_ = gray_.rows;
    cols_ = gray_.cols;
    cv::bilateralFilter(image_, bimage_, 6, 150, 150);
    cv::cvtColor(bimage_, bgray_, CV_BGR2GRAY);
    cout << "CoherentLine object constructed." << endl;
  }
  CoherentLine(const cv::Mat& image) {
    srand (static_cast<unsigned int>(time(NULL)));
    image_ = image.clone();
    cv::cvtColor(image_, gray_, CV_BGR2GRAY);
    rows_ = gray_.rows;
    cols_ = gray_.cols;
    cv::bilateralFilter(image_, bimage_, 6, 150, 150);
    cv::cvtColor(bimage_, bgray_, CV_BGR2GRAY);
    cout << "CoherentLine object constructed." << endl;
  }
  // Accessors:
  const int rows() const {
    return rows_;
  }
  const int cols() const {
    return cols_;
  }
  const cv::Mat& image() const {
    return image_;
  }
  const cv::Mat& gray() const {
    return gray_;
  }
  const cv::Mat& etf() {
    if (etf_.empty())
      GetEdegTangentFlow();
    return etf_;
  }
  const cv::Mat& dog_edge() {
    if (dog_edge_.empty())
      GetDogEdge();
    return dog_edge_;
  }
  const cv::Mat& fdog_edge() {
    if (fdog_edge_.empty())
      GetFDogEdge();
    return fdog_edge_;
  }
  const cv::Mat& canny_edge() {
    if (canny_edge_.empty())
      GetCannyEdge();
    return canny_edge_;
  }
  
private:
  
  // Members:
  int cols_;
  int rows_;
  cv::Mat image_;         // input image.
  cv::Mat gray_;          // gray-scale image.
  cv::Mat bimage_;        // bilateral blured input image.
  cv::Mat bgray_;         // bilateral blured gray-scale image.
  cv::Mat etf_;           // edge tangent flow.
  cv::Mat dog_edge_;      // edge response by using DoG operation.
  cv::Mat fdog_edge_;     // blur dog edge with edge tangent flow.
  cv::Mat canny_edge_;    // canny edge detection.
  cv::Mat thin_edge_;     // edge thinning result.
  
  // Functions:
  void GetEdegTangentFlow();
  void GetDogEdge();
  void GetStepEdge();
  void GetFDogEdge();
  void CalcStructureTensor(cv::Mat* st);
  void VisualizeByLIC(const cv::Mat& vf);
  void VisualizeByArrow(const cv::Mat& vf);
  void GetCannyEdge();


  
  // disallow evil copy and assign.
  CoherentLine(const CoherentLine&);
  void operator= (const CoherentLine&);
};



#endif /* defined(__AbstractLine__CoherentLine__) */
