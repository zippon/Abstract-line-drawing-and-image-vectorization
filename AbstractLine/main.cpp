//
//  main.cpp
//  AbstractLine
//
//  Created by Zhipeng Wu on 5/22/13.
//  Copyright (c) 2013 Zhipeng Wu. All rights reserved.
//

#include <iostream>
#include "CoherentLine.h"
#include "potrace_adaptor.h"
#include "potrace_loader.h"

int main(int argc, const char * argv[])
{

  // insert code here...
  // string img_file = "/Users/WU/Pictures/others/lena.jpg";
  string img_file = "/Users/WU/Pictures/others/bundle_origin.jpg";
  CoherentLine cl(img_file);
  cv::Mat dog_edge = cl.dog_edge();
  cv::imshow("dog", dog_edge);
  cv::waitKey();
  cv::Mat fdog_edge = cl.fdog_edge();
  cv::imshow("fdog", fdog_edge);
  cv::waitKey();
  cv::Mat vec_edge = ExecPotrace(fdog_edge);
  cv::imshow("vectorization", vec_edge);
  cv::waitKey();
  cv::Mat canvas(fdog_edge.size(), CV_8UC3);
  canvas = cv::Scalar::all(255);
  potrace_state_t* my_state = Raster2Vector(fdog_edge);
  ShowPath(my_state->plist, &canvas);
  potrace_state_free(my_state);
  cv::waitKey();
  return 0;
}

