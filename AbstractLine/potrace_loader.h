//
//  potrace_loader.h
//  potrace_test
//
//  Created by Zhipeng Wu on 5/29/13.
//  Copyright (c) 2013 Zhipeng Wu. All rights reserved.
//

#ifndef potrace_test_potrace_loader_h
#define potrace_test_potrace_loader_h

#include <iostream>
#include <string>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <potracelib.h>
#include <vector>

using std::string;
using std::cout;
using std::endl;
using std::system;
using std::vector;

const int kPrecision = 100;
int counter = 0;
cv::Scalar pen = CV_RGB(0, 0, 0);

/* macros for writing individual bitmap pixels */
#define BM_WORDSIZE ((int)sizeof(potrace_word))
#define BM_WORDBITS (8*BM_WORDSIZE)
#define BM_HIBIT (((potrace_word)1)<<(BM_WORDBITS-1))
#define bm_scanline(bm, y) ((bm)->map + (y)*(bm)->dy)
#define bm_index(bm, x, y) (&bm_scanline(bm, y)[(x)/BM_WORDBITS])
#define bm_mask(x) (BM_HIBIT >> ((x) & (BM_WORDBITS-1)))
#define bm_range(x, a) ((int)(x) >= 0 && (int)(x) < (a))
#define bm_safe(bm, x, y) (bm_range(x, (bm)->w) && bm_range(y, (bm)->h))
#define BM_USET(bm, x, y) (*bm_index(bm, x, y) |= bm_mask(x))
#define BM_UCLR(bm, x, y) (*bm_index(bm, x, y) &= ~bm_mask(x))
#define BM_UPUT(bm, x, y, b) ((b) ? BM_USET(bm, x, y) : BM_UCLR(bm, x, y))
#define BM_PUT(bm, x, y, b) (bm_safe(bm, x, y) ? BM_UPUT(bm, x, y, b) : 0)

/* return new un-initialized bitmap. NULL with errno on error */
static potrace_bitmap_t *bm_new(int w, int h) {
  potrace_bitmap_t *bm;
  int dy = (w + BM_WORDBITS - 1) / BM_WORDBITS;
  
  bm = (potrace_bitmap_t *) malloc(sizeof(potrace_bitmap_t));
  if (!bm) {
    return NULL;
  }
  bm->w = w;
  bm->h = h;
  bm->dy = dy;
  bm->map = (potrace_word *) malloc(dy * h * BM_WORDSIZE);
  if (!bm->map) {
    free(bm);
    return NULL;
  }
  return bm;
}

/* free a bitmap */
static void bm_free(potrace_bitmap_t *bm) {
  if (bm != NULL) {
    free(bm->map);
  }
  free(bm);
}

// Given an input image, return the tracing result in potrace_state_t.
static potrace_state_t* Raster2Vector(const cv::Mat& input_img) {
  cv::Mat mat_img;
  if (input_img.channels() != 1) {
    cv::cvtColor(input_img, mat_img, CV_BGR2GRAY);
  } else {
    input_img.copyTo(mat_img);
  }
  mat_img.convertTo(mat_img, CV_8UC1);
  
  // Step 1: load image content to potrace_bitmap_t.
  potrace_bitmap_t* bitmap_img;
  bitmap_img = bm_new(mat_img.cols, mat_img.cols);
  for(int j = 0; j < bitmap_img->dy * bitmap_img->h; ++j)
    bitmap_img->map[j] = 0;
  // set the pixel values. 
  for(int y = mat_img.rows - 1; y >= 0; --y){
  // for(int j = 0 - 1; j < mat_img.rows; ++j){
    for(int x = 0; x < mat_img.cols; ++x){
      // We treat white as background and black as foreground.
      if (mat_img.at<uchar>(y, x) < 128) {
        BM_PUT(bitmap_img, x, y, 1);
      }
    }
  }
  
  // Step 2: calculate potrace_state_t and return.
  static potrace_param_t *parameters = potrace_param_default();
  parameters->opticurve = 1;
  parameters->opttolerance = 0.2;
  parameters->turnpolicy = POTRACE_TURNPOLICY_MINORITY;
  parameters->turdsize = 100;
  parameters->alphamax = 1;
  //freed when program terminates (potrace_state_free)
  potrace_state_t* states = new potrace_state_t();
  states = potrace_trace(parameters, bitmap_img);
  if (!states || states->status != POTRACE_STATUS_OK) {
    cout << "Error in tracing..." << endl;
  }
  
  bm_free(bitmap_img);
  potrace_param_free(parameters);
  return states;
}


// Given an image path, return the tracing result in potrace_state_t.
static inline potrace_state_t* Raster2Vector(const string& file_path) {
  cv::Mat input_img = cv::imread(file_path, 0);
  if (input_img.empty()) {
    cout << "Input image error...";
    return NULL;
  }
  return Raster2Vector(input_img);
}

static inline cv::Point PointAdd(const cv::Point& p, const cv::Point& q) {
  return cv::Point(p.x + q.x, p.y + q.y);
}
static inline cv::Point PointTimes(float c, const cv::Point& p) {
  return cv::Point(p.x * c, p.y * c);
}
static inline cv::Point Bernstein(float u,
                                  const cv::Point& p0,
                                  const cv::Point& p1,
                                  const cv::Point& p2,
                                  const cv::Point& p3) {
  cv::Point a, b, c, d, r;
  
  a = PointTimes(pow(u,3), p0);
  b = PointTimes(3 * pow(u,2) * (1-u), p1);
  c = PointTimes(3 * u * pow((1-u), 2), p2);
  d = PointTimes(pow((1-u), 3), p3);
  return PointAdd(PointAdd(a, b), PointAdd(c, d));
}

static void DrawBezier(cv::Mat* canvas,
                       const cv::Point a,
                       const cv::Point u,
                       const cv::Point w,
                       const cv::Point b,
                       const cv::Scalar& pen,
                       vector<cv::Point>* contour) {
  cv::Point pre, now;
  pre = a;
  for (int i = 0; i <= kPrecision; ++i) {
    contour->push_back(pre);
    float t = 1 - static_cast<float>(i) / kPrecision;
    now = Bernstein(t, a, u, w, b);
    cv::line(*canvas, pre, now, pen);
    pre = now;
  }
}
// Plot curve on the canvas.
static void PlotCurve(cv::Mat* canvas,
                      const potrace_curve_t& curve,
                      const cv::Scalar& pen,
                      vector<cv::Point>* contour) {
  if (NULL == canvas)
    return;
  cv::Point a, u, w, b, begin;
  begin = cv::Point(curve.c[curve.n - 1][2].x,
                    curve.c[curve.n - 1][2].y);
  for (int i = 0; i < curve.n; ++i) {
    // Satrt point.
    a = i ? cv::Point(curve.c[i - 1][2].x, curve.c[i - 1][2].y) : begin;
    // Control point 1.
    u = cv::Point(curve.c[i][0].x, curve.c[i][0].y);
    // Control point 2.
    w = cv::Point(curve.c[i][1].x, curve.c[i][1].y);
    // End point.
    b = cv::Point(curve.c[i][2].x, curve.c[i][2].y);
    cv::circle(*canvas, a, 2, cv::Scalar(0, 0, 255), -1);
    cv::circle(*canvas, w, 2, cv::Scalar(0, 255, 0), -1);
    cv::circle(*canvas, b, 2, cv::Scalar(0, 0, 255), -1);
    
    if (POTRACE_CURVETO == curve.tag[i]) {
      cv::circle(*canvas, u, 2, cv::Scalar(0, 255, 0), -1);
      DrawBezier(canvas, a, u, w, b, pen, contour);
      
    } else if (POTRACE_CORNER == curve.tag[i]) {
      contour->push_back(a);
      contour->push_back(w);
      cv::line(*canvas, a, w, pen, 1);
      cv::line(*canvas, w, b, pen, 1);
    }
    cv::imshow("canvas", *canvas);
    cv::waitKey(100);
  }
  contour->push_back(begin);
}

static inline void ShowPath(potrace_path_t* my_path, cv::Mat* canvas) {
  potrace_path_t *p, *q;
  for (p = my_path; p; p = p->sibling) {
    // Show details.
    cout << "path: " << counter++ << ",\tarea: "
    << p->area << ",\tsign: " << (char)p->sign
    << ",\tcurve: " << p->curve.n << endl;
    // Draw curve.
    vector<vector<cv::Point> > contours;
    vector<cv::Point> contour;
    PlotCurve(canvas, p->curve, pen, &contour);
    // Fill region.
    contours.push_back(contour);
    if (my_path->sign == '+') {
      cv::drawContours(*canvas, contours, -1, cv::Scalar(0, 0, 0), -1);
    } else {
      cv::drawContours(*canvas, contours, -1, cv::Scalar(255, 255, 255), -1);
    }
    for (q = p->childlist; q; q = q->sibling) {
      ShowPath(q->childlist, canvas);
    }
  }
}

#endif
