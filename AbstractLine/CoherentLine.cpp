//
//  CoherentLine.cpp
//  AbstractLine
//
//  Created by Zhipeng Wu on 5/22/13.
//  Copyright (c) 2013 Zhipeng Wu. All rights reserved.
//

#include "CoherentLine.h"

#define	 DISCRETE_FILTER_SIZE	2048
#define  LOWPASS_FILTR_LENGTH	10.00000f
#define	 LINE_SQUARE_CLIP_MAX	100000.0f
#define	 VECTOR_COMPONENT_MIN   0.050000f
#define PI 3.1415926

cv::RNG rng(12345);
// Prepare 1-d gaussian template.
static void GetGaussianWeights(float* weights,
                               int neighbor,
                               float sigma) {
  if ((NULL == weights) || (neighbor < 0))
    return;
  float term1 = 1.0 / (sqrt(2.0 * PI) * sigma);
  float term2 = -1.0 / (2 * pow(sigma, 2));
  weights[neighbor] = term1;
  float sum = weights[neighbor];
  for (int i = 1; i <= neighbor; ++i) {
    weights[neighbor + i] = exp(pow(i, 2) * term2) * term1;
    weights[neighbor - i] =  weights[neighbor + i];
    sum += weights[neighbor + i] + weights[neighbor - i];
  }
  // Normalization
  for (int j = 0; j < neighbor * 2 + 1; ++j) {
    weights[j] /= sum;
  }
}

// Prepare 1-d difference of gaussian template.
static void GetDiffGaussianWeights(float* weights,
                                   int neighbor,
                                   float sigma_e,
                                   float sigma_r,
                                   float tau) {
  if ((NULL == weights) || (neighbor < 0))
    return;
  float* gaussian_e = new float[neighbor * 2 + 1];
  float* gaussian_r = new float[neighbor * 2 + 1];
  GetGaussianWeights(gaussian_e, neighbor, sigma_e);
  GetGaussianWeights(gaussian_r, neighbor, sigma_r);
  float sum = 0;
  for (int i = 0; i < neighbor * 2 + 1; ++i) {
    weights[i] = gaussian_e[i] - tau * gaussian_r[i];
    sum += weights[i];
  }
  // Normalization
  for (int j = 0; j < neighbor * 2 + 1; ++j) {
    weights[j] /= sum;
  }
  delete[] gaussian_e;
  delete[] gaussian_r;
}

void CoherentLine::GetEdegTangentFlow() {
  // Step 1: Cclculate the structure tensor.
  cv::Mat st;  // CV_32FC3 (E, G, F)
  CalcStructureTensor(&st);
  // Step 2: Gaussian blur the struct tensor. sst_sigma = 2.0
  float sigma_sst = 2;
  int gaussian_size = ceil(sigma_sst * 2) * 2 + 1;
  cv::GaussianBlur(st, st, cv::Size2i(gaussian_size, gaussian_size), sigma_sst);
  // Step 3: Extract etf_: CV_32FC3 (v2.x, v2.y, sqrt(lambda2).
  etf_ = cv::Mat::zeros(rows_, cols_, CV_32FC3);
  float E, G, F ,lambda1, v2x, v2y, v2;
  for (int r = 0; r < rows_; ++r) {
    for (int c = 0; c < cols_; ++c) {
      E = st.at<cv::Vec3f>(r, c)[0];
      G = st.at<cv::Vec3f>(r, c)[1];
      F = st.at<cv::Vec3f>(r, c)[2];
      lambda1 = 0.5 * (E + G + sqrtf((G - E) * (G - E) + 4 * F * F));
      v2x = E - lambda1;
      v2y = F;
      v2 = sqrtf(v2x * v2x + v2y * v2y);
      etf_.at<cv::Vec3f>(r, c)[0] = (0 == v2)? 0 : (v2x / v2);
      etf_.at<cv::Vec3f>(r, c)[1] = (0 == v2)? 0 : (v2y / v2);
      assert(E + G - lambda1 >= 0);
      etf_.at<cv::Vec3f>(r, c)[2] = sqrtf(E + G - lambda1);
    }
  }
  // What to show it?
  // VisualizeByLIC(etf_);
  // VisualizeByArrow(etf_);
}

void CoherentLine::GetDogEdge() {
  dog_edge_ = cv::Mat::zeros(rows_, cols_, CV_8UC1);
  float sigma_e = 1.0;
  float sigma_r = 1.6;
  float tau = 0.99;
  float phi = 2.0;
  int gaussian_size = ceilf(2.0 * sigma_r) * 2 + 1;
  cv::Mat blur_e, blur_r, gray;
  cv::bilateralFilter(gray_, gray, 5, 150, 150);
  gray.convertTo(gray, CV_32FC1);
  cv::GaussianBlur(gray, blur_e, cv::Size2i(gaussian_size, gaussian_size), sigma_e);
  cv::GaussianBlur(gray, blur_r, cv::Size2i(gaussian_size, gaussian_size), sigma_r);
  float diff;
  for (int r = 0; r < rows_; ++r) {
    for (int c = 0; c < cols_; ++c) {
      diff = blur_e.at<float>(r, c) - tau * blur_r.at<float>(r, c);
      if (diff > 0) {
        dog_edge_.at<uchar>(r, c) = 255;
      } else {
        dog_edge_.at<uchar>(r, c) = static_cast<uchar>(255 * (1 + tanhf(diff * phi)));
      }
    }
  }
//  cv::imshow("dog_edge", dog_edge());
//  cv::waitKey();
}

void CoherentLine::GetFDogEdge() {
  if (etf_.empty()) {
    GetEdegTangentFlow();
    cout << "EFT calculation finished." << endl;
  }
  fdog_edge_.create(rows_, cols_, CV_32FC1);
  fdog_edge_ = 255;
  cv::Mat f0 = cv::Mat::ones(rows_, cols_, CV_32FC1);
  cv::Mat f1 = cv::Mat::ones(rows_, cols_, CV_32FC1);
//  cv::Mat u0(rows_, cols_, CV_8UC1);
  cv::Mat u1 = cv::Mat::zeros(image_.size(), CV_8UC1);
  float sigma_e = 1.0;
  float sigma_r = 1.6;
  float sigma_m = 3.0;
  float tau = 0.99;
  // float phi = 2.0;
  int neighbor1 = static_cast<int>(ceilf(2.0 * sigma_r));
  float sin_theta, cos_theta;
  float* diff_gaussian_weights = new float[neighbor1 * 2 + 1];
  float* sample_pixels1, *sample_pixels2;
  float sum_diff, sum_dev, sum_1;
  GetDiffGaussianWeights(diff_gaussian_weights, neighbor1, sigma_e, sigma_r, tau);
  int neighbor2 = ceilf(2.0 * sigma_m);
  float* gaussian_weights = new float[neighbor2 * 2 + 1];
  GetGaussianWeights(gaussian_weights, neighbor2, sigma_m);
  cv::Mat gray;
  bgray_.copyTo(gray);

  // Step 1: do DoG along the gradient direction.
  for (int r = neighbor1; r < (rows_ - neighbor1); ++r) {
    for (int c = neighbor1; c < (cols_ - neighbor1); ++c) {
      // Get pixel gradient direction.
      cos_theta = etf_.at<cv::Vec3f>(r, c)[1];
      sin_theta = -1 * etf_.at<cv::Vec3f>(r, c)[0];
      sample_pixels1 = new float[neighbor1 * 2 + 1];
      sample_pixels1[neighbor1] = static_cast<float>(gray.at<uchar>(r, c));
      for (int k = 1; k <= neighbor1; ++k) {
        int r_offset = round(sin_theta * k);
        int c_offset = round(cos_theta * k);
        sample_pixels1[neighbor1 + k] =
        static_cast<float>(gray.at<uchar>(r + r_offset, c + c_offset));
        sample_pixels1[neighbor1 - k] =
        static_cast<float>(gray.at<uchar>(r - r_offset, c - c_offset));
      }
      // Calculate edge response.
      sum_diff = 0;
      sum_dev = 0;
      for (int k = 0; k < 2 * neighbor1 + 1; ++k) {
        sum_diff += sample_pixels1[k] * diff_gaussian_weights[k];
      }
      f0.at<float>(r, c) = sum_diff;
      delete[] sample_pixels1;
    }
  }
  cv::imshow("dog along tangent", f0);
  cv::waitKey();
  
  // Step 2: do Gaussian blur along tangent direction.
  for (int r = neighbor2; r < (rows_ - neighbor2); ++r) {
    for (int c = neighbor2; c < (cols_ - neighbor2); ++c) {
      // Get pixel tangent direction.
      cos_theta = etf_.at<cv::Vec3f>(r, c)[0];
      sin_theta = etf_.at<cv::Vec3f>(r, c)[1];
      sample_pixels2 = new float[neighbor2 * 2 + 1];
      sample_pixels2[neighbor2] = f0.at<float>(r, c);
      for (int k = 1; k <= neighbor2; ++k) {
        int r_offset = round(sin_theta * k);
        int c_offset = round(cos_theta * k);
        sample_pixels2[neighbor2 + k] = f0.at<float>(r + r_offset, c + c_offset);
        sample_pixels2[neighbor2 - k] = f0.at<float>(r - r_offset, c - c_offset);
      }
      // Calculate edge response.
      sum_1 = 0;
      for (int k = 0; k < 2 * neighbor2 + 1; ++k) {
        sum_1 += sample_pixels2[k] * gaussian_weights[k];
      }
      f1.at<float>(r, c) = sum_1;
      if (f1.at<float>(r, c) > 0) {
        u1.at<uchar>(r,c) = 0;
        fdog_edge_.at<float>(r, c) = 255;
      } else {
        u1.at<uchar>(r,c) = 255;
        fdog_edge_.at<float>(r, c) = 0;
      }
      delete[] sample_pixels2;
    }
  }
//  cv::imshow("fdog", fdog_edge_);
//  cv::waitKey();
//  cv::imshow("edge", u1);
//  cv::waitKey();
//  
  delete [] diff_gaussian_weights;
  delete [] gaussian_weights;
  // cv::imwrite("/Users/WU/Pictures/test_image/line/fdog1.bmp", fdog_edge_);


//  thin_edge_ = ZhangThinning(u1);
//  cv::imshow("thin", thin_edge_);
//  cv::waitKey();
  
//  vector<vector<cv::Point> > contours;
//  vector<cv::Vec4i> hierarchy;
//  cv::findContours(u1, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
//  cv::Mat drawing = cv::Mat::zeros(image_.size(), CV_8UC3);
//  for( int i = 0; i< contours.size(); i++ ) {
//    if (contours[i].size() < 20)
//      continue;
//    cout << contours[i].size() << endl;
//    cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255));
//    cv::drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, cv::Point());
//  }
  
//  /// Show in a window
//  cv::namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
//  imshow( "Contours", drawing);
//  cv::waitKey();
}


void CoherentLine::CalcStructureTensor(cv::Mat* st) {
  st->create(rows_, cols_, CV_32FC3);
  if (1 == image_.channels()) {
    // Gradient calculation.
    cv::Mat gx, gy;
    cv::Sobel(gray_, gx, CV_32F, 1, 0);
    cv::Sobel(gray_, gy, CV_32F, 0, 1);
    float dx, dy;
    for (int r = 0; r < rows_; ++r) {
      for (int c = 0; c < cols_; ++c) {
        dx = gx.at<float>(r, c);
        dy = gy.at<float>(r, c);
        st->at<cv::Vec3f>(r, c)[0] = dx * dx;  // E
        st->at<cv::Vec3f>(r, c)[1] = dy * dy;  // G
        st->at<cv::Vec3f>(r, c)[2] = dx * dy;  // F
      }
    }
    return;
  } else if ((3 == image_.channels()) || (4 == image_.channels())) {
    // BGR color space Gradient calculation.
    vector<cv::Mat> bgr_chnnels;
    cv::split(bimage_, bgr_chnnels);
    cv::Mat gx[3], gy[3];
    for (int k = 0; k < 3; ++k) {
      cv::Sobel(bgr_chnnels[k], gx[k], CV_32F, 1, 0);
      cv::Sobel(bgr_chnnels[k], gy[k], CV_32F, 0, 1);
    }
    cv::Vec3f fx, fy;
    for (int r = 0; r < rows_; ++r) {
      for (int c = 0; c < cols_; ++c) {
        fx = cv::Vec3f(gx[0].at<float>(r, c),
                       gx[1].at<float>(r, c),
                       gx[2].at<float>(r, c));
        fy = cv::Vec3f(gy[0].at<float>(r, c),
                       gy[1].at<float>(r, c),
                       gy[2].at<float>(r, c));
        st->at<cv::Vec3f>(r, c)[0] = fx.dot(fx);  // E
        st->at<cv::Vec3f>(r, c)[1] = fy.dot(fy);  // G
        st->at<cv::Vec3f>(r, c)[2] = fx.dot(fy);  // F
      }
    }
  } else {
    return;
  }
}

// Visualize a vector field by using LIC (Linear Integral Convolution).
void CoherentLine::VisualizeByLIC(const cv::Mat& vf) {
  assert(vf.channels() >= 2);
  vector<cv::Mat> vector_field;
  cv::split(vf, vector_field);
  cv::Mat white_noise(rows_, cols_, CV_8UC1);
  cv::Mat show_field(rows_, cols_, CV_8UC1);
  for (int r = 0; r < rows_; ++r) {
    for (int c = 0; c < cols_; ++c) {
      int n = rand();
      n = ((n & 0xff) + ((n & 0xff00) >> 8 )) & 0xff;
      white_noise.at<uchar>(r, c) = static_cast<uchar>(n);
    }
  }
  float p_LUT0[DISCRETE_FILTER_SIZE], p_LUT1[DISCRETE_FILTER_SIZE];
  for (int i = 0; i < DISCRETE_FILTER_SIZE; ++i) {
    p_LUT0[i] = p_LUT1[i] = i;
  }
  
  // Do LIC.
  float krnlen = LOWPASS_FILTR_LENGTH;
  int		advDir;						///ADVection DIRection (0: positive;  1: negative)
  int		advcts;						///number of ADVeCTion stepS per direction (a step counter)
  int		ADVCTS = int(krnlen * 3);	///MAXIMUM number of advection steps per direction to break dead loops
  
  float	vctr_x;						///x-component  of the VeCToR at the forefront point
  float	vctr_y;						///y-component  of the VeCToR at the forefront point
  float	clp0_x;						///x-coordinate of CLiP point 0 (current)
  float	clp0_y;						///y-coordinate of CLiP point 0	(current)
  float	clp1_x;						///x-coordinate of CLiP point 1 (next   )
  float	clp1_y;						///y-coordinate of CLiP point 1 (next   )
  float	samp_x;						///x-coordinate of the SAMPle in the current pixel
  float	samp_y;						///y-coordinate of the SAMPle in the current pixel
  float	tmpLen;						///TeMPorary LENgth of a trial clipped-segment
  float	segLen;						///SEGment   LENgth
  float	curLen;						///CURrent   LENgth of the streamline
  float	prvLen;						///PReVious  LENgth of the streamline
  float	W_ACUM;						///ACcuMulated Weight from the seed to the current streamline forefront
  float	texVal;						///TEXture VALue
  float	smpWgt;						///WeiGhT of the current SaMPle
  float	t_acum[2];					///two ACcUMulated composite Textures for the two directions, perspectively
  float	w_acum[2];					///two ACcUMulated Weighting values   for the two directions, perspectively
  float*	wgtLUT = NULL;				///WeiGhT Look Up Table pointing to the target filter LUT
  float	len2ID = (DISCRETE_FILTER_SIZE - 1) / krnlen;	///map a curve LENgth TO an ID in the LUT
  
  for (int r = 0; r < rows_; ++r) {
    for (int c = 0; c < cols_; ++c) {
      ///init the composite texture accumulators and the weight accumulators///
			t_acum[0] = t_acum[1] = w_acum[0] = w_acum[1] = 0.0f;
      ///for either advection direction///
      for(advDir = 0;  advDir < 2;  advDir ++) {
        advcts = 0;
				curLen = 0.0f;
        clp0_x = c + 0.5f;
				clp0_y = r + 0.5f;
        ///access the target filter LUT///
				wgtLUT = (advDir == 0) ? p_LUT0 : p_LUT1;
        ///until the streamline is advected long enough or a tightly  spiralling center / focus is encountered///
        while (curLen < krnlen && advcts < ADVCTS) {
          ///access the vector at the sample///
					vctr_x = vector_field[0].at<float>(r, c);
					vctr_y = vector_field[1].at<float>(r, c);
          ///in case of a critical point///
					if (vctr_x == 0.0f && vctr_y == 0.0f) {
						t_acum[advDir] = (advcts == 0) ? 0.0f : t_acum[advDir];		   ///this line is indeed unnecessary
						w_acum[advDir] = (advcts == 0) ? 1.0f : w_acum[advDir];
						break;
					}
          ///negate the vector for the backward-advection case///
					vctr_x = (advDir == 0) ? vctr_x : -vctr_x;
					vctr_y = (advDir == 0) ? vctr_y : -vctr_y;
          // clip the segment against the pixel boundaries
          // --- find the shorter from the two clipped segments.
					// replace  all  if-statements  whenever  possible  as
          // they  might  affect the computational speed.
					segLen = LINE_SQUARE_CLIP_MAX;
					segLen = (vctr_x < -VECTOR_COMPONENT_MIN) ?
              (int(clp0_x) - clp0_x ) / vctr_x : segLen;
					segLen = (vctr_x > VECTOR_COMPONENT_MIN) ?
              (int(int(clp0_x) + 1.5f) - clp0_x) / vctr_x : segLen;
					segLen = (vctr_y < -VECTOR_COMPONENT_MIN) ?
              (((tmpLen = (int(clp0_y) - clp0_y) / vctr_y) < segLen) ? tmpLen : segLen) : segLen;
					segLen = (vctr_y > VECTOR_COMPONENT_MIN) ?
              (((tmpLen = (int(int(clp0_y) + 1.5f) - clp0_y) / vctr_y)  <  segLen) ? tmpLen : segLen) : segLen;
          ///update the curve-length measurers///
					prvLen = curLen;
					curLen+= segLen;
					segLen+= 0.0004f;
          ///check if the filter has reached either end///
					segLen = (curLen > krnlen) ? ( (curLen = krnlen) - prvLen ) : segLen;
          ///obtain the next clip point///
					clp1_x = clp0_x + vctr_x * segLen;
					clp1_y = clp0_y + vctr_y * segLen;
          ///obtain the middle point of the segment as the texture-contributing sample///
					samp_x = (clp0_x + clp1_x) * 0.5f;
					samp_y = (clp0_y + clp1_y) * 0.5f;
          ///obtain the texture value of the sample///
					texVal = static_cast<float>(white_noise.at<uchar>(samp_y, samp_x));
          ///update the accumulated weight and the accumulated composite texture (texture x weight)///
					W_ACUM = wgtLUT[int(curLen * len2ID)];
					smpWgt = W_ACUM - w_acum[advDir];
					w_acum[advDir]  = W_ACUM;
					t_acum[advDir] += texVal * smpWgt;
          ///update the step counter and the "current" clip point///
					advcts ++;
					clp0_x = clp1_x;
					clp0_y = clp1_y;
          ///check if the streamline has gone beyond the flow field///
					if (clp0_x < 0.0f || clp0_x >= cols_ || clp0_y < 0.0f || clp0_y >= rows_)
            break;
        }
      }
      ///normalize the accumulated composite texture///
      texVal = (t_acum[0] + t_acum[1]) / (w_acum[0] + w_acum[1]);
      ///clamp the texture value against the displayable intensity range [0, 255]
			texVal = (texVal < 0.0f) ? 0.0f : texVal;
			texVal = (texVal > 255.0f) ? 255.0f : texVal;
      show_field.at<uchar>(r, c) = static_cast<uchar>(texVal);
    }
  }
  
  cv::imshow("Visualized Field", show_field);
  cv::waitKey();
}

void CoherentLine::VisualizeByArrow(const cv::Mat &vf) {
  cv::Mat show_filed = image_.clone();
  float angle, dx, dy, mag;
  cv::Point2d p, q;
  for (int r = 0; r < rows_; r += 20) {
    for (int c = 0; c < cols_; c += 20) {
      dx = vf.at<cv::Vec3f>(r, c)[0];
      dy = vf.at<cv::Vec3f>(r, c)[1];
      mag = vf.at<cv::Vec3f>(r, c)[2];
      if (mag > 0) {
        if (fabs(dx) < 0.0000001)
          angle = PI / 2;
        else
          angle = atan2f(dy, dx);
        p = cv::Point2d(c, r);
        q = cv::Point2d(c - (int)(0.5 * mag * cos(angle)),
                        r - (int)(0.5 * mag * sin(angle)));
        cv::line(show_filed, p, q, cv::Scalar(0, 0, 255));
        p.x = (int) (q.x + 0.1 * mag * cos(angle + PI / 5));
        p.y = (int) (q.y + 0.1 * mag * sin(angle + PI / 5));
        cv::line(show_filed, p, q, cv::Scalar(0, 0, 255));
        p.x = (int) (q.x + 0.1 * mag * cos(angle - PI / 5));
        p.y = (int) (q.y + 0.1 * mag * sin(angle - PI / 5));
        cv::line(show_filed, p, q, cv::Scalar(0, 0, 255));
      }
    }
  }
  cv::imshow("Visualized Field", show_filed);
  cv::waitKey();
}

void CoherentLine::GetCannyEdge() {
  cv::Canny(gray_, canny_edge_, 60, 120);
  canny_edge_ = 255 - canny_edge_;
  cv::imshow("Canny", canny_edge_);
  cv::waitKey();
}
