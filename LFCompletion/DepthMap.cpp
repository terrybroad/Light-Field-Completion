#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgproc/types_c.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/photo/photo.hpp"
#include <iostream>
#include <algorithm>

using namespace cv;
using namespace std;


vector<Mat> createDepthMap(vector<Mat> imgs)
{

    int imgNum = imgs.size();
    vector<Mat> laps;
    vector<Mat> imgsG;
    vector<Mat> smoothed;
    vector<Mat> gauss;
    vector<Mat> diffs;
    vector<Mat> gaussDiffs;

    imgsG.resize(imgNum);
    laps.resize(imgNum);
    smoothed.resize(imgNum);
    gauss.resize(imgNum);
    diffs.resize(imgNum);
    gaussDiffs.resize(imgNum);

    for(int i = 0; i < imgNum; i++)
    {
          cvtColor(imgs.at(i),imgsG.at(i), CV_BGR2GRAY);
          Laplacian(imgsG.at(i),laps.at(i),0,5);
          GaussianBlur(laps.at(i),smoothed.at(i),Size(55,55),10);

          GaussianBlur(imgsG.at(i),gauss.at(i),Size(11,11),11);
          diffs.at(i) =  imgsG.at(i) - gauss.at(i);
          GaussianBlur(diffs.at(i),gaussDiffs.at(i),Size(101,101),11);
    }

  int rows = laps.at(0).rows;
  int cols = laps.at(0).cols;

  Mat depthMap; Mat inFocus; Mat depthMap2; Mat inFocus2;
  depthMap = Mat::zeros(laps.at(0).size(), CV_8U);
  inFocus = Mat::zeros(imgs.at(0).size(), CV_8UC3);
  depthMap2 = Mat::zeros(laps.at(0).size(), CV_8U);
  inFocus2 = Mat::zeros(imgs.at(0).size(), CV_8UC3);

  for(int y = 0; y < rows; y++)
  {
    for(int x = 0; x < cols; x++)
    {
      int bestValue = 0;
      int depthIndex = 0;
      int bestValue2 = 0;
      int depthIndex2 = 0;

      if( y >= 3 && y < rows-3 && x>= 3 && x < cols-3)
      {
        for(int i = 0; i < imgNum; i++)
        {
          int value = 0;
          int value2 = 0;
          for(int j = -1; j < 2; j++)
          {
            for(int k = -1; k < 2; k++)
            {
              value += (int) smoothed.at(i).at<uchar>(y+j,x+k);
              value2 += (int) gaussDiffs.at(i).at<uchar>(y+j,x+k);
            }
          }

          if(value > bestValue)
          {
            bestValue = value;
            depthIndex = i;
          }
          if(value2 > bestValue2)
          {
            bestValue2 = value2;
            depthIndex2 = i;
          }
        }
      }
      depthMap.at<uchar>(y,x) = (uchar) 255 - depthIndex*(255/imgNum);
      inFocus.at<Vec3b>(y,x) = imgs.at(depthIndex).at<Vec3b>(y,x);

      depthMap2.at<uchar>(y,x) = (uchar) 255 - depthIndex2*(255/imgNum);
      inFocus2.at<Vec3b>(y,x) = imgs.at(depthIndex2).at<Vec3b>(y,x);
    }
  }

  vector<Mat> output;
  output.resize(2);
  output.at(0) = depthMap;
  output.at(1) = inFocus;

  return output;
}
