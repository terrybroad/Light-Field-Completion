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

//----------------------------------------------------------------------------------------------------------------------------------------
uchar getDepthMapColour(int depthIndex,int imgNum)
{
  return (uchar) (int)(255 - depthIndex*(255/imgNum));
}

//----------------------------------------------------------------------------------------------------------------------------------------
int getIndexFromDepthMap(uchar depthMapCol,int imgNum)
{
  double d = 1 - depthMapCol/255;

  d *= imgNum;

  if(d < 0){d = 0;}
  if(d >= imgNum-1){d = imgNum-1;}

  return (int) d;
}

//----------------------------------------------------------------------------------------------------------------------------------------
double getIndexFromDepthMapFloat(uchar depthMapCol,int imgNum)
{
  double d = 1 - depthMapCol/255;

  d *= imgNum;

  if(d < 0){d = 0;}
  if(d >= imgNum-1){d = imgNum-1;}

  return d;
}

//----------------------------------------------------------------------------------------------------------------------------------------
double getDepthDistance(uchar val, uchar val2)
{
  return (double) 1 - abs((int)val - (int)val2)/255;
}

//----------------------------------------------------------------------------------------------------------------------------------------
uchar getDepthDistanceUchar(uchar val, uchar val2)
{
  return (uchar) abs((int)val - (int)val2);
}

//----------------------------------------------------------------------------------------------------------------------------------------
vector<Mat> laplacianFocalStack(vector<Mat> &imgs)
{
  int imgNum = imgs.size();
  vector<Mat> laps;
  vector<Mat> imgsG;
  vector<Mat> smoothed;
  vector<Mat> boosted;

  imgsG.resize(imgNum);
  laps.resize(imgNum);
  smoothed.resize(imgNum);
  boosted.resize(imgNum);

  for(int i = 0; i < imgNum; i++)
  {
        cvtColor(imgs.at(i),imgsG.at(i), CV_BGR2GRAY);
        Laplacian(imgsG.at(i),laps.at(i),0,5);
        GaussianBlur(laps.at(i),smoothed.at(i),Size(55,55),10);
        smoothed.at(i).convertTo(boosted.at(i),0,2);
  }

  return boosted;
}

//----------------------------------------------------------------------------------------------------------------------------------------
vector<Mat> differenceOfGaussianFocalStack(vector<Mat> &imgs)
{
  int imgNum = imgs.size();

  vector<Mat> imgsG;
  vector<Mat> gauss;
  vector<Mat> diffs;
  vector<Mat> gaussDiffs;
  vector<Mat> boosted;
  imgsG.resize(imgNum);
  gauss.resize(imgNum);
  diffs.resize(imgNum);
  gaussDiffs.resize(imgNum);
  boosted.resize(imgNum);

  for(int i = 0; i < imgNum; i++)
  {
        cvtColor(imgs.at(i),imgsG.at(i), CV_BGR2GRAY);
        GaussianBlur(imgsG.at(i),gauss.at(i),Size(11,11),11);
        diffs.at(i) =  imgsG.at(i) - gauss.at(i);
        GaussianBlur(diffs.at(i),gaussDiffs.at(i),Size(101,101),11);
        gaussDiffs.at(i).convertTo(boosted.at(i),0,4);
  }

  return boosted;
}

//----------------------------------------------------------------------------------------------------------------------------------------
Mat averageImages(vector<Mat> &imgs)
{
  Mat avIm = Mat::zeros(imgs.at(0).size(), CV_8U);
  Mat avImBoosted = Mat::zeros(imgs.at(0).size(), CV_8U);;
  for(int y = 0; y < avIm.rows; y++)
  {
    for(int x = 0; x < avIm.cols; x++)
    {
      int val = 0;
      for(int i = 0; i < imgs.size(); i++)
      {
        val += (int) imgs.at(i).at<uchar>(y,x);
      }
      val /= imgs.size();
      avIm.at<uchar>(y,x) = (uchar) val;
    }
  }
  return avIm;
}

//----------------------------------------------------------------------------------------------------------------------------------------
Mat createDepthMap(vector<Mat> &imgs)
{
  Mat depthMap = Mat::zeros(imgs.at(0).size(), CV_8U);
  int rows = imgs.at(0).rows;
  int cols = imgs.at(0).cols;
  int imgNum = imgs.size();

  Mat av = averageImages(imgs);

  for(int y = 0; y < rows; y++)
  {
    for(int x = 0; x < cols; x++)
    {
      int bestValue = 0;
      int depthIndex = 0;

      for(int i = 0; i < imgNum; i++)
      {
        int value = 0;
        for(int j = -1; j < 2; j++)
        {
          for(int k = -1; k < 2; k++)
          {
            if(y+j >= 0 && y+j < rows && x+k >= 0 && x+k < cols)
            {
              value += (int) imgs.at(i).at<uchar>(y+j,x+k) - av.at<uchar>(y+j,x+k);
            }
          }
        }

        if(value > bestValue)
        {
          bestValue = value;
          depthIndex = i;
        }
      }


      depthMap.at<uchar>(y,x) = getDepthMapColour(depthIndex, imgNum); //(uchar) 255 - depthIndex*(255/imgNum);
    }
  }
  return depthMap;
}

//----------------------------------------------------------------------------------------------------------------------------------------
Mat getRelativeDepthMap(const Mat &img, uchar depthValue,int imgNum)
{
  Mat imgOut = Mat::ones(img.size(), CV_8U);

  for(int y = 0; y < imgOut.rows; y++)
  {
    for(int x = 0; x < imgOut.cols; x++)
    {
      imgOut.at<uchar>(y,x) = getDepthDistanceUchar(img.at<uchar>(y,x),depthValue);
    }
  }

  return imgOut;
}

//----------------------------------------------------------------------------------------------------------------------------------------
vector<Mat> getRelativeDepthMapStack(const Mat &img, int imgNum)
{
  vector<Mat> imgsOut;
  imgsOut.resize(imgNum);

  for(int i = 0; i < imgNum; i++)
  {
    imgsOut.at(i) = getRelativeDepthMap(img, getDepthMapColour(i, imgNum),imgNum);
  }
  return imgsOut;
}

//----------------------------------------------------------------------------------------------------------------------------------------
Mat createInFocusImage(Mat &depthMap, vector<Mat> &imgs)
{
  Mat inFocus = Mat::zeros(depthMap.size(), CV_8UC3);
  int rows = inFocus.rows;
  int cols = inFocus.cols;
  int imgNum = imgs.size();
  for(int y = 0; y < rows; y++)
  {
    for(int x = 0; x < cols; x++)
    {
      int depthIndex = 0;

      depthIndex = getIndexFromDepthMap(depthMap.at<uchar>(y,x), imgNum); //(255-depthMap.at<uchar>(y,x)) / (255/imgNum);
      inFocus.at<Vec3b>(y,x) = imgs.at(depthIndex).at<Vec3b>(y,x);

    }
  }
  return inFocus;
}

//----------------------------------------------------------------------------------------------------------------------------------------
Vec3b getInterpolatedCol(const Vec3b col1, const Vec3b col2, float ratio)
{
  Vec3b colOut;

  for(int i = 0; i < 3; i++)
  {
    colOut[i] = (uchar) (col1[i] * (ratio)) + (col2[i] * (1-ratio));
  }
  return colOut;
}

//----------------------------------------------------------------------------------------------------------------------------------------
Mat createInFocusImageInterpolate(Mat &depthMapBlurred, vector<Mat> &imgs)
{
  Mat inFocus = Mat::zeros(depthMapBlurred.size(), CV_8UC3);
  int rows = inFocus.rows;
  int cols = inFocus.cols;
  int imgNum = imgs.size();
  for(int y = 0; y < rows; y++)
  {
    for(int x = 0; x < cols; x++)
    {
      double depthIndex;
      depthIndex = getIndexFromDepthMapFloat(depthMapBlurred.at<uchar>(y,x), imgNum); //(255-depthMap.at<uchar>(y,x)) / (255/imgNum);

      inFocus.at<Vec3b>(y,x) = getInterpolatedCol(imgs.at(floor(depthIndex)).at<Vec3b>(y,x),imgs.at(floor(depthIndex+1)).at<Vec3b>(y,x), floor(depthIndex+1) - depthIndex);
    }
  }
  return inFocus;
}
