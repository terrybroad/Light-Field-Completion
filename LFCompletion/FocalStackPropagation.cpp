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
vector<int> getDepthMapIndicies(const Mat &depthMap, const Mat &mask,int imgNum)
{
  vector<int> indicies;
  int indexCount = 0;

  for(int y = 0; y < mask.rows; y++)
  {
    for(int x = 0; x < mask.cols; x++)
    {
      if((int) mask.at<uchar>(y,x) != 0)
      {
        if(indexCount == 0)
        {
          indexCount++;
          indicies.resize(indexCount);
          indicies.at(indexCount-1) = (int) depthMap.at<uchar>(y,x);
        }
        else
        {
          bool entered = false;
          for(int i = 0; i < indexCount; i++)
          {
            if((int) depthMap.at<uchar>(y,x) == indicies.at(i))
            {
              entered = true;
            }
          }
          if(!entered)
          {
            indexCount++;
            indicies.resize(indexCount);
            indicies.at(indexCount-1) = (int) depthMap.at<uchar>(y,x);
          }
        }
      }
    }
  }

  for(int i = 0; i < indexCount; i++)
  {
    indicies.at(i) = getIndexFromDepthMap((uchar) indicies.at(i),imgNum);
  }
  sort(indicies.begin(),indicies.end());

  return indicies;
}

//----------------------------------------------------------------------------------------------------------------------------------------
vector<Mat> splitSegments(const Mat &depthMap, const Mat &img, const Mat &mask, vector<int> &indicies, int imgNum)
{
  int arrSize = indicies.size();
  vector<Mat> segments;
  segments.resize(arrSize);

  for(int i = 0; i < arrSize; i++)
  {
    uchar depthMapCol = getDepthMapColour(indicies.at(i),imgNum);
    segments.at(i) = Mat::zeros(depthMap.size(), CV_8UC4);

    for(int y = 0; y < mask.rows; y++)
    {
      for(int x = 0; x < mask.cols; x++)
      {
        if((int) mask.at<uchar>(y,x) != 0 && (int)depthMap.at<uchar>(y,x) == (int)depthMapCol)
        {
          Vec4b s;
          Vec3b im;
          im = img.at<Vec3b>(y,x);
          s.val[0] = im.val[0];
          s.val[1] = im.val[1];
          s.val[2] = im.val[2];
          s.val[3] = (uchar) 255;

          segments.at(i).at<Vec4b>(y,x) = s;
        }
        else
        {
          segments.at(i).at<Vec4b>(y,x) = Vec4b(0,0,0,0);
        }
      }
    }
  }
  return segments;
}

//----------------------------------------------------------------------------------------------------------------------------------------
vector<Mat> splitSegmentMasks(const Mat &depthMap, const Mat &mask, vector<int> &indicies, int imgNum)
{
  int arrSize = indicies.size();
  vector<Mat> segments;
  segments.resize(arrSize);
  //vector<Mat> segmentsDilated;
  //segmentsDilated.resize(arrSize);
  //Mat strElement = getStructuringElement( MORPH_RECT,Size( 3, 3),Point(1,1));

  for(int i = 0; i < arrSize; i++)
  {
    uchar depthMapCol = getDepthMapColour(indicies.at(i),imgNum);
    segments.at(i) = Mat::zeros(depthMap.size(), CV_8U);

    for(int y = 0; y < mask.rows; y++)
    {
      for(int x = 0; x < mask.cols; x++)
      {
        if((int) mask.at<uchar>(y,x) != 0 && (int)depthMap.at<uchar>(y,x) == (int)depthMapCol)
        {
          segments.at(i).at<uchar>(y,x) = (uchar) 255;
        }
        else
        {
          segments.at(i).at<uchar>(y,x) = (uchar) 0;
        }
      }
    }
    //dilate(segments.at(i), segmentsDilated.at(i),strElement,Point(-1, -1), 1, 1, 1);
  }

  return segments;
}

//----------------------------------------------------------------------------------------------------------------------------------------
Rect getInFocusWindow(const Mat &laplace)
{
  Mat small;
  int scale = laplace.rows/10;
  resize(laplace,small,laplace.size()/scale);

  Point2i bestPixel;
  int bestValue = 0;

  for(int y = 0; y < small.rows; y++)
  {
    for(int x = 0; x < small.cols; x++)
    {
      int value = (int) small.at<uchar>(y,x);
      if(value > bestValue)
      {
        bestValue = value;
        bestPixel = Point2i(x,y);
      }
    }
  }
  return Rect((bestPixel*scale),Size(scale,scale));
}

//----------------------------------------------------------------------------------------------------------------------------------------
vector<Rect> getInFocusWindows(vector<Mat> &laplacians,vector<Mat> &relativeDepths)
{
  vector<Rect> windows;
  windows.resize(laplacians.size());

  Mat av = averageImages(laplacians);
  for(int i = 0; i < laplacians.size(); i++)
  {
    windows.at(i) = getInFocusWindow(laplacians.at(i) - av);
  }

  return windows;
}

//----------------------------------------------------------------------------------------------------------------------------------------
vector<Mat> getCroppedImages(const Rect window, vector<Mat> &imgs, const Mat &distanceMap)
{
  vector<Mat> smallImgs;
  vector<Mat> smallImgsG;
  smallImgs.resize(imgs.size());
  smallImgsG.resize(imgs.size());


  for(int i = 0; i < imgs.size(); i++)
  {
    smallImgs.at(i) = imgs.at(i)(window);
    cvtColor(smallImgs.at(i),smallImgsG.at(i),CV_BGR2GRAY);
    smallImgsG.at(i) = smallImgsG.at(i) - distanceMap(window);
  }

 return smallImgsG;
}

//----------------------------------------------------------------------------------------------------------------------------------------
double imDistance(const Mat &focused, const Mat &notFocused)
{
  Mat diff;
  absdiff(notFocused,focused,diff);
  return mean(diff)[0];
}

//----------------------------------------------------------------------------------------------------------------------------------------
int getCoeff(const Mat &focused, const Mat &notFocused)
{
  Mat blurred;
  int bestCoeff = 0;
  double lowestDistance = 500;
  int stepSize = 3;
  int kSize = 0;

  for(int i = 1; i < stepSize*2; i++)
  {
    kSize = i*stepSize;
    kSize = (kSize*2)+1;

    GaussianBlur(focused,blurred,Size(kSize,kSize),0);
    double dist = imDistance(blurred,notFocused);
    if(dist < lowestDistance)
    {
      bestCoeff = i*stepSize;
      lowestDistance = dist;
    }
  }

  int ogCoeff = bestCoeff;

    for(int i = -stepSize; i < stepSize; i++)
    {
      if(ogCoeff + i > 0)
      {
        kSize = ogCoeff+i;
        kSize = (kSize*2)+1;

        GaussianBlur(focused,blurred,Size(kSize,kSize),kSize);
        double dist = imDistance(blurred,notFocused);
        if(dist < lowestDistance)
        {
          bestCoeff = ogCoeff + i;
          lowestDistance = dist;
        }
      }
    }

  return (bestCoeff*2)+1;
}

//----------------------------------------------------------------------------------------------------------------------------------------
vector<int> getCoefficients(vector<Mat> &imgs, int focusedImage)
{
  vector<int> coefficients;
  coefficients.resize(imgs.size());

  for(int i = 0; i < imgs.size(); i++)
  {
    if(i == focusedImage)
    {
      coefficients.at(i) = 0;
    }
    else
    {
      coefficients.at(i) = getCoeff(imgs.at(focusedImage), imgs.at(i));
    }
  }
  return coefficients;
}

//----------------------------------------------------------------------------------------------------------------------------------------
Mat superImpose(const Mat &background, const Mat &foreground, const Mat &mask)
{
  Mat out = background.clone();
  if(background.size() == foreground.size())
  {
    for(int y = 0; y < out.rows; y++)
    {
      for(int x = 0; x < out.cols; x++)
      {
        Vec3b bVal = background.at<Vec3b>(y,x);
        Vec3b foVal = foreground.at<Vec3b>(y,x);

        int alpha = (int) mask.at<uchar>(y,x);
        if(alpha > 0)
        {
          double ratio = ((double)alpha/255);

          for(int i = 0; i < 3; i++)
          {
            out.at<Vec3b>(y,x)[i] = (uchar) floor(  (((double)bVal[i]) * (1-ratio) ) + ((double)foVal[i]* ratio) ) ;
          }
        }
        else
        {
          out.at<Vec3b>(y,x) = bVal;
        }
      }
    }
  }
  return out;
}


//----------------------------------------------------------------------------------------------------------------------------------------
vector<Mat> propagateSegment(vector<Mat> &imgs, const Mat &infilled, const Mat &segmentMask, const Mat&relativeDepth, const Rect window, int imgIndex)
{
  vector<Mat> imgsOut;
  vector<Mat> smallImgs;
  imgsOut.resize(imgs.size());
  smallImgs = getCroppedImages(window,imgs,relativeDepth);
  vector<int> coefficients = getCoefficients(smallImgs, imgIndex);

  Mat segmentMaskDilated,blurredMask, blurredMask2, blurredInfilled;

  GaussianBlur(segmentMask,blurredMask, Size(5,5), 5);
  for(int i = 0; i < imgs.size(); i++)
  {
    cout << "propagating layer - " + to_string(imgIndex) + " to layer " + to_string(i) << endl;
    int kSize = coefficients.at(i);
    if(i != imgIndex)
    {
      GaussianBlur(infilled, blurredInfilled, Size(kSize,kSize), kSize);
      imgsOut.at(i) = superImpose(imgs.at(i),blurredInfilled, blurredMask);

      Mat strElement = getStructuringElement( MORPH_RECT,Size( kSize, kSize),Point(1,1));
      dilate(segmentMask, segmentMaskDilated,strElement,Point(-1, -1), 1, 1, 1);
      GaussianBlur(segmentMaskDilated,blurredMask, Size(kSize,kSize), kSize);

      imgsOut.at(i) = superImpose(imgsOut.at(i),blurredInfilled, blurredMask - relativeDepth*0.5);
    }
    else if(i == imgIndex)
    {
      imgsOut.at(i) = superImpose(imgs.at(i),infilled, blurredMask - relativeDepth);
    }
  }

  return imgsOut;
}

//----------------------------------------------------------------------------------------------------------------------------------------
vector<Mat> propogateFocalStack(vector<Mat> &imgs, vector<Mat> &laplacians, const Mat &infilled, const Mat &mask, const Mat &depthMap,const Mat &depthMapBlurred)
{
  vector<Mat> imgsOut = imgs;
  vector<Mat> relativeDepths = getRelativeDepthMapStack(depthMapBlurred,imgs.size());
  vector<Rect> inFocusWindows = getInFocusWindows(laplacians,relativeDepths);
  vector<int> segmentIndicies = getDepthMapIndicies(depthMap,mask,imgs.size());
  vector<Mat> segments = splitSegmentMasks(depthMap,mask,segmentIndicies,imgs.size());


  for(int i = segments.size()-1; i > -1; i--)
  {
    int segmentIndex = segmentIndicies.at(i);
    imgsOut = propagateSegment(imgsOut, infilled, segments.at(i), relativeDepths.at(segmentIndex), inFocusWindows.at(segmentIndex), segmentIndex);
  }

  return imgsOut;
}
