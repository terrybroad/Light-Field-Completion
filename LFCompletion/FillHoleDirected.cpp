#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgproc/types_c.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/photo/photo.hpp"
#include "pixelStruct.h"
#include <iostream>

using namespace cv;
using namespace std;


//----------------------------------------------------------------------------------------------------------------------------------------
bool inImage(int x, int y, const Mat &img)
{
  return (x >= 0 && x < img.cols && y >= 0 && y < img.rows);
}

//----------------------------------------------------------------------------------------------------------------------------------------
bool windowInImage(int x, int y, const Mat &img, int windowSize)
{
  return (x - windowSize >= 0 && x + windowSize < img.cols && y - windowSize >= 0 && y + windowSize < img.rows);
}

//----------------------------------------------------------------------------------------------------------------------------------------
Mat getNeighbourhoodWindow(const Mat &img, Point2i pt, int windowSize)
{
  Mat output = Mat(windowSize * 2 + 1, windowSize * 2 + 1, 16);

  for(int y = 0; y < output.rows; y++)
  {
    for(int x = 0; x < output.cols; x++)
    {
      if(inImage( pt.x - windowSize + x, pt.y - windowSize + y, img))
      {
        output.at<int>(y, x, 0) = img.at<int>(pt.y - windowSize + y, pt.x - windowSize + x, 0);
      }
    }
  }

  return output;
}

//----------------------------------------------------------------------------------------------------------------------------------------
Mat getNeighbourhoodWindowFast(const Mat &img, Point2i pt, int windowSize)
{
  return img(Rect(Point2i(pt.x - windowSize, pt.y - windowSize), Size(windowSize * 2 + 1, windowSize * 2 + 1)));
}

//----------------------------------------------------------------------------------------------------------------------------------------
Mat getNeighbourhoodWindowMask(const Mat &img, Point2i pt, int windowSize)
{
  Mat output = Mat(windowSize * 2 + 1, windowSize * 2 + 1, 16);

  for(int y = 0; y < output.rows; y++)
  {
    for(int x = 0; x < output.cols; x++)
    {
      if(inImage( pt.x - windowSize + x, pt.y - windowSize + y, img))
      {
        output.at<uchar>(y, x, 0) = img.at<uchar>(pt.y - windowSize + y, pt.x - windowSize + x, 0);
      }
      else
      {
        output.at<uchar>(y, x, 0) = (uchar)1;
      }
    }
  }

  return output;
}
//----------------------------------------------------------------------------------------------------------------------------------------
double getDistSimple(const Mat &templ8, const Mat &templ9, int windowSize)
{
  double dist = 0;
  int count = 0;

  Mat diff;
  absdiff(templ8, templ9, diff);


  for(int i = 0; i < (windowSize*2) + 1; i++)
  {
    for(int j = 0; j < (windowSize*2) + 1; j++)
    {
      if( i != windowSize && j != windowSize)
      {
        count++;
        dist += abs((int) diff.at<uchar>(i,j));
      }
    }
  }

  return dist/count;
}

//----------------------------------------------------------------------------------------------------------------------------------------
const Point2i findBestPixelSimple(const Mat &templ8, const vector<Mat> &templates, const Mat &mask, int rows, int cols, const Point2i pos, int windowSize)
{
  Point2i bestPixel;
  Point2i ptemp;
  double bestValue = 100;

  int n = 1;
  int count = 0;
  bool yes = false;

  while(count < 30)
  {
    for(int i = -1; i < 2; i+=2) //ALTERNATE SIGN (-,-) then (-,+) then (+,-) then (+,+) -1,-1 ... -1,1 .. 1,-1,
    {
      for(int j = -1; j < 2; j+=2) //ALTERNATE SIGN (-,-) then (-,+) then (+,-) then (+,+)
      {
        for(int c = 0; c < 2*n; c++)
        {
          if(i < 0 && j < 0) {ptemp = Point2i(pos.x + (n*i) + c, pos.y + (n*j));}
          if(i < 0 && j > 0) {ptemp = Point2i(pos.x + (n*i) , pos.y + (n*j) - c );}
          if(i > 0 && j < 0) {ptemp = Point2i(pos.x + (n*i) , pos.y + (n*j) + c);}
          if(i > 0 && j > 0) {ptemp = Point2i(pos.x + (n*i) - c, pos.y + (n*j));}

          if(windowInImage(ptemp.x,ptemp.y,mask,windowSize))
          {
            if((int) mask.at<uchar>(ptemp.y,ptemp.x,0) == 0)
            {
              double dist = getDistSimple(templ8,templates.at(ptemp.y*cols + ptemp.x),windowSize);
              count++;
              if(dist < bestValue)
              {
                bestValue = dist;
                bestPixel.x = ptemp.x;
                bestPixel.y = ptemp.y;
              }
            }
          }
        }
      }
    }

    n++;
  }
  return bestPixel;
}

Mat fillDepthMapDirected(const Mat &depthMap, const Mat &inpaintMask)
{
  Mat inpainted,maskBinary,output;
  int windowSize = 2;
  int winLength = (windowSize*2) + 1;
  inpaint(depthMap, inpaintMask, inpainted, 3, INPAINT_TELEA);
  output = inpainted.clone();

  vector<Mat> templates(inpainted.rows*inpainted.cols);
  for(int y = 0; y < inpainted.rows; y++)
  {
    for(int x = 0; x < inpainted.cols; x++)
    {
      if(windowInImage(x,y,inpainted,windowSize)) { templates.at(y*inpainted.cols + x) = getNeighbourhoodWindow(inpainted,Point2i(x,y),windowSize); }
    }
  }


  for(int y = 0; y < inpainted.rows; y++)
  {
    for(int x = 0; x < inpainted.cols; x++)
    {
      if( (int) inpaintMask.at<uchar>(y,x,0) != 0)
      {
        Mat templ8 = getNeighbourhoodWindow(inpainted,Point2i(x,y),windowSize);
        Point2i newPos = findBestPixelSimple(templ8,templates,inpaintMask,inpainted.rows,inpainted.cols,Point2i(x,y),windowSize);
        output.at<int>(y,x,0) = inpainted.at<int>(newPos.y,newPos.x,0);

      }
     }
  }

  return output;
}


//----------------------------------------------------------------------------------------------------------------------------------------

bool isNeighbour(const Point2i pos, const Point2i ptemp, vector<Point2i> &linearArray, int windowSize,int cols)
{
  bool isN = false;

  for(int y = (-1*windowSize)+1; y < windowSize; y++)
  {
    for(int x = (-1*windowSize)+1; x < windowSize; x++)
    {
      if(!(y == 0 && x == 0))
      {
        if(linearArray.at((pos.y+ y)*cols + pos.x+x) == ptemp)
        {
          isN = true;
        }
      }
    }
  }
  return isN;
}

//----------------------------------------------------------------------------------------------------------------------------------------
double getDistDirected(const Mat &templ8, const Mat &templ9, const Mat &gaussian, int windowSize)
{
  double dist = 0;
  int count = 0;

  Mat diff;

  absdiff(templ8, templ9, diff);

  diff.mul(gaussian);

  for(int i = 0; i < (windowSize*2) + 1; i++)
  {
    for(int j = 0; j < (windowSize*2) + 1; j++)
    {
      if(!( i == windowSize && j == windowSize))
      {
        for(int k = 0; k < 3; k++)
        {
          dist += (diff.at<Vec3b>(i,j)[k])^2;
          count++;
        }
      }
    }
  }

  return sqrt(dist)/count;
}

//----------------------------------------------------------------------------------------------------------------------------------------
pixelStruct findBestPixelAshikhmin(const Mat &templ8, const vector<Mat> &templates, const Mat &gaussian, const Mat &mask, const Mat &depthMap, int rows, int cols, const Point2i pos, vector<Point2i> &linearArray, int windowSize)
{
  Point2i bestPixel = pos;
  Point2i ptemp;
  double bestValue = 1000000;


  int n = 1;
  int count = 0;


  for(int y = (-1*windowSize)+1; y < windowSize; y++)
  {
    for(int x = (-1*windowSize)+1; x < windowSize; x++)
    {
      int num = ((pos.y+y)*cols) + pos.x + x;
      if(num >= 0 && num < rows*cols)
      {
        ptemp = linearArray.at(num);
        if(ptemp.x != -1)
        {
          ptemp.x = ptemp.x + x*-1;
          ptemp.y = ptemp.y + y*-1;

          if(pos!= ptemp && windowInImage(ptemp.x,ptemp.y,mask,windowSize) && !((int) mask.at<uchar>(ptemp.y,ptemp.x,0) != 0)) //!isNeighbour(pos,ptemp,linearArray,floor(windowSize/2),cols)
          {
            double dist = getDistDirected(templ8, templates.at(ptemp.y*cols +ptemp.x),gaussian,windowSize);
            double depthDistance = getDepthDistance(depthMap.at<uchar>(pos), depthMap.at<uchar>(ptemp));
            dist /= depthDistance;

            if(isNeighbour(pos,ptemp,linearArray,windowSize,cols))
            {
              dist *= 1.4;
            }
            if(dist < bestValue)
            {
              bestValue = dist;
              bestPixel.x = ptemp.x;
              bestPixel.y = ptemp.y;
            }
          }
        }
      }
    }
  }

  pixelStruct p;

  if(pos != bestPixel)
  {
    p.x = bestPixel.x;
    p.y = bestPixel.y;
    p.distance = bestValue;
  }
  else
  {
    p.x = -2;
    p.y = -2;
    p.distance = 500;
  }

  return p;
}

//----------------------------------------------------------------------------------------------------------------------------------------
const Point2i findBestPixelExhaustive(int searchRange, const Mat &templ8, const vector<Mat> &templates, const Mat &gaussian, const Mat &mask, const Mat &depthMap, int rows, int cols,const Point2i pos, vector<Point2i> &linearArray, int windowSize)
{
  Point2i bestPixel = pos;
  Point2i ptemp;
  double bestValue = 100;


  int n = 1;
  int count = 0;

    bestValue = 100;
    while(count < searchRange && n < 250)
    {
      for(int i = -1; i < 2; i+=2) //ALTERNATE SIGN (-,-) then (-,+) then (+,-) then (+,+) -1,-1 ... -1,1 .. 1,-1,
      {
        for(int j = -1; j < 2; j+=2) //ALTERNATE SIGN (-,-) then (-,+) then (+,-) then (+,+)
        {
          for(int c = 0; c < 2*n; c++)
          {
            if(i < 0 && j < 0) {ptemp = Point2i(pos.x + (n*i) + c, pos.y + (n*j));}
            if(i < 0 && j > 0) {ptemp = Point2i(pos.x + (n*i) , pos.y + (n*j) - c );}
            if(i > 0 && j < 0) {ptemp = Point2i(pos.x + (n*i) , pos.y + (n*j) + c);}
            if(i > 0 && j > 0) {ptemp = Point2i(pos.x + (n*i) - c, pos.y + (n*j));}

            if(windowInImage(ptemp.x,ptemp.y,mask,windowSize))
            {
              if(!((int) mask.at<uchar>(ptemp.y,ptemp.x,0) != 0))
              {
                double dist = getDistDirected(templ8, templates.at(ptemp.y*cols + ptemp.x),gaussian,windowSize);
                double depthDistance = getDepthDistance(depthMap.at<uchar>(pos), depthMap.at<uchar>(ptemp));
                dist /= depthDistance;

                if(isNeighbour(pos,ptemp,linearArray,windowSize,cols))
                {
                  dist *= 1.4;
                }

                count++;
                if(dist < bestValue)
                {
                  bestValue = dist;
                  bestPixel.x = ptemp.x;
                  bestPixel.y = ptemp.y;
                }
              }
            }
          }
        }
      }
      n++;
    }
    cout << "Exhaustive search" << endl; cout << bestValue << endl;

  return bestPixel;
}

//----------------------------------------------------------------------------------------------------------------------------------------
vector<Point2i> getLinearArray(const Mat &maskBinary)
{
  vector<Point2i> linArr;
  linArr.resize(maskBinary.rows*maskBinary.cols);

  for(int y = 0; y < maskBinary.rows; y++)
  {
    for(int x = 0; x < maskBinary.cols; x++)
    {
      if((int) maskBinary.at<uchar>(y,x,0) != 0)
      {
        linArr.at(y*maskBinary.cols + x).x = -1;
        linArr.at(y*maskBinary.cols + x).y = -1;
      }
      else
      {
        linArr.at(y*maskBinary.cols + x).x = x;
        linArr.at(y*maskBinary.cols + x).y = y;
      }
    }
  }

  return linArr;
}

//----------------------------------------------------------------------------------------------------------------------------------------
vector<Mat> getTemplates(const Mat &img, int windowSize)
{
  vector<Mat> templates;
  templates.resize(img.rows*img.cols);

  for(int y = 0; y < img.rows; y++)
  {
    for(int x = 0; x < img.cols; x++)
    {
      if(windowInImage(x,y,img,windowSize)) { templates.at(y*img.cols + x) = getNeighbourhoodWindowFast(img,Point2i(x,y),windowSize); }

      cout << "Create template - x : " + to_string(x) + " y : " + to_string(y) << endl;
    }
  }
  return templates;
}


//----------------------------------------------------------------------------------------------------------------------------------------
Mat getGaussianMask(int windowSize)
{
  double kernalsize = (double)windowSize / 6.0;
  kernalsize = sqrt(kernalsize);
  Mat tmpGaussian = getGaussianKernel(windowSize * 2 + 1, kernalsize);
  return tmpGaussian * tmpGaussian.t();
}

//----------------------------------------------------------------------------------------------------------------------------------------
double gaussianMaskWeight(const Mat &gaussianMask)
{
  Mat blank = Mat::zeros(gaussianMask.size(), CV_8U);
  Mat white = Mat::ones(gaussianMask.size(), CV_8U);
  white*255;

  return getDistDirected(blank,white,gaussianMask,(gaussianMask.rows-1)/2);
}

//----------------------------------------------------------------------------------------------------------------------------------------
Mat addNoiseWithMask(const Mat &img, const Mat &maskBinary, RNG &rng)
{
  Mat out = img.clone();
  for(int y = 0; y < img.rows; y++)
  {
    for(int x = 0; x < img.cols; x++)
    {
      if((int) maskBinary.at<uchar>(y,x,0) != 0)
      {
        Vec3b a;
        a = img.at<Vec3b>(y,x);
        int randInt = rng.uniform(-17,17);
        for(int i = 0; i < 3; i++)
        {
          int num = (int)a.val[i] + randInt;
          if(num < 0){num = 0;}
          if(num > 255){num = 255;}
          a.val[i] = num;
        }
        out.at<Vec3b>(y,x) = a;
      }
    }
  }
  return out;
}

//----------------------------------------------------------------------------------------------------------------------------------------
int countUnfilled(const Mat &maskBinary)
{
  int unfilledCount = 0;

  for(int y = 0; y < maskBinary.rows; y++)
  {
    for(int x = 0; x < maskBinary.cols; x++)
    {
      if((int) maskBinary.at<uchar>(y,x,0) != 0)
      {
        unfilledCount++;
      }
    }
  }

  return unfilledCount;
}

//----------------------------------------------------------------------------------------------------------------------------------------
Mat fillImageDirected(const Mat &inpainted, const Mat &depthMap, const Mat&depthMapBlurred, const Mat &inpaintMask, int windowSize, int searchSize)
{
      Mat noisy, outLAB, outBGR;
      Mat maskBinary, originalMask, maskBig, maskBigger;
      Mat eroded, border, borderBig;
      Mat strElement,strElement2;
      Mat sampleCount;
      Mat gaussianMask;

      vector<Point2i> linearArray;
      vector<Mat> templates;

      RNG rng;

      int winLength = (windowSize*2) + 1;
      int rows = inpainted.rows;
      int cols = inpainted.cols;

      strElement = getStructuringElement( MORPH_RECT,Size( 3, 3),Point(-1,-1));
      strElement2 = getStructuringElement( MORPH_RECT,Size( windowSize+1, windowSize+1),Point(-1,-1));
      maskBinary = Mat::zeros(inpainted.size(), CV_8U);
      eroded = Mat::zeros(maskBinary.size(), CV_8U);
      border = Mat::zeros(maskBinary.size(), CV_8U);
      borderBig = Mat::zeros(maskBinary.size(), CV_8U);

      cout << "initialised matricies" << endl;

      threshold( inpaintMask, maskBinary, 127,255,0 );

      cout << "created binary mask" << endl;

      linearArray = getLinearArray(maskBinary);

      cout << "position index array created" << endl;

      noisy = addNoiseWithMask(inpainted,maskBinary, rng);

      cout << "noise added" << endl;

      cvtColor(inpainted, outLAB, CV_BGR2Lab);

      cout << "convereted to LAB colour space" << endl;

      templates = getTemplates(outLAB,windowSize);

      cout << "templates created" << endl;

      gaussianMask = getGaussianMask(windowSize);

      originalMask = maskBinary.clone();

      int unfilledCount = countUnfilled(maskBinary);

      cout << "pixels to be filled - " + to_string(unfilledCount) << endl;

      int epoch = 0;

      double winSizeWeight = gaussianMaskWeight(gaussianMask);

      winSizeWeight *= 1.75;

      while(unfilledCount > 0)
      {
        erode(maskBinary,eroded,strElement,Point(-1, -1), 1, 1, 1);
        subtract(maskBinary,eroded,border);

        for(int y = 0; y < rows; y++)
        {
          for(int x = 0; x < cols; x++)
          {
            if( (int) border.at<uchar>(y,x,0) != 0 )
            {
              unfilledCount--;
              if(windowInImage(x,y,originalMask,windowSize))
              {
                Mat templ8 = getNeighbourhoodWindowFast(outLAB,Point2i(x,y),windowSize);

                pixelStruct p; p.x = -2; p.y = -2; p.distance = 500;
                Point2i newPos;

                cout << "threshold" << endl; cout << winSizeWeight << endl;
                if(epoch > 0)
                {
                  p = findBestPixelAshikhmin(templ8,templates,gaussianMask,originalMask,depthMapBlurred,rows,cols,Point2i(x,y),linearArray,windowSize);
                }
                if(p.distance < winSizeWeight)
                {
                  outLAB.at<int>(y,x,0) = outLAB.at<int>(p.y,p.x,0);
                  newPos.x = p.x;
                  newPos.y = p.y;

                  cout << "Ashikhmin search" << endl; cout << p.distance << endl;
                }
                else
                {
                  newPos = findBestPixelExhaustive(searchSize,templ8,templates,gaussianMask,originalMask,depthMapBlurred,rows,cols,Point2i(x,y),linearArray,windowSize);
                  outLAB.at<int>(y,x,0) = outLAB.at<int>(newPos.y,newPos.x,0);

                }

                if(newPos != Point2i(x,y))
                {
                  linearArray.at(y*cols + x) = newPos;
                }

                cout << "current out pixel" << endl; cout << Point2i(x,y) << endl;
                cout << "picked pixel" << endl; cout << newPos << endl;
                cout << "unfilled count " << endl; cout << unfilledCount << endl;
                cout << "epoch " << endl; cout << epoch << endl;
              }
            }
           }
        }


        dilate(border,borderBig,strElement2,Point(-1, -1), 1, 1, 1);
        for(int y = 0; y < rows; y++)
        {
          for(int x = 0; x < cols; x++)
          {
            if(windowInImage(x,y,outLAB,windowSize) && (int) borderBig.at<uchar>(y,x,0) != 0) { templates.at(y*cols + x) = getNeighbourhoodWindow(outLAB,Point2i(x,y),windowSize); }
          }
        }

        maskBinary = eroded.clone();
        epoch++;
      }


      cvtColor(outLAB, outBGR, CV_Lab2BGR);
      cout << "convereted to BGR colour space" << endl;
      return outBGR;
}
