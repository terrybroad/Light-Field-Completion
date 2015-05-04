#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgproc/types_c.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/photo/photo.hpp"
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
double getDist(const Mat &templ8, const Mat &templ9, const Mat &mask1, const Mat &mask2, const Mat &gaussian,int windowSize)
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
        //if((int) mask1.at<uchar>(i,j) == 0 && (int) mask2.at<uchar>(i,j) == 0)
        {
          Vec3b a,b,c;
          a = templ8.at<Vec3b>(i,j);
          b = templ9.at<Vec3b>(i,j);
          c = diff.at<Vec3b>(i,j);
          for(int k = 0; k < 3; k++)
          {
            count++;
            if((int) mask1.at<uchar>(i,j) == 0 )
            {
              dist +=  (c.val[k]*2)^2;
              //dist +=  (c.val[k]*2);
            }
             else // penalise non-image elements
            {
            //
              dist += (c.val[k])^2;
            //  dist += (c.val[k]);
            }
          }
        }
      }
    }
  }

  if(dist == 0)
  {
    cout << "LUCKY NUMBER ZERO" << endl;
  }

  return sqrt(dist)/count;
  //return dist/count;
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
const Point2i findBestPixel(const Mat &templ8, const vector<Mat> &templates, const Mat &gaussian,const Mat &mask,const vector<Mat> &masks, const Mat &sampleCount, const Mat &depthMap, int rows, int cols,const Point2i pos, vector<Point2i> &linearArray, int windowSize)
{
  Point2i bestPixel = pos;
  Point2i ptemp;
  double bestValue = 100;


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
        ptemp.x = ptemp.x + x*-1;
        ptemp.y = ptemp.y + y*-1;

        if(ptemp.x != -1 && windowInImage(ptemp.x,ptemp.y,mask,windowSize))
        {
          if(pos!= ptemp && !isNeighbour(pos,ptemp,linearArray,windowSize,cols) && mask.at<uchar>(ptemp) == 0 && depthMap.at<uchar>(pos) == depthMap.at<uchar>(ptemp)) //!isNeighbour(pos,ptemp,linearArray,floor(windowSize/2),cols)
          {
            double dist = getDist(templ8, templates.at(ptemp.y*cols +ptemp.x),masks.at(pos.y*cols + pos.x),masks.at(ptemp.y*cols +ptemp.x),gaussian,windowSize);
            dist *= sampleCount.at<float>(pos.y*cols+pos.x);
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

  if(bestValue < 0.12)
  {
    cout << "Ashihkmin" << endl;
    cout << bestValue << endl;
  }
  else
  {
    bestValue = 100;
    while(count < 1500 && n < 250)
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
                if(depthMap.at<uchar>(pos) == depthMap.at<uchar>(ptemp))
                {
                  if(!isNeighbour(pos,ptemp,linearArray,windowSize,cols))
                  {
                    double dist = getDist(templ8,templates.at(ptemp.y*cols + ptemp.x),masks.at(pos.y*cols + pos.x),masks.at(ptemp.y*cols + ptemp.x),gaussian,windowSize);
                    dist *= sampleCount.at<float>(pos.y*cols+pos.x);
                    if( sampleCount.at<float>(pos.y*cols+pos.x) < 2) { count++;}
                    //count++;
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
        }
      }
      n++;
    }
    cout << "exhaustive search" << endl;
    cout << bestValue << endl;
  }
  return bestPixel;
}

//----------------------------------------------------------------------------------------------------------------------------------------
const Point2i findBestPixelSimple(const Mat &templ8, const vector<Mat> &templates, const Mat &mask, int rows, int cols, const Point2i pos, int windowSize)
{
  Point2i bestPixel;
  Point2i ptemp;
  double bestValue = 100;
  //double dist = 0;

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

Mat fillDepthMap(const Mat &depthMap, const Mat &inpaintMask)
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

Mat fillImage(const Mat &srcBGR, const Mat &depthMap, const Mat &inpaintMask)
{
  Mat srcLAB, texLAB, outLAB, outBGR, maskBinary,originalMask,eroded, border, strElement,sampleCount,borderBig,strElement2,strElement3,strElement4,inpainted,inpainted2,inpainted3,maskBig,maskBigger;

    RNG rng;

    int windowSize = 14;
    int winLength = (windowSize*2) + 1;

    int rows = srcBGR.rows;
    int cols = srcBGR.cols;

    strElement = getStructuringElement( MORPH_RECT,Size( 5, 5),Point(1,1));
    dilate(inpaintMask,inpaintMask,strElement,Point(-1, -1), 1, 1, 1);
    maskBinary = Mat::zeros(srcBGR.size(), CV_8U);
    threshold( inpaintMask, maskBinary, 127,255,0 );
    //strElement = getStructuringElement( MORPH_RECT,Size( 3, 3),Point(1,1));
    strElement2 = getStructuringElement( MORPH_RECT,Size( windowSize, windowSize),Point(-1,-1));
    strElement3 = getStructuringElement( MORPH_RECT,Size( windowSize, 1),Point(-1,-1));
    strElement4 = getStructuringElement( MORPH_RECT,Size( 1, windowSize),Point(-1,-1));
    eroded = Mat::zeros(maskBinary.size(), CV_8U);
    border = Mat::zeros(maskBinary.size(), CV_8U);
    borderBig = Mat::zeros(maskBinary.size(), CV_8U);
    sampleCount = Mat::ones(maskBinary.size(), CV_32F);

    double kernalsize = (double)windowSize / 6.0;
    kernalsize = sqrt(kernalsize);
    Mat tmpGaussian = getGaussianKernel(windowSize * 2 + 1, kernalsize);
    Mat gaussianMask = tmpGaussian * tmpGaussian.t();

    inpaint(srcBGR, inpaintMask, inpainted, 3, INPAINT_TELEA);
    dilate(inpaintMask,maskBig,strElement3,Point(-1, -1), 1, 1, 1);
    inpaint(srcBGR, maskBig, inpainted2, 3, INPAINT_TELEA);
    dilate(inpaintMask,maskBigger,strElement4,Point(-1, -1), 1, 1, 1);
    inpaint(srcBGR, maskBig, inpainted3, 3, INPAINT_TELEA);

    Mat inpaintCombine = inpainted.clone();

    int unfilledCount = 0;

    vector<Point2i> linearArray;
    linearArray.resize(rows*cols);

    for(int y = 0; y < rows; y++)
    {
      for(int x = 0; x < cols; x++)
      {
        if((int) maskBinary.at<uchar>(y,x,0) != 0)
        {
          unfilledCount++;

          linearArray.at(y*cols + x).x = -1;
          linearArray.at(y*cols + x).y = -1;

          Vec3b a,b,c;
          a = inpainted.at<Vec3b>(y,x);
          b = inpainted2.at<Vec3b>(y,x);
          c = inpainted3.at<Vec3b>(y,x);
          int randInt = rng.uniform(-17,17);
          for(int i = 0; i < 3; i++)
          {
            int num = (int)((a.val[i] + b.val[i] + c.val[i] )/3) + randInt;
            if(num < 0){num = 0;}
            if(num > 255){num = 255;}
            a.val[i] = num;
          }
          inpaintCombine.at<Vec3b>(y,x) = a;
        }
        else
        {
          linearArray.at(y*cols + x).x = x;
          linearArray.at(y*cols + x).y = y;
        }
      }
    }
    int unfilledCountOriginal = unfilledCount;

    cvtColor(inpaintCombine, outLAB, CV_BGR2Lab);

    vector<Mat> templates(rows*cols);
    vector<Mat> masks(rows*cols);

    for(int y = 0; y < rows; y++)
    {
      for(int x = 0; x < cols; x++)
      {
        if(windowInImage(x,y,outLAB,windowSize)) { templates.at(y*outLAB.cols + x) = getNeighbourhoodWindow(outLAB,Point2i(x,y),windowSize); }
      }
    }
    for(int y = 0; y < rows; y++)
    {
      for(int x = 0; x < cols; x++)
      {
         masks.at(y*cols + x) = getNeighbourhoodWindowMask(maskBinary,Point2i(x,y),windowSize);
      }
    }


    originalMask = maskBinary.clone();
    int epoch = 0;

    Mat outSmall;
    Mat strElement5 = getStructuringElement( MORPH_RECT,Size( windowSize*10, windowSize*10),Point(-1,-1));

    while(unfilledCount > 0)
    {
      dilate(border,borderBig,strElement2,Point(-1, -1), 1, 1, 1);
      erode(maskBinary,eroded,strElement,Point(-1, -1), 1, 1, 1);
      subtract(maskBinary,eroded,border);

      // if(epoch == 0)
      // {
      //   dilate(border,borderBig,strElement5,Point(-1, -1), 1, 1, 1);
      // }

      for(int y = 0; y < rows; y++)
      {
        for(int x = 0; x < cols; x++)
        {
          if( (int) border.at<uchar>(y,x,0) != 0 )
          {
            unfilledCount--;
            if(windowInImage(x,y,originalMask,windowSize))
            {
              Mat templ8 = getNeighbourhoodWindow(outLAB,Point2i(x,y),windowSize);
              Point2i newPos = findBestPixel(templ8,templates,gaussianMask,originalMask,masks,sampleCount,depthMap,rows,cols,Point2i(x,y),linearArray,windowSize);
              outLAB.at<int>(y,x,0) = outLAB.at<int>(newPos.y,newPos.x,0);

              if(newPos != Point2i(x,y))
              {
                linearArray.at(y*cols + x) = newPos;
              }
              float s = sampleCount.at<float>(newPos.y*cols+newPos.x);
              s+=2;
              sampleCount.at<float>(newPos.y*cols+newPos.x) =  s;
              cout << "current out pixel X" << endl; cout << x << endl;
              cout << "current out pixel Y" << endl; cout << y << endl;
              cout << "picked pixel X" << endl; cout << newPos.x << endl;
              cout << "picked pixel Y" << endl; cout << newPos.y << endl;
              cout << "unfilled count " << endl; cout << unfilledCount << endl;
              cout << "epoch " << endl; cout << epoch << endl;
            }
          }
         }
      }

      for(int y = 0; y < rows; y++)
      {
        for(int x = 0; x < cols; x++)
        {
          if(windowInImage(x,y,outLAB,windowSize) && (int) borderBig.at<uchar>(y,x,0) != 0) { templates.at(y*cols + x) = getNeighbourhoodWindow(outLAB,Point2i(x,y),windowSize); }
        }
      }
      for(int y = 0; y < rows; y++)
      {
        for(int x = 0; x < cols; x++)
        {
           if((int) borderBig.at<uchar>(y,x,0) != 0) { masks.at(y*cols + x) = getNeighbourhoodWindowMask(maskBinary,Point2i(x,y),windowSize); }
        }
      }

      cvtColor(outLAB, outBGR, CV_Lab2BGR);
      resize(outBGR,outSmall,outBGR.size()/2);
      imshow("progress",outSmall);
      maskBinary = eroded.clone();
      epoch++;
    }


    cvtColor(outLAB, outBGR, CV_Lab2BGR);
    return outBGR;
}
