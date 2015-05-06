#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgproc/types_c.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/photo/photo.hpp"
#include <iostream>

using namespace cv;
using namespace std;
Mat srcBGR, srcLAB, texLAB, outLAB, outBGR,inpaintMask,inpainted;
Point prevPt(-1,-1);

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
      if( i != windowSize && j != windowSize)
      {
        //if((int) mask1.at<uchar>(i,j) == 0 )//&& (int) mask2.at<uchar>(i,j) == 0)
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
            }
             else // penalise non-image elements
            {
              dist += (c.val[k])^2;
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
}


//----------------------------------------------------------------------------------------------------------------------------------------
const Point2i findBestPixel(const Mat &templ8, const vector<Mat> &templates, const Mat &gaussian,const Mat &mask,const vector<Mat> &masks, const Mat &sampleCount, int rows, int cols, RNG &rng,const Point2i pos, int windowSize)
{
  Point2i bestPixel;
  Point2i ptemp;
  double bestValue = 100;
  //double dist = 0;

  int n = 1;
  int count = 0;
  bool yes = false;

  vector<Point2i> candidates;
  int numCan = 0;


  while(count < 1000)
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
              double dist = getDist(templ8,templates.at(ptemp.y*cols + ptemp.x),masks.at(pos.y*cols + pos.x),masks.at(ptemp.y*cols + ptemp.x),gaussian,windowSize);
              dist *= sampleCount.at<float>(pos.y*cols+pos.x);
              if( sampleCount.at<float>(pos.y*cols+pos.x) < 2) { count++;}

              if(dist < bestValue)
              {
                //numCan++;
                //candidates.resize(numCan);
                //candidates.at(numCan-1) = ptemp;
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


//  cout << "n " << endl; cout << n << endl;
//  cout << "bestPixel y" << endl; cout << bestPixel.y << endl;
//  cout << "bestPixel x" << endl; cout << bestPixel.x << endl;
//  cout << "dist" << endl; cout << bestValue << endl;

  return bestPixel;//return candidates.at(rng.uniform(0,numCan-1));
}



static void onMouse( int event, int x, int y, int flags, void* )
{
    if( event == EVENT_LBUTTONUP || !(flags & EVENT_FLAG_LBUTTON) )
        prevPt = Point(-1,-1);
    else if( event == EVENT_LBUTTONDOWN )
        prevPt = Point(floor(x/2),floor(y/2));
    else if( event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON) )
    {
        Point pt(x,y);
        if( prevPt.x < 0 )
            prevPt = pt;
        line( inpaintMask, prevPt, pt, Scalar::all(255), 5, 200, 0 );
        line( srcBGR, prevPt, pt, Scalar::all(255), 5, 200, 0 );
      //  line( depthMap, prevPt, pt, Scalar::all(255), 5, 200, 0 );
        prevPt = pt;
        imshow("image", srcBGR);
    }
}

int main(int argc, char** argv )
{
    srcBGR = imread( argv[1], 3 );
    inpaintMask = Mat::zeros(srcBGR.size(), CV_8U);


    cvtColor(srcBGR, srcLAB, CV_BGR2Lab);

    int windowSize = 11;
    int winLength = (windowSize*2) + 1;

    Mat depth = imread( "depthmap2.jpg", 1);
    namedWindow( "image", CV_WINDOW_AUTOSIZE );
    /namedWindow( "output", CV_WINDOW_AUTOSIZE );

    imshow("image", srcBGR);
    setMouseCallback( "image", onMouse, 0 );

    RNG rng;
    //rng.range(-25,25);

    for(;;)
    {
        char c = (char)waitKey();

        if( c == 27 )
            break;

        if( c == 'r' )
        {
            inpaintMask = Scalar::all(0);
            imshow("image", srcBGR);
        }

        if( c == 'i' || c == ' ' )
        {
            Mat inpainted;
            inpaint(srcBGR, inpaintMask, inpainted, 3, INPAINT_TELEA);
          //  imshow("inpainted image", inpainted);
            cvtColor(inpainted, outLAB, CV_BGR2Lab);

            Mat maskBinary,originalMask,eroded, border, strElement,sampleCount,borderBig,strElement2;
            maskBinary = Mat::zeros(srcBGR.size(), CV_8U);
            threshold( inpaintMask, maskBinary, 127,255,0 );
            strElement = getStructuringElement( MORPH_RECT,Size( 3, 3),Point(1,1));
            strElement = getStructuringElement( MORPH_RECT,Size( windowSize, windowSize),Point(-1,-1));
            eroded = Mat::zeros(maskBinary.size(), CV_8U);
            border = Mat::zeros(maskBinary.size(), CV_8U);
            borderBig = Mat::zeros(maskBinary.size(), CV_8U);
            sampleCount = Mat::ones(maskBinary.size(), CV_32F);

            double kernalsize = (double)windowSize / 6.0;
            kernalsize = sqrt(kernalsize);
            Mat tmpGaussian = getGaussianKernel(windowSize * 2 + 1, kernalsize);
            Mat gaussianMask = tmpGaussian * tmpGaussian.t();



            int unfilledCount = 0;

            for(int y = 0; y < maskBinary.rows; y++)
            {
              for(int x = 0; x < maskBinary.cols; x++)
              {
                if((int) maskBinary.at<uchar>(y,x,0) != 0){ unfilledCount++; }
              }
            }
            int unfilledCountOriginal = unfilledCount;


            Mat blurred = outLAB.clone();
            GaussianBlur(outLAB,blurred,Size(75,75),0,0);

            for(int y = 0; y < outLAB.rows; y++)
            {
              for(int x = 0; x < outLAB.cols; x++)
              {
                if((int) maskBinary.at<uchar>(y,x,0) != 0)
                {
                  Vec3b a;
                  a = blurred.at<Vec3b>(y,x);
                  for(int i = 0; i < 3; i++)
                  {
                    int num = a.val[i] + rng.uniform(-17,17);
                    a.val[i] = num;
                  }
                  outLAB.at<Vec3b>(y,x) = a;
                }
              }
            }


            vector<Mat> templates(outLAB.rows*outLAB.cols);
            vector<Mat> masks(maskBinary.rows*maskBinary.cols);

            for(int y = 0; y < outLAB.rows; y++)
            {
              for(int x = 0; x < outLAB.cols; x++)
              {
                if(windowInImage(x,y,outLAB,windowSize)) { templates.at(y*outLAB.cols + x) = getNeighbourhoodWindow(outLAB,Point2i(x,y),windowSize); }
              }
            }
            for(int y = 0; y < maskBinary.rows; y++)
            {
              for(int x = 0; x < maskBinary.cols; x++)
              {
                 masks.at(y*maskBinary.cols + x) = getNeighbourhoodWindowMask(maskBinary,Point2i(x,y),windowSize);
              }
            }

            originalMask = maskBinary.clone();
            int epoch = 0;

            while(unfilledCount > 0)
            {
              dilate(border,borderBig,strElement2,Point(-1, -1), 1, 1, 1);
              erode(maskBinary,eroded,strElement,Point(-1, -1), 1, 1, 1);
              subtract(maskBinary,eroded,border);


              for(int y = 0; y < outLAB.rows; y++)
              {
                for(int x = 0; x < outLAB.cols; x++)
                {
                  if( (int) border.at<uchar>(y,x,0) != 0)
                  {
                    unfilledCount--;
                    Mat templ8 = getNeighbourhoodWindow(outLAB,Point2i(x,y),windowSize);
                    Point2i newPos = findBestPixel(templ8,templates,gaussianMask,originalMask,masks,sampleCount,outLAB.rows,outLAB.cols,rng,Point2i(x,y),windowSize);
                    outLAB.at<int>(y,x,0) = outLAB.at<int>(newPos.y,newPos.x,0);

                    float s = sampleCount.at<float>(newPos.y*outLAB.cols+newPos.x);
                    s+=2;
                    sampleCount.at<float>(newPos.y*outLAB.cols+newPos.x) =  s;
                    cout << "current out pixel X" << endl; cout << x << endl;
                    cout << "current out pixel Y" << endl; cout << y << endl;
                    cout << "unfilled count " << endl; cout << unfilledCount << endl;
                    cout << "epoch " << endl; cout << epoch << endl;
                  }
                 }
              }

              for(int y = 0; y < outLAB.rows; y++)
              {
                for(int x = 0; x < outLAB.cols; x++)
                {
                  if(windowInImage(x,y,outLAB,windowSize) && (int) borderBig.at<uchar>(y,x,0) != 0) { templates.at(y*outLAB.cols + x) = getNeighbourhoodWindow(outLAB,Point2i(x,y),windowSize); }
                }
              }
              for(int y = 0; y < maskBinary.rows; y++)
              {
                for(int x = 0; x < maskBinary.cols; x++)
                {
                   if((int) borderBig.at<uchar>(y,x,0) != 0) { masks.at(y*maskBinary.cols + x) = getNeighbourhoodWindowMask(maskBinary,Point2i(x,y),windowSize); }
                }
              }

              maskBinary = eroded.clone();
              epoch++;
          }



          for(int y = 0; y < sampleCount.rows; y++)
          {
            for(int x = 0; x < sampleCount.cols; x++)
            {
               sampleCount.at<float>(y*sampleCount.cols + x) /= 255;
            }
          }

        //  imshow("used",sampleCount);

            cvtColor(outLAB, outBGR, CV_Lab2BGR);
            imshow("output",outBGR);

        }
        imwrite("filled.jpg",outBGR);
        imwrite("masked.jpg",srcBGR);
    }










}
