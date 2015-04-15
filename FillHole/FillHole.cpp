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
        output.at<float>(y, x, 0) = img.at<int>(pt.y - windowSize + y, pt.x - windowSize + x, 0);
      }
    }
  }

  return output;
}
//----------------------------------------------------------------------------------------------------------------------------------------
double getDist(const Mat &templ8, const Mat &templ9, int windowSize)
{
  double dist = 0;

  vector<Mat> channels1(3);
  vector<Mat> channels2(3);

  split(templ8,channels1);
  split(templ9,channels2);

  int count = 0;

/*
  for(int i = 0; i < (windowSize*2) + 1; i++)
  {
    for(int j = 0; j < (windowSize*2) + 1; j++)
    {
      if( i != windowSize && j != windowSize)
      {
          for(int k = 0; k < 3; k++)
          {
            count++;
            dist += ( (channels1.at(k).at<int>(i,j) - channels2.at(k).at<int>(i,j)) ^2);
          }
      }
    }
  }
*/

  Mat diff;

  absdiff(templ8, templ9, diff);

  vector<Mat> channelsDiff(3);

  split(diff,channelsDiff);

  for(int i = 0; i < (windowSize*2) + 1; i++)
  {
    for(int j = 0; j < (windowSize*2) + 1; j++)
    {
      if( i != windowSize && j != windowSize)
      {

          for(int k = 0; k < 3; k++)
          {
            count++;
            dist +=  (double) abs(channelsDiff.at(k).at<int>(i,j) );
          }

      //  count++;
      //   dist +=  abs(channelsDiff.at(1).at<int>(i,j));
      //dist +=  abs(diff.at<int>(i,j,0));
      }
    }
  }

  if(dist == 0)
  {
    cout << "LUCKY NUMBER ZERO" << endl;
  }

  cout << channelsDiff.at(1).at<int>(1,1)  << endl; cout << "BLAR" << endl;
  cout << channelsDiff.at(1).at<int>(1,1,0)  << endl; cout << "BLARBLAR" << endl;
  cout << diff.at<int>(1,1,0)  << endl; cout << "BLARBLARBLAR" << endl;
  return sqrt(dist/count);
}

//----------------------------------------------------------------------------------------------------------------------------------------
int findBestP(const Mat &templ8, const Mat &img,const Point2i pos, int windowSize)
{
  Point2i bestPixel;
  double bestValue = 100000000000;

  for(int y = windowSize; y < img.rows - windowSize; y++)
  {
    for(int x = windowSize; x < img.cols - windowSize; x++)
    {
      if(windowInImage(x,y,img,windowSize))
      {
        Mat templ9 = getNeighbourhoodWindow(img,Point2i(x,y),windowSize);

        double dist = getDist(templ8,templ9,windowSize);

if(x == pos.x && y == pos.y)
{
  cout << "DISTANCE WITH ME" << endl; cout << dist << endl;
//  dist = getDist(templ8,templ8,windowSize);
//  cout << "DISTANCE WITH ME" << endl; cout << dist << endl;

}
        if(dist < bestValue)
        {
          bestValue = dist;
          bestPixel.x = x;
          bestPixel.y = y;
        }


     }
    }
  }
  cout << "bestPixel x" << endl; cout << bestPixel.x << endl;
  cout << "bestPixel y" << endl; cout << bestPixel.y << endl;
  cout << "dist" << endl; cout << bestValue << endl;

  return img.at<int>(bestPixel.y,bestPixel.x,0);
}

/*
//----------------------------------------------------------------------------------------------------------------------------------------
const Point2i findBestPixel(const Mat &templ8, const vector<Mat> &templates, int rows, int cols, int windowSize)
{
  Point2i bestPixel;
  Point2i ptemp;
  double bestValue = 100;
  //double dist = 0;

  int n = 1;

  bool yes = false;


  while(n < 100)
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

        //  if(windowInImage(ptemp.x,ptemp.y,mask,windowSize))
        //  {
            //if(mask.at<int>(ptemp.y,ptemp.x,0) == 0)
            //{
              double dist = getDist(templ8,templates.at(ptemp.y*cols + ptemp.x),windowSize);

              if(dist < bestValue)
              {
                bestValue = dist;
                bestPixel.x = ptemp.x;
                bestPixel.y = ptemp.y;
              }
              if(bestValue < 0.05 )
              {
                yes = true;
              }
      //    }
        }
      }
    }

    n++;
  }


  cout << "n " << endl; cout << n << endl;
  cout << "bestPixel y" << endl; cout << bestPixel.y << endl;
  cout << "bestPixel x" << endl; cout << bestPixel.x << endl;
  cout << "dist" << endl; cout << bestValue << endl;

  return bestPixel;
}
*/


static void onMouse( int event, int x, int y, int flags, void* )
{
    if( event == EVENT_LBUTTONUP || !(flags & EVENT_FLAG_LBUTTON) )
        prevPt = Point(-1,-1);
    else if( event == EVENT_LBUTTONDOWN )
        prevPt = Point(x,y);
    else if( event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON) )
    {
        Point pt(x,y);
        if( prevPt.x < 0 )
            prevPt = pt;
        line( inpaintMask, prevPt, pt, Scalar::all(255), 5, 8, 0 );
        line( srcBGR, prevPt, pt, Scalar::all(255), 5, 8, 0 );
        prevPt = pt;
        imshow("image", srcBGR);
    }
}

int main(int argc, char** argv )
{
    srcBGR = imread( argv[1], 3 );
    inpaintMask = Mat::zeros(srcBGR.size(), CV_8U);
    cvtColor(srcBGR, srcLAB, CV_BGR2Lab);

  //  srcLAB.create(srcBGR.size(),16);
//    outLAB.create(srcBGR.size(),16);
//    outBGR.create(srcBGR.size(),16);

    cout << srcLAB.type() << endl;

    int windowSize = 3;
    int winLength = (windowSize*2) + 1;

    imshow("image", srcBGR);
    setMouseCallback( "image", onMouse, 0 );


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
            inpaint(srcBGR, inpaintMask, inpainted, 3, INPAINT_NS);
            imshow("inpainted image", inpainted);
            //cvtColor(inpainted, outLAB, CV_BGR2Lab);
/*
            vector<Mat> templates(srcLAB.rows*srcLAB.cols);

            for(int y = 0; y < srcLAB.rows; y++)
            {
              for(int x = 0; x < srcLAB.cols; x++)
              {
                if(windowInImage(x,y,srcLAB,windowSize))
                {
                  templates.at(y*srcLAB.cols + x) = getNeighbourhoodWindow(srcLAB,Point2i(x,y),windowSize);
                }
              }
            }
*/
            outLAB = inpainted.clone();

            for(int y = 0; y < outLAB.rows; y++)
            {
              for(int x = 0; x < outLAB.cols; x++)
              {
                if(inpaintMask.at<int>(y,x,0) != 0)
                {
                  Mat templ8 = getNeighbourhoodWindow(outLAB,Point2i(x,y),windowSize);

                  //Point2i newPos = findBestP(templ8,srcLAB,windowSize);
                  //outLAB.at<int>(y,x,0) = srcLAB.at<int>(y,x,0);
                  outLAB.at<int>(y,x,0) = findBestP(templ8,srcBGR,Point2i(x,y),windowSize);

                  imshow("progress",outLAB);
                  cout << "current out pixel X" << endl; cout << x << endl;
                  cout << "current out pixel Y" << endl; cout << y << endl;
                  //cout << "COLOUR" << endl; cout << texFilled.at<int>(y,x,0) << endl;
                }
              }
            }





            //cvtColor(outLAB, outBGR, CV_Lab2BGR);
            outBGR = outLAB.clone();
            imshow("output",outBGR);
        }

    }










}
