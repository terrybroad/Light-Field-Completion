#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <algorithm>    // std::shuffle
#include <array>        // std::array
#include <random>       // std::default_random_engine
#include <chrono>       // std::chrono::system_clock

using namespace std;
using namespace cv;

Mat src; Mat output;
char window_name1[] = "Unprocessed Image";
char window_name2[] = "Processed Image";

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
double getDist(const Mat &templ8, const Mat &templ9, int windowSize)
{
  double dist = 0;

  vector<Mat> channels1(3);
  vector<Mat> channels2(3);

  split(templ8,channels1);
  split(templ9,channels2);

  int count;

  for(int i = 0; i < windowSize+1; i++)
  {
    for(int j = 0; j < templ8.cols; j++)
    {
      count++;
      if( (i < windowSize) || (i == windowSize && j < windowSize))
      {
        for(int k = 0; k < 3; k++)
        {
          dist += abs((channels1.at(k).at<int>(i,j) - channels2.at(k).at<int>(i,j)))^2;
        }
      }
    }
  }

  return sqrt(dist)/count;
}


//----------------------------------------------------------------------------------------------------------------------------------------
int findBestPixel(const Mat &templ8, const Mat &img,int windowSize)
{
  Point2i bestPixel;
  double bestValue = -1;

  for(int y = 0; y < img.rows; y++)
  {
    for(int x = 0; x < img.cols; x++)
    {
      if(windowInImage(x,y,img,windowSize))
      {
        Mat templ9 = getNeighbourhoodWindow(src,Point2i(x,y),windowSize);

        double dist = getDist(templ8,templ9,windowSize);

        if(dist < bestValue || bestValue < 0)
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

//----------------------------------------------------------------------------------------------------------------------------------------
int findBestPixelFast(const Mat &templ8, const vector<Mat> &templates,int rows, int cols,int windowSize)
{
  Point2i bestPixel;
  double bestValue = -1;

  for(int y = windowSize; y < rows-windowSize; y++)
  {
    for(int x = windowSize; x < cols-windowSize; x++)
    {
        double dist = getDist(templ8,templates.at(y*cols + x),windowSize);

        if(dist < bestValue || bestValue < 0)
        {
          bestValue = dist;
          bestPixel.x = x;
          bestPixel.y = y;
        }
    }
  }
  cout << "bestPixel x" << endl; cout << bestPixel.x << endl;
  cout << "bestPixel y" << endl; cout << bestPixel.y << endl;
  cout << "dist" << endl; cout << bestValue << endl;

  return templates.at(bestPixel.y*cols+bestPixel.x).at<int>(windowSize,windowSize,0);
}


//----------------------------------------------------------------------------------------------------------------------------------------
//                    MAIN FUNCTION
//----------------------------------------------------------------------------------------------------------------------------------------
int main( int argc, char** argv )
{
    /// Load the source image
    src = imread( argv[1], 3 );

    //Mat lab;
    //src = (1.0/255.0) * src;
    //cvtColor(src, lab, CV_RGB2Lab);

    namedWindow( window_name1, WINDOW_AUTOSIZE );
    imshow("Unprocessed Image",src);

    output = src.clone();
    int rows = src.rows;
    int cols = src.cols;

    int windowSize = 4;
    int winLength = (windowSize*2) + 1;

    cout << rows << endl;
    cout << cols << endl;

    RNG rng( 0xFFFFFFFF );

    // -------------------------------------------------------
    //            RANDOMIZE IMAGE
    // -------------------------------------------------------

    vector<Point2i> linearArray, newLinearArray;

    linearArray.resize(rows*cols);
    newLinearArray.resize(rows*cols);

    for(int y = 0; y < rows; y++)
    {
      for(int x = 0; x < cols; x++)
      {
        linearArray.at(y*cols + x).x = x;
        linearArray.at(y*cols + x).y = y;
      }
    }

    random_shuffle(linearArray.begin(), linearArray.end());

    for(int y = 0; y < rows; y++)
    {
      for(int x = 0; x < cols; x++)
      {
        Point2i p = linearArray.at(y*cols + x);
        output.at<int>(y,x,0) = src.at<int>(p.y, p.x,0);
      }
    }

    for(int i = 0; i < 1; i++)
    {
      for(int y = 0; y < rows; y++)
      {
        for(int x = 0; x < cols; x++)
        {
          Mat templ8 = getNeighbourhoodWindow(output,Point2i(x,y),windowSize);
          vector<Point2i> candidates;
          vector<double> dist;
          candidates.resize(winLength*winLength);
          dist.resize(winLength*winLength);
          int count = 0;
          int bestValue = -1;
          double lowestValue = 0;

          for(int k = 0; k < winLength/2; k++)
          {
            for(int l = 0; l < winLength; l++)
            {
              Point2i relPos, refPos, rrPos;

              relPos = Point2i(x-(windowSize+l),y-(windowSize+k));

              if(windowInImage(relPos.x,relPos.y,output,windowSize))
              {
                refPos = linearArray.at(relPos.y*cols + relPos.x);

                if(windowInImage(refPos.x,refPos.y,output,windowSize))
                {
                  rrPos = Point2i(refPos.x + (l-windowSize),refPos.y+(k-windowSize));

                  if(windowInImage(rrPos.x,rrPos.y,output,windowSize))
                  {
                    candidates.at(count) = Point2i(rrPos.x,rrPos.y);
                    Mat templ9 = getNeighbourhoodWindow(src,rrPos,windowSize);
                    dist.at(count) = getDist(templ8,templ9,windowSize);

                    if(dist.at(count) < lowestValue || count == 0)
                    {
                      bestValue = count;
                      lowestValue = dist.at(count);
                    }
                    count++;
                  }
                }
              }
            }
          }
          cout << bestValue << endl;
          cout << "pos" << endl;
          cout << x << endl;
          cout << y << endl;
          if(bestValue >= 0)
          {
            output.at<int>(y,x,0) = src.at<int>(candidates.at(bestValue).y,candidates.at(bestValue).x,0);
            newLinearArray.at(y*cols + x) = candidates.at(bestValue);
          }
        }
      }

      linearArray = newLinearArray;
      i++;
    }

    //Mat templ8 = getNeighbourhoodWindow(output,Point2i(40,30),2);

    namedWindow( window_name2, WINDOW_AUTOSIZE );
    imshow("Processed Image",output);

    imwrite( "Randomised_Image.tiff", output );

    waitKey();
    return 0;
}
