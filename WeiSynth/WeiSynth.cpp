#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <algorithm>    // std::shuffle
#include <array>        // std::array
#include <random>       // std::default_random_engine
#include <chrono>       // std::chrono::system_clock

using namespace std;
using namespace cv;

Mat src; Mat output; Mat downSrc; Mat upOut;
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

    namedWindow( window_name1, WINDOW_AUTOSIZE );


    int windowSize = 4;
    int winLength = (windowSize*2) + 1;
    RNG rng( 0xFFFFFFFF );

    int height = 50;
    int width = 50;

    output.create(floor(height),floor(width),16);

/*    downSrc.create(src.rows/2,src.cols/2,16);
    Mat downSrc2; downSrc2.create(src.rows/4,src.cols/4,16);
    Mat downSrc3; downSrc3.create(src.rows/8,src.cols/8,16);
    pyrDown(src,downSrc,downSrc.size());
    pyrDown(downSrc,downSrc2,downSrc2.size());
    pyrDown(downSrc2,downSrc3,downSrc3.size()); */

    imshow("Unprocessed Image",src);

    for(int i = 0; i < output.rows; i++)
    {
      for (int j = 0; j < output.cols; j++)
      {
          output.at<int>(i,j,0) = rng.uniform(0,0xFFFFFFFF);
      }
    }

    vector<Mat> templates(src.rows*src.cols);

    for(int y = 0; y < src.rows; y++)
    {
      for(int x = 0; x < src.cols; x++)
      {
        if(windowInImage(x,y,src,windowSize))
        {
          templates.at(y*src.cols + x) = getNeighbourhoodWindow(src,Point2i(x,y),windowSize);
        }
      }
    }

    for(int y = 0; y < output.rows; y++)
    {
      for(int x = 0; x < output.cols; x++)
      {
        if(windowInImage(x,y,output,windowSize))
        //if(1)
        {
          Mat templ8 = getNeighbourhoodWindow(output,Point2i(x,y),windowSize);

          //output.at<int>(y,x,0) = findBestPixel(templ8,src,windowSize);
          output.at<int>(y,x,0) = findBestPixelFast(templ8,templates,src.rows,src.cols,windowSize);

          cout << "current out pixel Y" << endl; cout << y << endl;
          cout << "current out pixel X" << endl; cout << x << endl;
        }
      }
    }


    namedWindow( window_name2, WINDOW_AUTOSIZE );
    imshow("Processed Image",output);

    imwrite( "output Image.tiff", output );

    waitKey();
    return 0;
}
