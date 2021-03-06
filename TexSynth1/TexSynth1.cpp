#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <algorithm>    // std::shuffle
#include <array>        // std::array
#include <random>       // std::default_random_engine
#include <chrono>       // std::chrono::system_clock
#include <string.h>

using namespace std;
using namespace cv;

Mat src; Mat output;
char window_name1[] = "Unprocessed Image";
char window_name2[] = "Processed Image";

//----------------------------------------------------------------------------------------------------------------------------------------
bool inImage(int x, int y, const Mat &img, int windowSize)
{
  return (x - windowSize >= 0 && x + windowSize < img.cols && y - windowSize >= 0 && y + windowSize < img.rows);
}

//----------------------------------------------------------------------------------------------------------------------------------------
Mat getNeighbourhoodWindow(const Mat &img, Point2i pt, int windowSize)
{
  Mat output = Mat(windowSize * 2 + 1, windowSize * 2 + 1, 16);

  for(int i = 0; i < output.rows; i++)
  {
    for(int j = 0; j < output.cols; j++)
    {
      if(inImage( pt.x + i, pt.y + j, img, windowSize))
      {
        output.at<int>(i, j) = img.at<int>(pt.y - windowSize + i, pt.x - windowSize + j);
      }
    }
  }

  return output;
}

//----------------------------------------------------------------------------------------------------------------------------------------
double getDist(const Mat &templ8, const Mat &templ9)
{
  double dist = 0;

  vector<Mat> channels1(3);
  vector<Mat> channels2(3);

  split(templ8,channels1);
  split(templ9,channels2);

  for(int i = 0; i < templ8.rows; i++)
  {
    for(int j = 0; j < templ8.cols; j++)
    {
      if(( i != floor(templ8.rows / 2) && j != floor(templ8.cols/2) && j < floor(templ8.cols/2) ) || (i != floor(templ8.rows / 2) && j < ceil(templ8.cols/2)))
      {
        for(int k = 0; k < 3; k++)
        {
          dist += abs(channels1.at(k).at<int>(i,j) - channels2.at(k).at<int>(i,j));
        }
      }
    }
  }

  return dist;
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

    int windowSize = 5;
    int winLength = (windowSize*2) + 1;

    cout << rows << endl;
    cout << cols << endl;

    RNG rng( 0xFFFFFFFF );

    // -------------------------------------------------------
    //            RANDOMIZE IMAGE
    // -------------------------------------------------------

    vector<Point2i> linearArray;

    linearArray.resize(rows*cols);

    for(int i = 0; i < rows; i++)
    {
      for(int j = 0; j < cols; j++)
      {
        linearArray.at(j*rows + i).x = i;
        linearArray.at(j*rows + i).y = j;
      }
    }

    //random_shuffle(linearArray.begin(), linearArray.end());

    for(int i = 0; i < rows; i++)
    {
      for(int j = 0; j < cols; j++)
      {
        int indexX = linearArray.at(j*rows + i).x;
        int indexY = linearArray.at(j*rows + i).y;
        output.at<int>(i,j,0) = src.at<int>(indexX, indexY,0);
      }
    }


    // -------------------------------------------------------
    //                Check Pixels
    // -------------------------------------------------------
    for(int i = 0; i < output.rows/2 ; i++)
    {
      for(int j = 0; j < output.cols/2 ; j++)
      {
          Mat templ8 = getNeighbourhoodWindow(output,Point2i(i,j),windowSize);

          vector<Point2i> candidates;
          vector<double> dist;
          int count = 0;
          int bestValue = 0;
          double lowestValue = 0;

          candidates.resize(winLength*winLength);
          dist.resize(winLength*winLength);

          cout << i  <<  endl;
          cout << j << endl;


          for(int k = 0; k < winLength; k++)
          {
            for(int l = 0; l < winLength; l++)
            {
              Point2i relPos, refPos, rrPos;


              relPos = Point2i(i-(windowSize+k),j-(windowSize+l));

              if( relPos.x > windowSize && relPos.y > windowSize && relPos.x < rows - windowSize && relPos.y < cols - windowSize)
              {
                  refPos = linearArray.at(relPos.y*rows + relPos.x);

                  if(inImage(refPos.x,refPos.y,output,windowSize) )
                  {

                    rrPos = Point2i(refPos.x + (k-windowSize),refPos.y+(l-windowSize));
                    //rrPos = Point2i(refPos.x ,refPos.y);
                    //cout << "thrills" << endl;

                    if(inImage(rrPos.x,rrPos.y,output,windowSize))
                    {
                      candidates.at(count) = Point2i(rrPos.x,rrPos.y);

                      Mat templ9 = getNeighbourhoodWindow(src,rrPos,windowSize);

                      //cout << "pills" << endl;
                      dist.at(count) = getDist(templ8,templ9);

                        if(dist.at(count) < lowestValue || count == 0)
                        {
                          bestValue = count;
                          lowestValue = dist.at(count);
                          //cout << "lager" << endl;
                        }

                      count++;
                    }
                   }
                }


             }

            }

            cout << bestValue << endl;
            cout << dist.at(bestValue) << endl;

        output.at<int>(i,j,0) = src.at<int>(candidates.at(bestValue).x,candidates.at(bestValue).y,0);


      }
    }



    //Mat templ8 = getNeighbourhoodWindow(output,Point2i(40,30),2);

    namedWindow( window_name2, WINDOW_AUTOSIZE );
    imshow("Processed Image",output);

    imwrite( "Randomised_Image.tiff", output );
    imwrite("ShouldBeEmmaAgain?.tiff", src);

    waitKey();
    return 0;
}
