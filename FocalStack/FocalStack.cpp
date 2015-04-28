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


int main(int argc, char** argv )
{

    int imgNum = 11;
    vector<Mat> imgs;
    vector<Mat> laps;
    vector<Mat> imgsG;
    vector<Mat> smoothed;
    vector<Mat> gauss;
    vector<Mat> diffs;
    vector<Mat> gaussDiffs;

    imgs.resize(imgNum);
    imgsG.resize(imgNum);
    laps.resize(imgNum);
    smoothed.resize(imgNum);
    gauss.resize(imgNum);
    diffs.resize(imgNum);
    gaussDiffs.resize(imgNum);

    for(int i = 0; i < imgNum; i++)
    {
      char filename[50];

          //if(i < 10) { sprintf( filename, "wasp-stk_0%d.jpg", i ); } else { sprintf( filename, "wasp-stk_%d.jpg", i ); }
          sprintf( filename, "stack4/reordered%d.jpg", i );
  				imgs.at(i) = imread( filename, 1);

          /*if (!img)
  					{
  						printf("Error: Image not found.\n");
  						return 2; //error : not found image
  					} */

          cvtColor(imgs.at(i),imgsG.at(i), CV_BGR2GRAY);
          Laplacian(imgsG.at(i),laps.at(i),0,5);
          GaussianBlur(laps.at(i),smoothed.at(i),Size(55,55),10);

          GaussianBlur(imgsG.at(i),gauss.at(i),Size(11,11),11);
          diffs.at(i) =  imgsG.at(i) - gauss.at(i);
          GaussianBlur(diffs.at(i),gaussDiffs.at(i),Size(101,101),11);

          //  imshow( filename, smoothed.at(i) );                   // Show our image inside it.
            imshow(filename,gaussDiffs.at(i));
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


  Mat diff = imgsG.at(4) - imgsG.at(8);

  Mat diff2 = diff-diffs.at(0);
  //Mat lapDiff;

  //Laplacian(diff,lapDiff,0,5);

  for(;;)
  {
//  imshow("im1",diffs.at(0));
  imshow("diff",diff);
//  imshow("diffs2",diff2);
  //imshow("lapDiff",lapDiff);
  }




/*
    imshow("depthMap",depthMap);
    imshow("blar",inFocus);

    imwrite("stack4/depthMap1.jpg",depthMap);
    imwrite("stack4/infocus1.jpg",inFocus);


        imshow("depthMap2",depthMap2);
        imshow("blar2",inFocus2);

        imwrite("stack4/depthMap2.jpg",depthMap2);
        imwrite("stack4/infocus2.jpg",inFocus2);

*/

}
