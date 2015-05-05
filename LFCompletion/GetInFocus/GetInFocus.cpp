#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgproc/types_c.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/photo/photo.hpp"
#include <iostream>
#include "../DepthMap.cpp"
#include "../FillHoleDirected.cpp"
#include "../FocalStackPropagation.cpp"
#include "../pixelStruct.h"

using namespace cv;
using namespace std;
Mat infocus, infocusS,inpainted;
Mat out, outS;
Mat mask, maskS;
Mat depthMap,depthMapS,depthMapF,depthMapFS,depthMapBlurred,depthMapFBlurred;
vector<Mat> imgs;
vector<Mat> laplacians;
vector<int> segmentIndicies;
vector<Mat> segments;
vector<Mat> gauss;
vector<Mat> windows;

//------------------------------------------------------------
string retrieveString( char* buf, int max ) {

    size_t len = 0;
    while( (len < max) && (buf[ len ] != '\0') ) {
        len++;
    }

    return string( buf, len );

}

int main(int argc, char** argv )
{
    char* fn = argv[1];
    string filename = retrieveString(fn,100);

    bool imageLoad = true;
    int imNum = 0;


    while(imageLoad)
    {
      stringstream ss;
      string thisFilename;
      imgs.resize(imNum+1);
      ss << filename << imNum << ".jpg";
      thisFilename = ss.str();
      imgs.at(imNum) = imread(thisFilename,3);
      if(imgs.at(imNum).empty())
      {
        imageLoad = false;
        imgs.resize(imNum);
      }
      else
      {
        imNum++;
      }
    }

    //laplacians = laplacianFocalStack(imgs);
    gauss = differenceOfGaussianFocalStack(imgs);

    string name;
    name = filename+"_depthMapEdited.jpg";
    depthMap  = imread(name,1);
    if(depthMap.empty())
    {
      depthMap = createDepthMap(gauss);
      name = filename+"_depthMap.jpg";
      imwrite(name,depthMap);
    }

    infocus = createInFocusImage(depthMap,imgs);

    name = filename+"_infocus.jpg";
    imwrite(name,infocus);

    cout << "finished, all files are written" << endl;

    return 0;
}
