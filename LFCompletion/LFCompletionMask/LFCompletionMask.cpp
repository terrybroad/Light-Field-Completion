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
    cout << "images loaded" << endl;

    laplacians = laplacianFocalStack(imgs);
    gauss = differenceOfGaussianFocalStack(imgs);

    depthMap = createDepthMap(gauss);
    GaussianBlur(depthMap,depthMapBlurred,Size(15,15),0);
    cout << "depth map created" << endl;
    infocus = createInFocusImage(depthMap,imgs);
    //infocus = createInFocusImageInterpolate(depthMapBlurred,imgs);
    cout << "infocus image created" << endl;




    Size size = infocus.size();
    Size smallSize = size/2;
    resize(depthMap, depthMapS,smallSize);
    resize(infocus, infocusS,smallSize);
    Mat originalS = infocusS.clone();
    mask = Mat::zeros(size, CV_8U);
    maskS = Mat::zeros(smallSize, CV_8U);

    imshow("image", infocusS);

    bool notFilled = true;

    string name;
    name = filename+"_mask.jpg";

    mask = imread(name,0);

    if(mask.empty())
    {
      return 0;
    }

    cout << "mask read" << endl;
    notFilled = false;

    depthMapF= fillDepthMapDirected(depthMap,mask);
    GaussianBlur(depthMapF,depthMapFBlurred,Size(15,15),0);


    inpaint(infocus, mask, inpainted, 3, INPAINT_TELEA);

    cout<< "image preliminary inpainted" << endl;


    out = fillImageDirected(inpainted,depthMapF,depthMapBlurred,mask,11,1500);

    resize(out,outS,smallSize);
    imshow("in focus filled",outS);


    //out = infocus;
    cout<< "image completed - next to propagate through the focal stack" << endl;

    vector<Mat> outImages;
    outImages = propogateFocalStack(imgs, gauss, out, mask, depthMapF, depthMapBlurred);


    for(int i = 0; i < outImages.size(); i++)
    {
      stringstream ss;
      string outputName;
      ss << filename << "_completed" << i << ".jpg";
      outputName = ss.str();
      cout << "writing " << outputName << endl;
      //imshow(outputName, outImages.at(i));
      imwrite(outputName, outImages.at(i));
    }
    notFilled = false;

    resize(depthMapF,depthMapFS, smallSize);
    name = filename+"_filled.jpg";
    imwrite(name,out);
    name = filename+"_depthMapFilled.jpg";
    imwrite(name,depthMapF);
    name = filename+"_infocus.jpg";
    imwrite(name,infocus);

    cout << "finished, all files are written" << endl;

    return 0;

}
