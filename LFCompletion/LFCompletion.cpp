#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgproc/types_c.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/photo/photo.hpp"
#include <iostream>
#include "DepthMap.cpp"
#include "FillHole.cpp"
#include "FocalStackPropagation.cpp"

using namespace cv;
using namespace std;
Mat infocus, infocusS, out, outS, mask, maskS, depthMap,depthMapS,depthMapF,depthMapFS;
vector<Mat> imgs;
vector<Mat> laplacians;
vector<int> segmentIndicies;
vector<Mat> segments;
vector<Mat> gauss;
vector<Mat> windows;

Point prevPt(-1,-1);



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
        line( maskS, prevPt, pt, Scalar::all(255), 5, 200, 0 );
        line( infocusS, prevPt, pt, Scalar(0,255,0), 5, 200, 0 );
        prevPt = pt;
        imshow("image", infocusS);
    }
}


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

    laplacians = laplacianFocalStack(imgs);
    gauss = differenceOfGaussianFocalStack(imgs);

    depthMap = createDepthMap(gauss);
    infocus = createInFocusImage(depthMap,imgs);






    Size size = infocus.size();
    Size smallSize = size/2;
    resize(depthMap, depthMapS,smallSize);
    resize(infocus, infocusS,smallSize);
    Mat originalS = infocusS.clone();
    mask = Mat::zeros(size, CV_8U);
    maskS = Mat::zeros(smallSize, CV_8U);

    imshow("image", infocusS);
    setMouseCallback( "image", onMouse, 0 );

    bool notFilled = true;

    while(notFilled)
    {
        char c = (char)waitKey();
        //imshow("image", infocusS);
        if( c == 27 )
            break;

        if( c == 'r' )
        {
            maskS = Scalar::all(0);
            infocusS = originalS.clone();
            imshow("image", infocusS);
        }

        if( c == 'i' || c == ' ' )
        {
          resize(maskS,mask,size,INTER_CUBIC);
          notFilled = false;
          depthMapF = fillDepthMap(depthMap,mask);


          out = fillImage(infocus,depthMapF,mask);
          //out = infocus;
          cout<< "image completed - next to propagate through the focal stack" << endl;

          vector<Mat> outImages;
          outImages = propogateFocalStack(imgs, gauss, out, mask, depthMapF);


          for(int i = 0; i < outImages.size(); i++)
          {
            stringstream ss;
            string outputName;
            ss << filename << "_completed" << i << ".jpg";
            outputName = ss.str();
            //imshow(outputName, outImages.at(i));
            imwrite(outputName, outImages.at(i));
          }
          notFilled = false;
        }

    }

    resize(out,outS,smallSize);
    resize(depthMapF,depthMapFS, smallSize);
    string name = filename+"_filled.jpg";
    imwrite(name,out);
    name = filename+"_depthMapFilled.jpg";
    imwrite(name,depthMapF);
    name = filename+"_maskArea.jpg";
    imwrite(name,infocusS);
    while(1)
    {
    imshow("filled",outS);
    imshow("depthMap",depthMapFS);
    }

}
