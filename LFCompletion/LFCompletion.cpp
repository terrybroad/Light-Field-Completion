#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgproc/types_c.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/photo/photo.hpp"
#include <iostream>
#include "DepthMap.cpp"
#include "FillHole.cpp"

using namespace cv;
using namespace std;
Mat infocus, infocusS, out, outS, mask, maskS, depthMap,depthMapS,depthMapF,depthMapFS;
vector<Mat> imgs;
vector<Mat> inputs;

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
      imgs.resize(imNum+1);
      stringstream ss;
      ss << filename << imNum << ".jpg";
      string thisFilename = ss.str();
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

    inputs = createDepthMap(imgs);
    depthMap = inputs.at(0);
    infocus = inputs.at(1);

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
        }

    }

    resize(out,outS,smallSize);
    resize(depthMapF,depthMapFS, smallSize);
    imwrite("filled.jpg",out);
    imwrite("depthmap.jpg",depthMapF);
    imwrite("infocus.jpg",infocusS);
    while(1)
    {
    imshow("filled",outS);
    imshow("depthMap",depthMapFS);
    }

}
