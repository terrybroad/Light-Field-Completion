#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/photo/photo.hpp"

#include <iostream>

using namespace cv;
using namespace std;

static void help()
{
    cout << "\nCool inpainging demo. Inpainting repairs damage to images by floodfilling the damage \n"
            << "with surrounding image areas.\n"
            "Using OpenCV version %s\n" << CV_VERSION << "\n"
    "Usage:\n"
        "./inpaint [image_name -- Default ../data/fruits.jpg]\n" << endl;

    cout << "Hot keys: \n"
        "\tESC - quit the program\n"
        "\tr - restore the original image\n"
        "\ti or SPACE - run inpainting algorithm\n"
        "\t\t(before running it, paint something on the image)\n" << endl;
}

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

  int count = 0;

  for(int i = 0; i < windowSize+1; i++)
  {
    for(int j = 0; j < templ8.cols; j++)
    {
      if( (i < windowSize) || (i == windowSize && j < windowSize))
      {
        for(int k = 0; k < 3; k++)
        {
          count++;
          dist += abs((channels1.at(k).at<int>(i,j) - channels2.at(k).at<int>(i,j)))^2;
        }
      }
    }
  }

  return sqrt(dist/count);
}


//----------------------------------------------------------------------------------------------------------------------------------------
const Point2i findBestPixelGrow(const Mat &templ8, const vector<Mat> &templates, const Mat &mask,int rows, int cols,const vector<Mat> &masks, const Point2i pos,int windowSize)
{
  Point2i bestPixel;
  Point2i ptemp;
  double bestValue = 100;
  //double dist = 0;

  int n = 1;

  bool yes = false;

/*
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

          if(windowInImage(ptemp.x,ptemp.y,mask,windowSize))
          {
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
          }
        }
      }
    }

    n++;
  }

*/

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


  cout << "n " << endl; cout << n << endl;
  cout << "bestPixel y" << endl; cout << bestPixel.y << endl;
  cout << "bestPixel x" << endl; cout << bestPixel.x << endl;
  cout << "dist" << endl; cout << bestValue << endl;

  return bestPixel;
}


Mat img, inpaintMask;
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
        line( inpaintMask, prevPt, pt, Scalar::all(255), 5, 8, 0 );
        line( img, prevPt, pt, Scalar::all(255), 5, 8, 0 );
        prevPt = pt;
        imshow("image", img);
    }
}


int main( int argc, char** argv )
{
    char* filename = argc >= 2 ? argv[1] : (char*)"../data/fruits.jpg";
    //Mat img0 = imread(filename, 3);
    Mat img0  = imread( argv[1], 3 );
    if(img0.empty())
    {
        cout << "Couldn't open the image " << filename << ". Usage: inpaint <image_name>\n" << endl;
        return 0;
    }

    help();

    namedWindow( "image", WINDOW_NORMAL );
    namedWindow( "inpainted image", WINDOW_NORMAL );


    img = img0.clone();
    inpaintMask = Mat::zeros(img.size(), CV_8U);

    int windowSize = 4;
    int winLength = (windowSize*2) + 1;

    imshow("image", img);
    setMouseCallback( "image", onMouse, 0 );

    for(;;)
    {
        char c = (char)waitKey();

        if( c == 27 )
            break;

        if( c == 'r' )
        {
            inpaintMask = Scalar::all(0);
            img0.copyTo(img);
            imshow("image", img);
        }

        if( c == 'i' || c == ' ' )
        {
            Mat inpainted;
            inpaint(img, inpaintMask, inpainted, 3, INPAINT_NS);
            imshow("inpainted image", inpainted);

            vector<Mat> templates(img.rows*img.cols);
            vector<Mat> masks(inpaintMask.rows*inpaintMask.cols);

            for(int y = 0; y < img.rows; y++)
            {
              for(int x = 0; x < img.cols; x++)
              {
                if(windowInImage(x,y,img,windowSize))
                {
                  //templates.at(y*img.cols + x)
                  templates.at(y*img.cols + x) = getNeighbourhoodWindow(img,Point2i(x,y),windowSize);
                  masks.at(y*inpaintMask.cols + x) = getNeighbourhoodWindow(inpaintMask,Point2i(x,y),windowSize);
                }
              }
            }

            Mat texFilled;
            texFilled = inpainted.clone();

            for(int y = 0; y < texFilled.rows; y++)
            {
              for(int x = 0; x < texFilled.cols; x++)
              {
                if(inpaintMask.at<int>(y,x,0) != 0)
                {
                  Point2i pos(x,y);
                  Mat templ8 = getNeighbourhoodWindow(img,pos,windowSize);
                  Point2i newPos = findBestPixelGrow(templ8,templates,inpaintMask,img.rows,img.cols,masks,pos,windowSize);
                  texFilled.at<int>(pos.y,pos.x,0) = img.at<int>(newPos.y,newPos.x,0);
                  inpainted.at<int>(newPos.y,newPos.x,0) = 0;

                  cout << "current out pixel Y" << endl; cout << y << endl;
                  cout << "current out pixel X" << endl; cout << x << endl;
                  cout << "COLOUR" << endl; cout << texFilled.at<int>(y,x,0) << endl;
                }
              }
            }

            imshow("inpainted image", inpainted);
            imshow("texfilled image", texFilled);

        }
    }

    return 0;
}
