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

int main( int argc, char** argv )
{
    /// Load the source image
    src = imread( argv[1], 3 );

    namedWindow( window_name1, WINDOW_AUTOSIZE );
    imshow("Unprocessed Image",src);

    output = src.clone();
    int rows = src.rows;
    int cols = src.cols;

    cout << rows << endl;
    cout << cols << endl;

    RNG rng( 0xFFFFFFFF );

    vector<int> linearArray;

    linearArray.resize(rows*cols);

    for(int i = 0; i < rows; i++)
    {
      for(int j = 0; j < cols; j++)
      {
        linearArray.at(j*rows + i) = j*rows + i;
      }
    }

    random_shuffle(linearArray.begin(), linearArray.end());

    for(int i = 0; i < rows; i++)
    {
      for(int j = 0; j < cols; j++)
      {
        int index = linearArray.at(j*rows + i);
        output.at<int>(i,j) = src.at<int>(index % rows, floor(index/cols));
      }
    }

    namedWindow( window_name2, WINDOW_AUTOSIZE );
    imshow("Processed Image",output);

    imwrite( "Randomised_Image.tiff", output );
    imwrite("ShouldBeEmmaAgain?.tiff", src);

    waitKey();
    return 0;
}
