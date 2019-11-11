
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

void laplican() {

    Mat mat1, mat2, mat3;
    mat1 = imread("assets/lena.png", IMREAD_COLOR);
    
    cout << "Origional Mat properties" << endl;
    cout << "Type: " << mat1.type() << endl;
    cout << "Depth: " << mat1.depth() << endl;
    cout << "Channels: " << mat1.channels() << endl << endl;

    cvtColor(mat1, mat2, COLOR_BGR2GRAY);

    cout << "Gray Mat properties" << endl;
    cout << "Type: " << mat2.type() << endl;
    cout << "Depth: " << mat2.depth() << endl;
    cout << "Channels: " << mat2.channels() << endl << endl;

    Laplacian(mat2, mat3, 0);

    cout << "Laplican Mat properties" << endl;
    cout << "Type: " << mat2.type() << endl;
    cout << "Depth: " << mat2.depth() << endl;
    cout << "Channels: " << mat2.channels() << endl << endl;

    imshow("Original", mat1);
    imshow("Gray", mat2);
    imshow("Laplican", mat3);
    
    waitKey();
}

void blurring()
{
  /// Declare variables
  Mat src, dst;

  Mat kernel;
  Point anchor;
  double delta;
  int ddepth;
  int kernel_size;
  char* window_name = "filter2D Demo";

  int c;

  /// Load an image
  src = imread("assets/lena.png", IMREAD_COLOR);

  /// Create window
  //namedWindow( window_name, CV_WINDOW_AUTOSIZE );

  /// Initialize arguments for the filter
  anchor = Point( -1, -1 );
  delta = 0;
  ddepth = -1;

  /// Loop - Will filter the image with different kernel sizes each 0.5 seconds
  int ind = 0;
  while( true )
    {
      c = waitKey(500);
      /// Press 'ESC' to exit the program
      if( (char)c == 27 )
        { break; }

      /// Update kernel size for a normalized box filter
      kernel_size = 3 + 2*( ind%5 );
      kernel = Mat::ones( kernel_size, kernel_size, CV_32F )/ (float)(kernel_size*kernel_size);

        cout << "Kernel size: " << kernel_size << endl;

      /// Apply filter
      filter2D(src, dst, ddepth , kernel, anchor, delta, BORDER_DEFAULT );
      imshow( window_name, dst );
      ind++;
    }
    destroyWindow(window_name);
}

int main()
{
    blurring();
    laplican();
    return 0;
}