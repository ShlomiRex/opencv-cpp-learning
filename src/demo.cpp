
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

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
    destroyAllWindows();
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
      imshow( "filter2D Demo", dst );
      ind++;
    }

    destroyAllWindows();
}

/*
This function shows pixel manipulation with 1 channel and 3 channels
*/
void rgb() {
    Mat mat;
	mat = imread("assets/lena.png", IMREAD_COLOR);
	vector<Mat> bgr_planes;
	split(mat, bgr_planes);
	imshow("Original", mat);
	imshow("Blue", bgr_planes[0]);
	imshow("Green", bgr_planes[1]);
	imshow("Red", bgr_planes[2]);
	
	Mat mat1, mat2, mat3;
	cvtColor(bgr_planes[0], mat1, COLOR_GRAY2BGR); //Blue
	cvtColor(bgr_planes[1], mat2, COLOR_GRAY2BGR); //Green
	cvtColor(bgr_planes[2], mat3, COLOR_GRAY2BGR); //Red

	cout << "BGR 3 channels type:" << mat1.type() << endl;

	//Because we converted GRAY to BGR, we have 3 channels of color instead of just 1. 
	for(int i = 0; i < mat1.rows; i++) {
		for(int j = 0; j < mat1.cols; j++) {
			Vec3b& colour = mat1.at<Vec3b>(j, i);
			//colour[0] = 0 //Blue
			colour[1] = 0; //Green
			colour[2] = 0; //Red
		}
	}

	for(int i = 0; i < mat2.rows; i++) {
		for(int j = 0; j < mat2.cols; j++) {
			Vec3b& colour = mat2.at<Vec3b>(j, i);
			colour[0] = 0; //Blue
			//colour[1] = 0; //Green
			colour[2] = 0; //Red
		}
	}

	for(int i = 0; i < mat3.rows; i++) {
		for(int j = 0; j < mat3.cols; j++) {
			Vec3b& colour = mat3.at<Vec3b>(j, i);
			colour[0] = 0; //Blue
			colour[1] = 0; //Green
			//colour[2] = 0; //Red
		}
	}
	
	imshow("Blue 2", mat1);
	imshow("Green 2", mat2);
	imshow("Red 2", mat3);

	waitKey();
    destroyAllWindows();
}

/*
This demo shows use of mask (white and black) by color
Tip: https://stackoverflow.com/questions/10469235/opencv-apply-mask-to-a-color-image
*/
void binary_trunc_mask() {
	Mat mat, gray;
	mat = imread("assets/lena.png", IMREAD_COLOR);
	cvtColor(mat, gray, COLOR_BGR2GRAY);
	imshow("Origional", mat);
	imshow("Gray", gray);

	Mat bin_mask, inv_bin_mask, trunc_mask, output, output2, output3;

	threshold(gray, bin_mask, 127, 255, THRESH_BINARY);
	mat.copyTo(output, bin_mask);
	imshow("Binary Mask", bin_mask);
	imshow("Binary mask applied", output);

	threshold(gray, inv_bin_mask, 127, 255, THRESH_BINARY_INV);
	mat.copyTo(output2, inv_bin_mask);
	imshow("Inverted Binary Mask", inv_bin_mask);
	imshow("Inverted binary mask applied", output2);

	threshold(gray, trunc_mask, 127, 255, THRESH_TRUNC);
	mat.copyTo(output3, trunc_mask);
	imshow("Trunc Mask", trunc_mask);
	imshow("Trunc mask applied", output3);

	waitKey();
	destroyAllWindows();
}

void color_mask() {
	Mat mat, gray;
	mat = imread("assets/lena.png", IMREAD_COLOR);
	cvtColor(mat, gray, COLOR_BGR2GRAY);
	imshow("Origional", mat);
	imshow("Gray", gray);

	vector<Mat> bgr_planes;
	split(mat, bgr_planes);

	Mat blue = bgr_planes[0], green =  bgr_planes[1], red = bgr_planes[2];
	imshow("Blue", blue);
	imshow("Green",green);
	imshow("Red", red);

	Mat bin_mask_blue, output, output2;

	threshold(blue, bin_mask_blue, 127, 255, THRESH_BINARY);
	mat.copyTo(output, bin_mask_blue);
	imshow("Binary Mask (blue)", bin_mask_blue);
	imshow("Binary mask (blue) applied", output);

	output.copyTo(output2);
	for(int i = 0; i < output2.rows; i++) {
		for(int j = 0; j < output2.cols; j++) {
			Vec3b& colour = output2.at<Vec3b>(j, i);
			//Image is BGR, so, change only channel #1 and #2 leave #0 as is
			colour[1] = 0;
			colour[2] = 0;

		}
	}
	imshow("Binary mask applied (blue), only blue channel", output2);

	waitKey();
	destroyAllWindows();
}
//Code: https://docs.opencv.org/master/d8/dbc/tutorial_histogram_calculation.html
void image_color_histogram()
{
	Mat src = imread("assets/lena.png", IMREAD_COLOR);

	vector<Mat> bgr_planes;
	split(src, bgr_planes);

	int histSize = 256;
	float range[] = { 0, 256 }; //the upper boundary is exclusive
	const float* histRange = { range };

	bool uniform = true, accumulate = false;
	Mat b_hist, g_hist, r_hist;
	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);
	int hist_w = 512, hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);
	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
	}
	imshow("Source image", src);
	imshow("calcHist Demo", histImage);
	waitKey();
}

void adaptive_threshold() {
	Mat mat, out, out2, out3;
    mat = imread("assets/lena.png", IMREAD_COLOR);
	cvtColor(mat, mat, COLOR_BGR2GRAY);

	threshold(mat, out3, 127, 255, THRESH_BINARY);
	imshow("Global threshold (v=127)", out3);

	adaptiveThreshold(mat, out, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 11, 2);
	imshow("Adaptive Threshold: Binary Mean", out);

	adaptiveThreshold(mat, out2, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);
	imshow("Adaptive Threshold: Binary Gaussian", out2);

	waitKey();
	destroyAllWindows();
}

void otsu_threshold() {
	Mat mat, gray, out, out2, out3, out4, out5;
    mat = imread("assets/lena.png", IMREAD_COLOR);
	cvtColor(mat, gray, COLOR_BGR2GRAY);

	//The value 127 is not used! Otsu algorithem finds optimal value for thresholding (middle of histogram peaks is optimal)
	threshold(gray, out, 127, 255, THRESH_OTSU);

	imshow("Original", mat);
	imshow("Gray", gray);
	imshow("Global threshold (v=127)", out);

	mat.copyTo(out2, out);

	imshow("Otsu threshold", out2);

	Size ksize = {5,5};
	GaussianBlur(gray, out3, ksize, 0);
	//Notice the sum of THRESH types
	threshold(out3, out4, 0, 255, THRESH_OTSU + THRESH_BINARY);
	imshow("Gaussian threshold", out4);
	mat.copyTo(out5, out4);
	imshow("Otsu + Binary + Gaussian",out5);


	waitKey();
	destroyAllWindows();
}

void denoising() {
	Mat mat, gray, out, out2, out3, out4, out5;
    mat = imread("assets/lena.png", IMREAD_COLOR);
	cvtColor(mat, gray, COLOR_BGR2GRAY);
	imshow("Original", mat);

	fastNlMeansDenoisingColored(mat, out, 12);
	imshow("Noise Reduction - Fast Mean", out);

	// Create a window
	namedWindow("My Window", 1);

	//Create trackbar to change brightness
	int iSliderValue = 25;
	createTrackbar("Noise reduction - h value", "My Window", &iSliderValue, 50);

	Mat dst;
	while (true)
	{
		//Change the brightness and contrast of the image (For more infomation http://opencv-srf.blogspot.com/2013/07/change-contrast-of-image-or-video.html)
		
		//mat.convertTo(dst, -1, 1, iSliderValue); 
		fastNlMeansDenoisingColored(mat, dst, iSliderValue);

		//show the brightness and contrast adjusted image
		imshow("My Window", dst);

		// Wait until user press some key for 50ms
		int iKey = waitKey(50);

		//if user press 'ESC' key
		if (iKey == 27)
		{
			break;
		}
	}

	waitKey();
	destroyAllWindows();
}

void sobel() {
	Mat mat, gray, out, out2, out3, out4, out5;
    mat = imread("assets/lena.png", IMREAD_COLOR);
	cvtColor(mat, gray, COLOR_BGR2GRAY);
	imshow("Original", mat);

	Sobel(gray, out, CV_64F, 1, 0, 5); //Sobel Dx 
	Sobel(gray, out2, CV_64F, 0, 1, 5); //Sobel Dy

	int dx_ksize = 9;
	int dy_ksize = 9;

	Sobel(gray, out, CV_64F, 1, 0, dx_ksize);
	Sobel(gray, out2, CV_64F, 0, 1, dy_ksize);

	imshow("Sobel Dx", out);
	imshow("Sobel Dy", out2);

	waitKey();
	destroyAllWindows();
}

/*
Captures video from camera, and thresholds the RED color of the video.
The good pixels are displayed in white while the bad pixels (which do not pass the thresholding) are black
Original code: https://www.opencv-srf.com/2010/09/object-detection-using-color-seperation.html
I added to the code the original image, after the mask applied (threshold mask) so we can see color instead of black and white elipse
*/
void red_color_threshold() {
	VideoCapture cap(0); //capture the video from web cam

    if ( !cap.isOpened() )  // if not success, exit program
    {
		cout << "Cannot open the web cam" << endl;
        exit(1);
    }

    namedWindow("Control"); //create a window called "Control"

	int iLowH = 0;
	int iHighH = 179;

	int iLowS = 0; 
	int iHighS = 255;

	int iLowV = 0;
	int iHighV = 255;

	//Create trackbars in "Control" window
	
	createTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
	createTrackbar("HighH", "Control", &iHighH, 179);

	createTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
	createTrackbar("HighS", "Control", &iHighS, 255);

	createTrackbar("LowV", "Control", &iLowV, 255); //Value (0 - 255)
	createTrackbar("HighV", "Control", &iHighV, 255);

    while (true)
    {
        Mat imgOriginal;

        bool bSuccess = cap.read(imgOriginal); // read a new frame from video

         if (!bSuccess) //if not success, break loop
        {
             cout << "Cannot read a frame from video stream" << endl;
             break;
        }

		Mat imgHSV;

		cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV
		
		Mat imgThresholded;

		inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image
			
		//morphological opening (remove small objects from the foreground)
		erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
		dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 

		//morphological closing (fill small holes in the foreground)
		dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 
		erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

		imshow("Thresholded Image - Mask", imgThresholded); //show the thresholded image
		imshow("Original", imgOriginal); //show the original image

		Mat imgAfterMask;
		imgOriginal.copyTo(imgAfterMask, imgThresholded);

		imshow("Thresholded Image", imgAfterMask); //I added this to the code to give it a little touch

		if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
		{
			cout << "esc key is pressed by user" << endl;
			break; 
		}
    }
}

int main()
{
	//sobel();
	//image_color_histogram();
	//binary_trunc_mask();
	//color_mask();
    //rgb();
    //blurring();
	//otsu_threshold();
    //laplican();
	//adaptive_threshold();
	//denoising();

	red_color_threshold();
    return 0;
}