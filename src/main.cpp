#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>

using namespace std;
using namespace cv;
int main()
{
	Mat img = imread("assets/lena.png");
	Mat gray_img, hsv_img, hls_img, luv_img, yuv_img, lab_img;
	cvtColor(img, gray_img, COLOR_BGR2GRAY);
	cvtColor(img, hsv_img, COLOR_BGR2HSV);
	cvtColor(img, hls_img, COLOR_BGR2HLS);
	cvtColor(img, luv_img, COLOR_BGR2Luv);
	cvtColor(img, yuv_img, COLOR_BGR2YUV);
	cvtColor(img, lab_img, COLOR_BGR2Lab);
	imshow("Gray image", gray_img);
	imshow("HSV Image", hsv_img);
	imshow("HLS Image", hls_img);
	imshow("Luv Image", luv_img);
	imshow("YUV Image", yuv_img);
	imshow("Lab Image", lab_img);

	VideoCapture cap(0);
	if (!cap.open(0))
		return 0;
	Mat a;
	while (1) {
		Mat frame;
		cap >> frame;
		if (frame.empty()) break; // end of video stream
		imshow("this is you, smile! :)", frame);
		if (waitKey(10) == 27) break; // stop capturing by pressing ESC 
	}
	waitKey();
	return 0;
}