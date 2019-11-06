
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>

int main()
{
    cv::Mat mat1, mat2, mat3;
    mat1 = cv::imread("assets/lena.png", cv::IMREAD_COLOR);
    cv::cvtColor(mat1, mat2, cv::COLOR_BGR2GRAY);

    //cv::cornerHarris(mat2, mat3, 2,3,0.04);
    cv::cornerHarris(mat2, mat3, 4, 3, 0.4);
    cv::imshow("", mat3);
    mat1.convertTo(mat1, -1, 1, 50);
    cv::imshow("a", mat1);
    mat1.convertTo(mat1, -1, 1, 150);
    cv::imshow("b", mat1);
    cv::waitKey(0);
    return 0;
}