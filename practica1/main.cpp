#include <iostream>
#include <vector>

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

// Function Header
void ChangeColorSpace (int a, void * arg);

// Constants
enum color_spaces {
    RGB = 0,
    CMY,
    HSI,
    HSV,
    HSV_OPENCV
};

const String WINDOW_NAME = "Color Space",
             RELATIVE_PATH = "../../vision/images/RGB.jpg";

// Global variables
int color_space = RGB,
    max_color_space = 4;

Mat src,
    dst;



int main( int argc, char** argv ) {
    cout << "INIT\n";

    src = imread( RELATIVE_PATH, IMREAD_COLOR );
    if ( src.empty() ) {
        cout << "Could not open or find the image!\n" << endl;
        cout << "Usage: " << argv[0] << " <Input image>" << endl;
        return -1;
    }

    // Show image
    namedWindow(WINDOW_NAME, WINDOW_AUTOSIZE );
    imshow(WINDOW_NAME, src);

    // Create Erosion Trackbar 
    createTrackbar( "Element:\n  0:  RGB  \n  1:  CMY  \n  2:  HSI  \n  3:  HSV  \n  4:  HSV  OpenCV", 
                    WINDOW_NAME,           
                    &color_space, max_color_space,           
                    ChangeColorSpace );

    ChangeColorSpace (0, 0);

    cout << "END\n";

    // Wait to press a key
    waitKey(0);

    return 0;
}
void ChangeColorSpace (int a, void * arg) 
{

    vector <Mat> BGR_vector,
                 CMY_vector,
                 channels;

    

    switch (color_space) 
    {
        case RGB:
            cout << "RGB selected\n";
            // Vector de Mat BGR
            split(src, BGR_vector);

            channels.push_back(BGR_vector[0]);
            channels.push_back(BGR_vector[1]);
            channels.push_back(BGR_vector[2]);

            merge(channels, dst);
            imshow(WINDOW_NAME, dst);             
            break;

        case CMY:
            cout << "CMY selected\n";
            // Vector de Mat BGR
            split(src, CMY_vector);

            // Transformamos BGR a CMY. uchar --> [0,255]
            for (int i = 0; i < src.rows; i++) {
                for (int j = 0; j < src.cols; j++) {
                    CMY_vector[0].at<uchar>(i, j) = (uchar)(255 - (uint)CMY_vector[0].at<uchar>(i, j));
                    CMY_vector[1].at<uchar>(i, j) = (uchar)(255 - (uint)CMY_vector[1].at<uchar>(i, j));
                    CMY_vector[2].at<uchar>(i, j) = (uchar)(255 - (uint)CMY_vector[2].at<uchar>(i, j));
                }
            }

            channels.push_back(CMY_vector[0]);
            channels.push_back(CMY_vector[1]);
            channels.push_back(CMY_vector[2]);

            merge(channels, dst);
            imshow(WINDOW_NAME, dst);
            break;

        case HSI:
            cout << "HSI selected\n";
            break;

        case HSV:
            cout << "HSV selected\n";
            break;

        case HSV_OPENCV:
            cout << "HSV OpenCV selected\n";
            break;
        
        default:
            cout << "Something unexpected happened in ChangeColorSpace\n";
    }
}