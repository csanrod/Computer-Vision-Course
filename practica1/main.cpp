#include <iostream>

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

// Global variables
int color_space = 0,
    max_color_space = 4;

// Function Header
void ChangeColorSpace (int option, void * arg);

int main( int argc, char** argv ) {
    cout << "INIT\n";
    const string RELATIVE_PATH = "../../vision/images/RGB.jpg";

    Mat src = imread( RELATIVE_PATH, IMREAD_COLOR );
    if ( src.empty() ) {
        cout << "Could not open or find the image!\n" << endl;
        cout << "Usage: " << argv[0] << " <Input image>" << endl;
        return -1;
    }

    // Show image
    namedWindow( "Practise 1", WINDOW_AUTOSIZE );
    imshow("Practise 1", src);

    // Create Erosion Trackbar 
    createTrackbar( "Element:\n  0:  RGB  \n  1:  CMY  \n  2:  HSI  \n  3:  HSV  \n  4:  HSV  OpenCV", "Practise 1",           
                    &color_space, max_color_space,           
                    ChangeColorSpace );

    // Wait to press a key
    waitKey(0);

    return 0;
}
void ChangeColorSpace (int option, void * arg) 
{
    cout << "Stuff inside ChangeColorSpace\n";
    cout << option << endl;
    cout << arg << endl;
}