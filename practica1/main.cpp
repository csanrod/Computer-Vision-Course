#include <iostream>

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

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

    // Wait to press a key
    waitKey(0);

    return 0;
}