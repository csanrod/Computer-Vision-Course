#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

enum slider {
    ORIGINAL = 0,
    FOURIER,
    HPF,
    LPF,
    AND
};

const String WINDOW_NAME = "Practise 2";
int option = ORIGINAL;
Mat image_input,
    image_output,
    complexImg,
    spectrum_original,
    HPFilter,
    LPFilter;

static void help(char ** argv) {
    cout << endl
        <<  "This program demonstrated the use of the discrete Fourier transform (DFT). " << endl
        <<  "The dft of an image is taken and it's power spectrum is displayed."  << endl << endl
        <<  "Usage:"                                                                      << endl
        << argv[0] << " [image_name -- default images/lenna.jpg]" << endl << endl;
}

// Compute the Discrete fourier transform
Mat computeDFT(Mat image) {
    // 1. Expand the image to an optimal size. 
    Mat padded;                      
    int m = getOptimalDFTSize( image.rows );
    int n = getOptimalDFTSize( image.cols ); // on the border add zero values
    copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols, BORDER_CONSTANT, Scalar::all(0));
    
    // 2. Make place for both the complex and the real values
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

    // 3. Make the Discrete Fourier Transform
    dft(complexI, complexI, DFT_COMPLEX_OUTPUT);      // this way the result may fit in the source matrix
    return complexI;
}

// 6. Crop and rearrange
void fftShift(Mat magI) {
    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
    
    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols/2;
    int cy = magI.rows/2;

    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
}


// Calculate dft spectrum
Mat spectrum(const Mat &complexI) {
    Mat complexImg = complexI.clone();
    // Shift quadrants
    fftShift(complexImg);

    // 4. Transform the real and complex values to magnitude
    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    Mat planes_spectrum[2];
    split(complexImg, planes_spectrum);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes_spectrum[0], planes_spectrum[1], planes_spectrum[0]);// planes[0] = magnitude
    Mat spectrum = planes_spectrum[0];

    // 5. Switch to a logarithmic scale
    spectrum += Scalar::all(1);                    // switch to logarithmic scale
    log(spectrum, spectrum);

    // 7. Normalize
    normalize(spectrum, spectrum, 0, 1, NORM_MINMAX); // Transform the matrix with float values into a
                                            // viewable image form (float between values 0 and 1).
    return spectrum;
}

void SliderCallback (int a, void * arg) 
{
    switch (option)
    {
        case ORIGINAL:
            cout << "(0) Original selected" << endl;
            imshow(WINDOW_NAME, image_input);
            break;

        case FOURIER:
            cout << "(1) Fourier selected" << endl;
            complexImg = computeDFT(image_input);
            spectrum_original = spectrum(complexImg);

            imshow(WINDOW_NAME, spectrum_original);
            break;

        case HPF:
            cout << "(2) HPF selected" << endl;

            imshow(WINDOW_NAME, HPFilter);
            break;

        case LPF:
            cout << "(3) LPF selected" << endl;
            
            imshow(WINDOW_NAME, LPFilter);
            break;

        case AND:
            cout << "(4) AND selected" << endl;
            break;
    }
}

void create_LPFilter ()
{
    LPFilter = Mat::zeros(512, 512, CV_8UC1);
    circle(LPFilter, Point(512/2,512/2), 50.0, Scalar(255, 255, 255), -1, 8);
}

void create_HPFilter ()
{
    HPFilter = Mat::zeros(512, 512, CV_8UC1);
    HPFilter.setTo(Scalar(255, 255, 255));
    circle(HPFilter, Point(512/2,512/2), 50.0, Scalar(0, 0, 0), -1, 8);
}

int main(int argc, char ** argv) {
    help(argv);
    const char* filename = argc >=2 ? argv[1] : "../../vision/images/lenna.jpg";
    image_input = imread( samples::findFile( filename ), IMREAD_GRAYSCALE);
    if( image_input.empty()){
        cout << "Error opening image" << endl;
        return EXIT_FAILURE;
    }

    create_LPFilter();
    create_HPFilter();

    // Resize lenna
    resize(image_input, image_input, Size(512, 512));

    // Slider
    namedWindow(WINDOW_NAME, WINDOW_AUTOSIZE );
    imshow(WINDOW_NAME, image_input);

    createTrackbar( "Element:\n  0:  Original  \n  1:  Fourier  \n  2:  HPF  \n  3:  LPF  \n  4:  AND", 
                    WINDOW_NAME,           
                    &option, 4,           
                    SliderCallback );

    SliderCallback(0, 0);

    // // Compute the Discrete fourier transform
    // Mat complexImg = computeDFT(I);
    // Mat filter = complexImg.clone();

    // // Get the spectrum
    // Mat spectrum_original = spectrum(complexImg);

    // // 6. Crop and rearrange
    // fftShift(complexImg);
    // //doSomethingWithTheSpectrum();   
    // fftShift(complexImg); // rearrage quadrants

    // // Get the spectrum
    // Mat spectrum_filter = spectrum(complexImg);

    // // 8. Results
    // imshow("Input Image"       , I   );    // Show the result
    // imshow("Spectrum original", spectrum_original);
    // imshow("Spectrum filter", spectrum_filter);

    // // 9. Calculating the idft
    // Mat inverseTransform;
    // idft(complexImg, inverseTransform, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
    // normalize(inverseTransform, inverseTransform, 0, 1, NORM_MINMAX);
    // imshow("Reconstructed", inverseTransform);

    waitKey(0);
    return EXIT_SUCCESS;
}

