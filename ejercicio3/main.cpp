/* ------------------------------------------------------------ */
// Programador: Cristian Sánchez Rodríguez
//
// Asignatura: Visión Artificial (Ejercicio 2)
//
// Descripción: En esta práctica se verá aplicado de forma  
//				práctica transformaciones del dominio y espaciales
//              de la imagen "lenna"    
//
//				Se aplica lo siguiente: 
//				    - Espectro de frecuencias
//				    - Filtro paso alto
//				    - Filtro paso bajo
//                  - Operación AND sobre operación umbral
/* ------------------------------------------------------------ */

// -- Includes -- //
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

// -- Constantes y variables globales -- //
const String WINDOW_NAME = "Practise 3";
int min_shrink_slider = 0,
    max_shrink_slider = 30,
    min_value = 0,
    max_value = 30;

Mat image_input,                // Imagen original
    complexImg,                 // Transformada discreta de fourier               
    LPFilter,                   // Transformada filtrada paso bajo       
    LPinverseTransform,
    histogram,
    image_output;         // Transformada inversa paso bajo

// -- Métodos fourier -- //
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

// -- Métodos propios -- //
uint Minimum (uint a, uint b)
{
    double result;

    if (a < b)
        result = a;
    else
        result = b;

    return result;
}

uint Maximum (uint a, uint b)
{
    double result;

    if (a > b)
        result = a;
    else
        result = b;

    return result;
}

// Filtro paso bajo
void aply_LPFilter ()
{
    complexImg = computeDFT(image_input);

    fftShift(complexImg);

    LPFilter = complexImg.clone();
    LPFilter.setTo(Scalar(0, 0, 0));
    circle(LPFilter, Point(LPFilter.rows/2,LPFilter.cols/2), 50.0, Scalar(255, 255, 255), -1, 8);
    mulSpectrums(complexImg, LPFilter, complexImg, 0);

    fftShift(complexImg);

    //LPFilter = spectrum(complexImg);
    
    idft(complexImg, LPinverseTransform, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
    normalize(LPinverseTransform, LPinverseTransform, 0, 1, NORM_MINMAX);
}

// Callbacks sliders
void cross_control()
{
    if (max_shrink_slider < min_shrink_slider) {
        min_value = 50;
        max_value = 150;
    } else {
        min_value = min_shrink_slider;
        max_value = max_shrink_slider;
    }
}

void sliderCallback (int a, void * arg) 
{
    cross_control();
    aply_LPFilter();

    double C_max = max_value, 
           C_min = min_value,
           r_max = 0, 
           r_min = 255,
           r_k = 0;

    // Guardo en histogram la imagen LP con valores uchar de 0 a 255
    histogram = image_input.clone();
    for (int row = 0; row < LPinverseTransform.rows; row++){
        for (int col = 0; col < LPinverseTransform.cols; col++){
            histogram.at<uchar>(row,col) = (uchar)(LPinverseTransform.at<float>(row,col)*255);
            r_max = Maximum(r_max, histogram.at<uchar>(row,col));
            r_min = Minimum(r_min, histogram.at<uchar>(row,col));
        }      
    }

    // Contracción histograma
    cout << "C_max: "  << C_max << endl;
    cout << "C_min: "  << C_min << endl;
    cout << "r_max: "  << r_max << endl;
    cout << "r_min: "  << r_min << endl << endl;

    for (int row = 0; row < LPinverseTransform.rows; row++){
        for (int col = 0; col < LPinverseTransform.cols; col++){
            r_k = histogram.at<uchar>(row,col);
            histogram.at<uchar>(row,col) = (uchar)(((C_max - C_min)/(r_max - r_min))*(r_k - r_min) + C_min);
        }      
    }


    // Pintar histograma
    int histSize = 256;
    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange = { range };
    bool uniform = true, accumulate = false;

    Mat hist, input_hist;
    calcHist( &histogram, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &image_input, 1, 0, Mat(), input_hist, 1, &histSize, &histRange, uniform, accumulate );
    

    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );

    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    normalize(input_hist, input_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

    for( int i = 1; i < histSize; i++ ) {
        line(histImage, Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ),
             Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
             Scalar( 255, 0, 0), 2, 8, 0);

        line(histImage, Point( bin_w*(i-1), hist_h - cvRound(input_hist.at<float>(i-1)) ),
             Point( bin_w*(i), hist_h - cvRound(input_hist.at<float>(i)) ),
             Scalar( 0, 0, 255), 2, 8, 0  );
    }

    // Resta pixel a pixel
    image_output = image_input.clone();
    for (int row = 0; row < image_input.rows; row++){
        for (int col = 0; col < image_input.cols; col++)
            image_output.at<uchar>(row, col) = histogram.at<uchar>(row, col) - image_input.at<uchar>(row, col);
    }

    imshow("Resta pixel", image_output);
    imshow("Shrink", histImage);
    imshow(WINDOW_NAME, histogram);
    //cout << "-- Me ejecuto crack! --" << endl;
}

// -- MAIN -- //
int main(int argc, char ** argv) {
    const char* filename = argc >=2 ? argv[1] : "./lenna.jpg";

    // Input en escala de grises para aplicar las operaciones
    image_input = imread( samples::findFile( filename ), IMREAD_GRAYSCALE);
    if( image_input.empty()){
        cout << "Error opening image" << endl;
        return EXIT_FAILURE;
    }

    // Resize lenna
    resize(image_input, image_input, Size(512, 512));

    // Slider del menu
    namedWindow(WINDOW_NAME, WINDOW_AUTOSIZE );
    imshow(WINDOW_NAME, image_input);

    createTrackbar( "Shrink value min: 0-255", 
                    WINDOW_NAME,           
                    &min_shrink_slider, 255,           
                    sliderCallback );

    createTrackbar( "Shrink value max: 0-255", 
                    WINDOW_NAME,           
                    &max_shrink_slider, 255,           
                    sliderCallback );

    sliderCallback(0, 0);

    // End
    waitKey(0);
    return EXIT_SUCCESS;
}