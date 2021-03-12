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
int min_shrink_value = 0,
    max_shrink_value = 30;

Mat image_input,                // Imagen original
    complexImg,                 // Transformada discreta de fourier               
    LPFilter,                   // Transformada filtrada paso bajo       
    LPinverseTransform;         // Transformada inversa paso bajo

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
void minCallback (int a, void * arg) 
{
    printf("min: %d\n", min_shrink_value);
}

void maxCallback (int a, void * arg) 
{
    printf("max: %d\n", max_shrink_value);
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
                    &min_shrink_value, 255,           
                    minCallback );
    minCallback(0, 0);

    createTrackbar( "Shrink value max: 0-255", 
                    WINDOW_NAME,           
                    &max_shrink_value, 255,           
                    maxCallback );
    maxCallback(0, 0);

    // End
    waitKey(0);
    return EXIT_SUCCESS;
}