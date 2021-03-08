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
enum slider {
    ORIGINAL = 0,
    FOURIER,
    HPF,
    LPF,
    AND
};

const String WINDOW_NAME = "Practise 2";
int option = ORIGINAL;

Mat image_input,                // Imagen original
    complexImg,                 // Transformada discreta de fourier
    HPFilter,                   // Transformada filtrada paso alto
    LPFilter,                   // Transformada filtrada paso bajo
    HPinverseTransform,         // Imagen filtrada (inversa) paso alto
    LPinverseTransform,         // Imagen filtrada (inversa) paso bajo
    highP,                      // Umbral paso alto p
    lowP,                       // Umbral paso bajo p
    AND_output;                 // Imagen con AND aplicado de los umbrales

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
// Filtro paso alto
void aply_HPFilter ()
{
    complexImg = computeDFT(image_input);

    fftShift(complexImg);

    HPFilter = complexImg.clone();
    HPFilter.setTo(Scalar(255, 255, 255));
    circle(HPFilter, Point(HPFilter.rows/2,HPFilter.cols/2), 50.0, Scalar(0, 0, 0), -1, 8);
    mulSpectrums(complexImg, HPFilter, complexImg, 0);

    fftShift(complexImg);

    HPFilter = spectrum(complexImg);

    idft(complexImg, HPinverseTransform, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
    normalize(HPinverseTransform, HPinverseTransform, 0, 1, NORM_MINMAX);
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

    LPFilter = spectrum(complexImg);
    
    idft(complexImg, LPinverseTransform, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
    normalize(LPinverseTransform, LPinverseTransform, 0, 1, NORM_MINMAX);
}

// And sobre umbral pixel a pixel
void aply_AND ()
{
    // Umbral p = 0.4 a HPinverseTransform
    aply_HPFilter();
    highP = Mat::zeros(512, 512, CV_8UC1);
    float p = 0.4;
    for (int row = 0; row < HPinverseTransform.rows; row++){
        for (int col = 0; col < HPinverseTransform.cols; col++){
            float px_value = HPinverseTransform.at<float>(row,col); 
            if (px_value > p)
                highP.at<uchar>(row,col) = (uint)255;
            else
                highP.at<uchar>(row,col) = (uint)0;
        }
    }

    // Umbral p = 0.6 a LPinverseTransform
    aply_LPFilter();
    lowP = Mat::zeros(512, 512, CV_8UC1);
    p = 0.6;
    for (int row = 0; row < LPinverseTransform.rows; row++){
        for (int col = 0; col < LPinverseTransform.cols; col++){
            float px_value = LPinverseTransform.at<float>(row,col);        
            if (px_value > p)
                lowP.at<uchar>(row,col) = (uint)255;
            else
                lowP.at<uchar>(row,col) = (uint)0;
        }
    }

    // AND
    bitwise_and(highP, lowP, AND_output);
}

// Callback del menu
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
            imshow(WINDOW_NAME, spectrum(complexImg));
            break;

        case HPF:
            cout << "(2) HPF selected" << endl;
            aply_HPFilter();
            imshow(WINDOW_NAME, HPinverseTransform); //HPinverseTransform son FLOATS ENTRE 0 Y 1
            break;

        case LPF:
            cout << "(3) LPF selected" << endl;
            aply_LPFilter();
            imshow(WINDOW_NAME, LPinverseTransform);
            break;

        case AND:
            cout << "(4) AND selected" << endl;
            aply_AND();
            imshow(WINDOW_NAME, AND_output);
            break;
    }
}

// -- MAIN -- //
int main(int argc, char ** argv) {
    const char* filename = argc >=2 ? argv[1] : "../../vision/images/lenna.jpg";

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
    createTrackbar( "Element:\n  0:  Original  \n  1:  Fourier  \n  2:  HP Filter  \n  3:  LP Filter  \n  4:  AND", 
                    WINDOW_NAME,           
                    &option, 4,           
                    SliderCallback );
    SliderCallback(0, 0);

    // End
    waitKey(0);
    return EXIT_SUCCESS;
}