/* ------------------------------------------------------------ */
// Programador: Cristian Sánchez Rodríguez
//
// Asignatura: Visión Artificial (Ejercicio 4)
//
// Descripción: Este ejercicio consiste en aplicar diversos modos  
//				para obtener contornos/bordes de la imagen damas.jpg 
//
//				Modos aplicados: 
//				    - Hough
//				    - Contours
//				    - Centroids
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
enum cte 
{
    HOUGH = 0,
    CONTOURS,
    CENTROID
};

const String WINDOW_NAME = "Practise 4";

Mat image_input,
    image_output;

int mode,
    canny_thresh = 100,
    hough_acc = 200,
    hough_rad = 30,
    aspect_ratio = 1;

void sliderCallback (int a, void * arg) 
{
    cout << endl << "mode: " << mode << endl;
    cout << "canny_thresh: " << canny_thresh << endl;
    cout << "hough_acc: " << hough_acc << endl;
    cout << "hough_rad: " << hough_rad << endl;
    cout << "aspect_ratio: " << aspect_ratio << endl << endl;
}

// -- MAIN -- //
int main(int argc, char ** argv) {
    const char* filename = argc >=2 ? argv[1] : "./damas.jpg";

    // Input en color de imagen damas.jpg
    image_input = imread( samples::findFile( filename ), IMREAD_COLOR);
    if( image_input.empty()){
        cout << "Error opening image" << endl;
        return EXIT_FAILURE;
    }

    // Resize de damas
    resize(image_input, image_input, Size(512, 512));

    // Sliders
    namedWindow(WINDOW_NAME, WINDOW_AUTOSIZE );
    imshow(WINDOW_NAME, image_input);

    createTrackbar( "0: Hough - 1: Contours - 2: Centroids", 
                    WINDOW_NAME,           
                    &mode, 2,           
                    sliderCallback );
    
    createTrackbar( "Canny thresh [0-255]:", 
                    WINDOW_NAME,           
                    &canny_thresh, 255,           
                    sliderCallback );

    createTrackbar( "Hough lines accumulator [0-300]:", 
                    WINDOW_NAME,           
                    &hough_acc, 300,           
                    sliderCallback );

    createTrackbar( "Hough radius value max [0-50]:", 
                    WINDOW_NAME,           
                    &hough_rad, 50,           
                    sliderCallback );

    createTrackbar( "Aspect ratio value * 0.01 [0-4]:", 
                    WINDOW_NAME,           
                    &aspect_ratio, 4,           
                    sliderCallback );

    sliderCallback(0, 0);

    // End
    waitKey(0);
    return EXIT_SUCCESS;
}