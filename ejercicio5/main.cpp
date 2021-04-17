/* ------------------------------------------------------------ */
// Programador: Cristian Sánchez Rodríguez
//
// Asignatura: Visión Artificial (Ejercicio 5)
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
#include <cmath>
#include <iostream>
#include <iomanip>

using namespace cv;
using namespace std;

// -- Constantes y variables globales -- //
enum slider_mode 
{
    RECT = 0,
    CROSS,
    ELLIPSE
};

const String WINDOW_NAME = "Skeleton Demo";

// Matrices empleadas para procesar
Mat image_input,
    image_output;

// Variables de los sliders
int iterations = 60,
    element = RECT,
    kernel_size = 1;

RNG rng(12345);

// Método de trazado para depurar sliders
void print_debug_info ()
{
    cout << endl << "-------------------" << endl;
    cout << "Iterations: " << iterations << endl;
    cout << "Element: " << element << endl;
    cout << "Kernel size: " << kernel_size << endl;
    cout << "-------------------" << endl << endl;
}

// Método Callback
void sliderCallback (int a, void * arg) 
{
    switch (element)
    {
        case RECT:
        {
            cout << "RECT selected" << endl;
            break;
        }

        case CROSS:
        {
            cout << "CROSS selected" << endl;
            break;
        }

        case ELLIPSE:
        {
            cout << "ELLIPSE selected" << endl;
            break;
        }
    }

    print_debug_info ();
    imshow(WINDOW_NAME, image_output);
}

// -- MAIN -- //
int main(int argc, char ** argv) {
    const char* filename = argc >=2 ? argv[1] : "./model.png";

    // Input en color de imagen damas.jpg
    image_input = imread( samples::findFile( filename ), IMREAD_COLOR);
    if( image_input.empty()){
        cout << "Error opening image" << endl;
        return EXIT_FAILURE;
    }

    // Resize
    resize(image_input, image_input, Size(512, 512));
    image_output = image_input.clone();

    // Sliders
    namedWindow(WINDOW_NAME, WINDOW_AUTOSIZE );
    imshow(WINDOW_NAME, image_output);

    createTrackbar( "Iterations:\n0-100", 
                    WINDOW_NAME,           
                    &iterations, 100,           
                    sliderCallback );
    
    createTrackbar( "Element:\n0: Rect - 1:Cross - 2:Ellipse", 
                    WINDOW_NAME,           
                    &element, 2,           
                    sliderCallback );

    createTrackbar( "Kernel size:\n2n + 1:", 
                    WINDOW_NAME,           
                    &kernel_size, 5,           
                    sliderCallback );

    sliderCallback(0, 0);
    waitKey(0);
    return EXIT_SUCCESS;
}