/* ------------------------------------------------------------ */
// Programador: Cristian Sánchez Rodríguez
//
// Asignatura: Visión Artificial (Ejercicio 5)
//
// Descripción: En este programa, se busca obtener el esqueleto de una  
//				imagen  binarizada. Se mostrará el proceso durante el
//              display.
/* ------------------------------------------------------------ */

// -- Includes -- //
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <iomanip>

using namespace cv;
using namespace std;

// -- Constantes y variables globales -- //
enum element_type
{
    RECT = 0,
    CROSS,
    ELLIPSE
};

const String WINDOW_NAME = "Skeleton Demo";
int counter = 0;    // Contador de iteraciones

// Matrices empleadas para procesar
Mat image_input,
    skeleton_mat,
    open_mat,
    image_output;

// Variables de los sliders (inicializadas)
int iterations = 60,
    element = CROSS,
    kernel_size = 1;

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
    image_output = image_input.clone();
    bool done = false;                // Condición del loop para saber si han finalizado las iteraciones    
    Mat tresh = image_input.clone();

    // Aplico un threshold para obtener resultados más consistentes.
    // En función del valor del thresh, interpreta el esqueleto mejor o peor.
    threshold( image_input, 
               tresh, 
               127, 
               255,  
               THRESH_BINARY);

    Mat struct_element = getStructuringElement( element, 
                                                Size( 2*kernel_size + 1, 2*kernel_size + 1 ), 
                                                Point( kernel_size, kernel_size ) );

    Mat eroded, temp;
    Mat skel = Mat::zeros(tresh.size(), CV_8UC1);

    while ((!done) && (iterations != 0)) {
        // Open
        morphologyEx( tresh, 
                      eroded, 
                      MORPH_ERODE, 
                      struct_element );

        morphologyEx( eroded, 
                      temp, 
                      MORPH_DILATE, 
                      struct_element );

        //Subtract
        subtract(tresh, temp, temp);
        // Union
        bitwise_or(skel, temp, skel);
        // Redefinir erosionada
        eroded.copyTo(tresh);
        
        if (counter == iterations-1)
            done = true;

        counter++;
        imshow(WINDOW_NAME, skel); 
        waitKey(50);
    };
    waitKey(1000);
    counter = 0;
    
    // Display, se pinta en rojo los píxeles de skel en la imagen original
    cvtColor(image_output, image_output, COLOR_GRAY2BGR);
    vector<Mat> out_channels, channels;
    split( image_output, out_channels );

    for (int row = 0; row < skel.rows; row++){
        for (int col = 0; col < skel.cols; col++){
            if ((uint)skel.at<uchar>(row, col) == 255){
                    out_channels[0].at<uchar>(row, col) = 0;
                    out_channels[1].at<uchar>(row, col) = 0;
                    out_channels[2].at<uchar>(row, col) = 255;
            }
        }
    }
    
    channels.push_back(out_channels[0]);
    channels.push_back(out_channels[1]);
    channels.push_back(out_channels[2]);

    merge(channels, image_output);
    imshow(WINDOW_NAME, image_output);
}

// -- MAIN -- //
int main(int argc, char ** argv) {
    const char* filename = argc >=2 ? argv[1] : "./model.png";

    // Input
    image_input = imread( samples::findFile( filename ), IMREAD_GRAYSCALE);
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