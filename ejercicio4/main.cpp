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
    CENTROIDS
};

const String WINDOW_NAME = "Practise 4";

Mat image_input,
    canny_mat,
    gray_output,
    image_output;

int mode = HOUGH,
    canny_thresh = 100,
    hough_acc = 200,
    hough_rad = 30,
    aspect_ratio = 1;

vector<Vec2f> lines;
vector<Vec3f> circles;

void print_debug_info ()
{
    cout << endl << "-------------------" << endl;
    cout << "mode: " << mode << endl;
    cout << "canny_thresh: " << canny_thresh << endl;
    cout << "hough_acc: " << hough_acc << endl;
    cout << "hough_rad: " << hough_rad << endl;
    cout << "aspect_ratio: " << aspect_ratio << endl;
    cout << "-------------------" << endl << endl;
}

void sliderCallback (int a, void * arg) 
{
    image_output = image_input.clone();

    Canny(image_input, canny_mat, canny_thresh, canny_thresh*2, 3);
    //cvtColor(canny_mat, grey_output, COLOR_GRAY2BGR);

    switch (mode)
    {
        case HOUGH:
            cout << "(0) Hough selected" << endl; 

            // Hough estándar para las líneas            
            HoughLines( canny_mat, lines, 1, CV_PI/180, hough_acc, 0, 0 ); 

            for( size_t i = 0; i < lines.size(); i++ ) {
                float rho = lines[i][0], theta = lines[i][1];
                Point pt1, pt2;
                double a = cos(theta), b = sin(theta);
                double x0 = a*rho, y0 = b*rho;
                pt1.x = cvRound(x0 + 1000*(-b));
                pt1.y = cvRound(y0 + 1000*( a));
                pt2.x = cvRound(x0 - 1000*(-b));
                pt2.y = cvRound(y0 - 1000*( a));
                line( image_output, pt1, pt2, Scalar(0,255, 0), 3, LINE_AA );
            }    

            // Hough circular
            cvtColor(image_input, gray_output, COLOR_BGR2GRAY);
            medianBlur(gray_output, gray_output, 5);

            HoughCircles(gray_output, circles, HOUGH_GRADIENT, 1,
                         gray_output.rows/16,
                         100, 30, 1, hough_rad);

            for( size_t i = 0; i < circles.size(); i++ ) {
                Vec3i c = circles[i];
                Point center = Point(c[0], c[1]);
                // circle center
                circle( image_output, center, 1, Scalar(0,100,100), 3, LINE_AA);
                // circle outline
                int radius = c[2];
                circle( image_output, center, radius, Scalar(0,0,0), 3, LINE_AA);
            }
    
            break;
        
        case CONTOURS:
            cout << "(1) Contours selected" << endl;
            break;
        
        case CENTROIDS:
            cout << "(2) Centroids selected" << endl;
            break;
    }

    print_debug_info ();
    imshow(WINDOW_NAME, image_output);
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

    image_output = image_input.clone();

    // Sliders
    namedWindow(WINDOW_NAME, WINDOW_AUTOSIZE );
    imshow(WINDOW_NAME, image_output);

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