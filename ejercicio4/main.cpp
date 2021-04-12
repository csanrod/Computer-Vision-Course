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
#include <cmath>
#include <iostream>
#include <iomanip>

using namespace cv;
using namespace std;

// -- Constantes y variables globales -- //
enum slider_mode 
{
    HOUGH = 0,
    CONTOURS,
    CENTROIDS
};

const String WINDOW_NAME = "Practise 4";

// Matrices empleadas para procesar
Mat image_input,
    canny_mat,
    gray_output,
    gauss_mat,
    image_output;

// Variables de los sliders
int mode = HOUGH,
    canny_thresh = 100,
    hough_acc = 200,
    hough_rad = 30,
    aspect_ratio = 1;

RNG rng(12345);

// Método de trazado para depurar sliders
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

// Método para obtener y pintar la transformada de hough (lineal y circular)
void aply_hough ()
{
    // -- Hough lineal -- //
    vector<Vec2f> lines;            
    HoughLines( canny_mat, lines, 1, CV_PI/180, hough_acc, 0, 0 ); 

    // Pintado de líneas
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

    // -- Hough circular -- //
    cvtColor(image_input, gray_output, COLOR_BGR2GRAY);
    medianBlur(gray_output, gray_output, 5); // (Opcional) actúa sobre el ruido.

    vector<Vec3f> circles;
    HoughCircles(gray_output, circles, HOUGH_GRADIENT, 1,
                    gray_output.rows/16,
                    100, 30, 1, hough_rad);

    // Pintado de círculos
    for( size_t i = 0; i < circles.size(); i++ ) {
        Vec3i c = circles[i];
        Point center = Point(c[0], c[1]);
        circle( image_output, center, 1, Scalar(0,100,100), 3, LINE_AA);
        int radius = c[2];
        circle( image_output, center, radius, Scalar(0,0,0), 3, LINE_AA);
    } 
}

// Método para obtener y pintar contornos
void aply_contours ()
{
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours( canny_mat, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE );

    vector<Rect> boundRect( contours.size() );
    for (int i = 0; i < contours.size(); i++)    
        boundRect[i] = boundingRect( contours[i] );

    // Pintado de contornos
    float relation;
    for( size_t i = 0; i< contours.size(); i++ ) {
        relation = float(boundRect[i].width)/boundRect[i].height;
        if ((abs(1 - relation)) < (aspect_ratio*0.01)) {
            Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
            drawContours( image_output, contours, (int)i, color,  2, LINE_8, hierarchy, 1  );
        }
    }
}

// Método para obtener y pintar centroides
void aply_centroids ()
{
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours( canny_mat, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE );

    vector<Moments> M(contours.size());
    for (int i = 0; i < contours.size(); i++)
        M[i] = moments(contours[i]);
    
    // Pintar centroides de radio 4
    double Cx, Cy;
    for (int i = 0; i < contours.size(); i++){
        Cx = M[i].m10/M[i].m00;
        Cy = M[i].m01/M[i].m00;
        Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
        Point centroid = Point(Cx, Cy);
        circle( image_output, centroid, 4, color, 3, LINE_AA);
    }

    // Pintar puntos Hough circular de radio 6
    vector<Vec3f> circles;
    HoughCircles(gray_output, circles, HOUGH_GRADIENT, 1,
                    gray_output.rows/16,
                    100, 30, 1, hough_rad);

    for( size_t i = 0; i < circles.size(); i++ ) {
        Vec3i c = circles[i];
        Point center = Point(c[0], c[1]);
        // circle center
        circle( image_output, center, 6, Scalar(255,0,0), 3, LINE_AA);
    }   

    // Pintar puntos que disten menos de 4 con radio 10
    for( size_t i = 0; i < circles.size(); i++ ) {
        Vec3i c = circles[i];
        Point center = Point(c[0], c[1]);
        for (int i = 0; i < contours.size(); i++){
            Cx = M[i].m10/M[i].m00;
            Cy = M[i].m01/M[i].m00;
            if ((abs(Cx - c[0]) < 4)&&(abs(Cy - c[1]) < 4)){
                circle( image_output, center, 10, Scalar(0,255,0), 3, LINE_AA);
                break;
            }
        }
    }
}

// Método Callback
void sliderCallback (int a, void * arg) 
{
    image_output = image_input.clone();
    cvtColor( image_input, gray_output, COLOR_BGR2GRAY );
    Canny(gray_output, canny_mat, canny_thresh, canny_thresh*2, 3);

    switch (mode)
    {
        case HOUGH:
        {
            aply_hough();
            break;
        }

        case CONTOURS:
        {
            aply_contours();
            break;
        }

        case CENTROIDS:
        {
            aply_centroids();
            break;
        }
    }

    // print_debug_info ();
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
    waitKey(0);
    return EXIT_SUCCESS;
}