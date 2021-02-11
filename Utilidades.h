//
// Created by leo_mx on 10/2/21.
//

#ifndef PROYECTOFINALCV_UTILIDADES_H
#define PROYECTOFINALCV_UTILIDADES_H

#include <iostream>
#include <cstdlib>


#include <opencv2/core/core.hpp> // Contiene los elementos básicos como el objeto Mat (matriz que representa la imagen)
#include <opencv2/highgui/highgui.hpp> // Contiene los elementos para crear una interfaz gráfica básica

#include <opencv2/imgcodecs/imgcodecs.hpp> // Contiene las funcionalidad para acceder a los códecs que permiten leer diferentes formatos de imagen (JPEG, JPEG-2000, PNG, TIFF, GIF, etc.)

#include <opencv2/videoio/videoio.hpp>

#include <opencv2/imgproc/imgproc.hpp> // Librería para realizar operaciones de PDI

#include <string>
#include "Python.h"

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <array>
#include <fstream>

#include <opencv2/objdetect/objdetect.hpp> // Librería contiene funciones para detección de objetos

#include <opencv2/features2d/features2d.hpp> // Librería que contiene funciones para detección de descriptores como SIFT

#include <opencv2/xfeatures2d/nonfree.hpp>
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;


class Utilidades {
private:
    string myText;

    int calcularDiferenciasSIFT(const string& comparator, string fileClass);
    Mat cropImage(Mat img);
    void leerArchivo();

public:
    int th_val = 0, max_val = 255;
    int numverificar = 2;
    int eleccionMet = 0; // ELIMINAR
    int hMin = 10, sMin = 11, vMin = 121, hMax = 30, sMax = 100, vMax = 100;
    Mat img;
    Mat img_threshold;
    Mat img_gray;
    Mat img_gray_clon;
    Mat img_roi;
    Mat hsv;
    Mat hsvChanels[3];

    int contador;


    void initPython();
    void predecirCNN(Mat original);

    int getClase();

    static void eventoRaton(int evento, int x, int y, int flags, void *pData);

    void crearTrackbars(const string &nombre_ventana);

    int mainMethod();

};


#endif //PROYECTOFINALCV_UTILIDADES_H
