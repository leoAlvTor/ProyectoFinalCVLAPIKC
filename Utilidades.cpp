//
// Created by leo_mx on 10/2/21.
//

#include "Utilidades.h"

bool band = false;
Mat Utilidades::cropImage(Mat img){
    Mat threshold_output;
    threshold( img, threshold_output, 20, 255, THRESH_BINARY );
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours( threshold_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );
    vector<vector<Point> > contours_poly( contours.size() );
    Rect boundRect;
    double maxArea = 0.0;
    for( int i = 0; i < contours.size(); i++ )
    {
        double area = contourArea(contours[i]);
        if(area > maxArea) {
            maxArea = area;
            approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
            boundRect = boundingRect( Mat(contours_poly[i]) );
        }
    }
    return img(boundRect);
}

int Utilidades::calcularDiferenciasSIFT(const string &comparator, string fileClass) {
    Mat cmp = cropImage(imread(comparator, CV_8UC1));
    Mat file = cropImage(imread(fileClass, CV_8UC1));
    resize(cmp, cmp, Size(), 0.7, 0.7);
    resize(file, file, Size(), 0.5, 0.5);

    Ptr<SIFT> sDetector = SIFT::create();
    vector<KeyPoint> keypointsCat, keypointsfile;
    Mat descriptorCat, descriptorfile;

    sDetector->detect(cmp, keypointsCat);
    sDetector->detect(file, keypointsfile);

    sDetector->compute(cmp, keypointsCat, descriptorCat);
    sDetector->compute(file, keypointsfile, descriptorfile);

    BFMatcher matcher;
    vector<vector<DMatch> > matches;
    matcher.knnMatch(descriptorfile, descriptorCat, matches, 2);

    float ratio = 0.67;
    int matchesFiltrados = 0;
    for(int i=0;i<matches.size();i++){
        if(matches[i][0].distance < ratio*matches[i][1].distance)
            matchesFiltrados ++;
    }
    return matchesFiltrados;
}

int Utilidades::getClase() {
    string comparador = "/home/leo_mx/CLionProjects/ProyectoFinalCV/cmake-build-debug/predecir.png";

    int coincidenciaClase1 = calcularDiferenciasSIFT(comparador, "/home/leo_mx/CLionProjects/ProyectoFinalCV/cmake-build-debug/clase1.png");
    int coincidenciaClase2 = calcularDiferenciasSIFT(comparador, "/home/leo_mx/CLionProjects/ProyectoFinalCV/cmake-build-debug/clase2.png");
    int coincidenciaClase3 = calcularDiferenciasSIFT(comparador, "/home/leo_mx/CLionProjects/ProyectoFinalCV/cmake-build-debug/clase3.png");

    vector<int> values;
    values.emplace_back(coincidenciaClase1);
    values.emplace_back(coincidenciaClase2);
    values.emplace_back(coincidenciaClase3);
    int min = *max_element(values.begin(), values.end());
    for(int i = 0; i<values.size(); i++)
        if(values[i] == min)
            return i+1;
    return -1;
}

void Utilidades::leerArchivo() {
    ifstream MyReadFile("/home/leo_mx/CLionProjects/ProyectoFinalCV/cmake-build-debug/file.txt");
    while (getline (MyReadFile, myText)) {
        continue;
    }
    MyReadFile.close();
}

void Utilidades::initPython() {
    Py_Initialize();
    PyRun_SimpleString("import sys\n"
                       "if not sys.argv:\n"
                       "  sys.argv.append(\"(C++)\")"
                       "\nfrom PIL import Image\n"
                       "import numpy as np\n"
                       "from tensorflow.keras.models import model_from_json");
    PyRun_SimpleString("def cargar_modelo(archivo_modelo='/home/leo_mx/CLionProjects/ProyectoFinalCV/modelo.json', archivo_pesos='/home/leo_mx/CLionProjects/ProyectoFinalCV/pesos.h5'):\n"
                       "    with open(archivo_modelo, 'r') as f:\n"
                       "        modelo = model_from_json(f.read())\n"
                       "    \n"
                       "    modelo.load_weights(archivo_pesos)\n"
                       "    return modelo\n"
                       "modelo = cargar_modelo()\n");
}

void Utilidades::predecirCNN(Mat original) {

    resize(original, original, cv::Size(100, 100));
    imwrite("predecir.png", original);

    PyRun_SimpleString("img = Image.open('/home/leo_mx/CLionProjects/ProyectoFinalCV/cmake-build-debug/predecir.png').convert('L')\n"
                       "img = np.asarray(img)\n"
                       "img = img.reshape(-1, 100, 100, 1)\n"
                       "img = img//255\n"
                       "values = modelo.predict(img)\n"
                       "val = values.argmax(axis=-1)[0]\n"
                       "with open('file.txt', 'w') as f:\n"
                       "    f.write('%d' % val)");
    leerArchivo();
    if(stoi(myText) == 5){
        system("firefox -new-tab \"https://youtu.be/ID-iJOw9rLo\"");
    }else if(stoi(myText) == 0){
        system("cowsay hello world from C++!");
    }else{
        system("espeak \"Hello World\"");
    }

}

void Utilidades::eventoRaton(int evento, int x, int y, int flags, void *pData) {
    band = false;
    if(evento == EVENT_LBUTTONDOWN) {
        if (band) {
            band = false;
        } else {
            band = true;
        }
    }
}

void Utilidades::crearTrackbars(const string &nombre_ventana) {

    createTrackbar("TH_val",nombre_ventana,&th_val,255,NULL,NULL);
    createTrackbar("max_val",nombre_ventana,&max_val,255,NULL,NULL);
    createTrackbar("Tipo de Procesado",nombre_ventana,&numverificar,10,NULL,NULL);

    createTrackbar("H-Min",nombre_ventana,&hMin,30,NULL,NULL);
    createTrackbar("S-Min",nombre_ventana,&sMin,255,NULL,NULL);
    createTrackbar("V-Min",nombre_ventana,&vMin,255,NULL,NULL);

    createTrackbar("H-Max",nombre_ventana,&hMax,180,NULL,NULL);
    createTrackbar("S-Max",nombre_ventana,&sMax,255,NULL,NULL);
    createTrackbar("V-Max",nombre_ventana,&vMax,255,NULL,NULL);

}

int Utilidades::mainMethod() {
    band = false;
    VideoCapture cam(0);
    if (!cam.isOpened()) {
        cout << "ERROR not opened " << endl;
        return -1;
    }
    initPython();
    namedWindow("Original_image", WINDOW_AUTOSIZE);
    namedWindow("Gray_image", WINDOW_AUTOSIZE);
    namedWindow("clonGray", WINDOW_AUTOSIZE);
    namedWindow("Thresholded_image", WINDOW_AUTOSIZE);

    char a[40];
    int count = 0;
    while (1) {
        bool b = cam.read(img);
        if (!b) {
            cout << "ERROR : cannot read" << endl;
            return -1;
        }
        setMouseCallback("Thresholded_image", eventoRaton, NULL);
        if (band) {
            if (contador == 0) {
                predecirCNN(img_threshold);
                contador++;
                band = false;
            }
        } else {
            contador = 0;
        }

        img_roi = img;

        if (numverificar % 2 == 0) {
            //cout<<"Aplicando espacio de color hsv mas threshold"<<endl;
            cvtColor(img_roi, hsv, COLOR_BGR2HSV);
            split(hsv,hsvChanels);
            img_gray=hsvChanels[2];

        } else {
            //cout<<"Aplicando Escala de gis mas threshold"<<endl;
            cvtColor(img_roi, img_gray, COLOR_BGR2GRAY);
            //inRange(img_gray,Scalar(hMin,sMin,vMin),Scalar(hMax,sMax,vMax),img_gray);
            // cvtColor(img_roi, img_gray, COLOR_BGR2GRAY);
        }

        // resize(img_roi, img_roi, Size(), 0.5,0.5);


        GaussianBlur(img_gray, img_gray, Size(19, 19), 0.0, 0);
        threshold(img_gray, img_threshold,th_val , max_val, THRESH_BINARY + THRESH_OTSU);

        img_gray_clon=img_gray.clone();
        resize(img_gray_clon, img_gray_clon, Size(), 0.5,0.5);
        crearTrackbars("clonGray");

        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        findContours(img_threshold, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());

        if (!contours.empty()) {
            size_t indexOfBiggestContour = -1;
            size_t sizeOfBiggestContour = 0;

            for (size_t i = 0; i < contours.size(); i++) {
                if (contours[i].size() > sizeOfBiggestContour) {
                    sizeOfBiggestContour = contours[i].size();
                    indexOfBiggestContour = i;
                }
            }
            vector<vector<int> > hull(contours.size());
            vector<vector<Point> > hullPoint(contours.size());
            vector<vector<Vec4i> > defects(contours.size());
            vector<vector<Point> > defectPoint(contours.size());
            vector<vector<Point> > contours_poly(contours.size());
            Point2f rect_point[4];
            vector<RotatedRect> minRect(contours.size());
            vector<Rect> boundRect(contours.size());
            for (size_t i = 0; i < contours.size(); i++) {
                if (contourArea(contours[i]) > 5000) {
                    convexHull(contours[i], hull[i], true);
                    convexityDefects(contours[i], hull[i], defects[i]);
                    if (indexOfBiggestContour == i) {
                        minRect[i] = minAreaRect(contours[i]);
                        for (size_t k = 0; k < hull[i].size(); k++) {
                            int ind = hull[i][k];
                            hullPoint[i].push_back(contours[i][ind]);
                        }
                        count = 0;

                        for (size_t k = 0; k < defects[i].size(); k++) {
                            if (defects[i][k][3] > 13 * 256) {
                                /*   int p_start=defects[i][k][0];   */
                                int p_end = defects[i][k][1];
                                int p_far = defects[i][k][2];
                                defectPoint[i].push_back(contours[i][p_far]);
                                circle(img_roi, contours[i][p_end], 3, Scalar(0, 255, 0), 2);
                                count++;
                            }

                        }

                        strcpy(a, myText.c_str());

                        putText(img, a, Point(70, 70), FONT_HERSHEY_SIMPLEX, 3, Scalar(255, 0, 0), 2, 8, false);
                        //drawContours(img_threshold, contours, i, Scalar(255, 255, 0), 2, 8, vector<Vec4i>(), 0, Point());
                        //drawContours(img_threshold, hullPoint, i, Scalar(255, 255, 0), 1, 8, vector<Vec4i>(), 0, Point());
                        drawContours(img_roi, hullPoint, i, Scalar(0, 0, 255), 2, 8, vector<Vec4i>(), 0, Point());
                        approxPolyDP(contours[i], contours_poly[i], 3, false);
                        boundRect[i] = boundingRect(contours_poly[i]);
                        // rectangle(img_roi, boundRect[i].tl(), boundRect[i].br(), Scalar(255, 0, 0), 2, 8, 0);
                        minRect[i].points(rect_point);
                        for (size_t k = 0; k < 4; k++) {
                            line(img_roi, rect_point[k], rect_point[(k + 1) % 4], Scalar(0, 255, 0), 2, 8);
                        }

                    }
                }
            }


            int val = waitKey(30);
            if (val == 27) {
                return -1;
            }else if(val == 97){
                imwrite("clase1.png", img_threshold);
            }else if(val == 98){
                imwrite("clase2.png", img_threshold);
            }else if(val == 99){
                imwrite("clase3.png", img_threshold);
            }else if(val == 112){
                imwrite("predecir.png", img_threshold);
            }else if(val == 32){
                myText = "Clase: "+to_string(getClase());
            }
        }

        imshow("Original_image", img);
        imshow("Gray_image", img_gray);
        imshow("clonGray", img_gray_clon);
        imshow("Thresholded_image", img_threshold);
    }
}

