#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ml/ml.hpp"

#include "opencv2/features2d/features2d.hpp"

#include <cstdio>
#include <iostream>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <string>

#include <Windows.h>

using namespace cv;
using namespace ml;
using namespace std;

//значения для сокращения размеров картинки
const int WIDTH_SIZE = 50; //ширина
const int HEIGHT_SIZE = 50; //высота
const int IMAGE_DATA_SIZE = WIDTH_SIZE * HEIGHT_SIZE; //суммарный размер, пиксели (далее количество нейронов на входе)

//вывод результата весов последнего слоя для отладки
void print(Mat& mat, int prec, string name)
{
    for (int i = 0; i<mat.size().height; i++)
    {
        printf("%s [ ", name);
        for (int j = 0; j<mat.size().width; j++)
        {
            printf("%.*f",prec,mat.at<float>(i, j));
            if (j != mat.size().width - 1)
                cout << ", ";
            else
                cout << " ]" << endl;
        }
    }
}

//загрузить файл картинки в матрицу
bool loadImage(string imagePath, Mat& outputImage)
{
    Mat image = imread(imagePath, IMREAD_GRAYSCALE); //грузим с цветовым пространством - оттенки серого
	 	
    Mat temp;

    // проверка на успешность
    if (image.empty()) {
        cout << "Could not open or find the image" << std::endl;
        return false;
    }

    // изменяем размер картинки до заданных
    Size size(WIDTH_SIZE, HEIGHT_SIZE);
    resize(image, temp, size, 0, 0, CV_INTER_AREA);
	
    // преобразуем цвестность пикселей из 0..255 до 0..1
    temp.convertTo(outputImage, CV_32FC1, 1.0/255.0);
	
	return true;
}

//получить перечень файлов папки в виде "подпапка\имя"
//далее "подпапка" станет типом, к которому будет вестись классификация картинок
vector<string> getFilesNamesInFolder(string folder, string type)
{
    vector<string> names;
    char search_path[200];
    sprintf(search_path, "%s/*.*", folder.c_str());   //маска поиска файлов
    WIN32_FIND_DATA fd;
    HANDLE hFind = ::FindFirstFile(search_path, &fd);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                names.push_back(type + "\\" + fd.cFileName);	//формируем вектор "подпапка \ имя"
            } 
			string fn = fd.cFileName;
			if (!(fn == "." || fn=="..")) {   //идем глубже рекурсивно
				vector<string> b = getFilesNamesInFolder(folder+"\\"+fn, fn);
				names.insert(names.end(), b.begin(), b.end());
			};
        } while (::FindNextFile(hFind, &fd));
        ::FindClose(hFind);
    }
    return names;
}

//классы, или типы картинок, имена взяты из имен подпапок, в которых они находятся
//значение - номер типа, соответствует номеру выходного нейрона при классификации
std::map<string, int> types;

//сохранить типы в файл для быстрого повторого запуска
bool save_types() {
	FILE *f = fopen("save.tp","wb");
	if (!f) {
		printf("cant save types to save.tp!\n");
		return false;
	};
	for (map<string,int>::iterator it = types.begin(); it!=types.end(); it++) { //построчная запись
		fputs(it->first.c_str(),f);  //имя
		fputs("\n",f); 
		char buf[10];
		sprintf(buf,"%d",it->second);		
		fputs(buf,f);   //номер
		fputs("\n",f);
	};
	fclose(f);
	return true;
};

//загрузить типы (классы) картинок их сохраненного файла, в формате, который описан выше
bool load_types() {
	FILE *f = fopen("save.tp","rb");
	if (!f) {
		printf("cant load types from save.tp!\n");
		return false;
	};
	char buf[1000];
	while (!feof(f)) {
		fgets(buf,200,f);     
		if (feof(f)) break;
		string a = buf;  //type имя класса
		fgets(buf,200,f); //type номер класса
		if (feof(f)) break;
		int n = atoi(buf);
		types[a] = n;		
	}
	fclose(f);
	return true;
};

//класс картинки
class pict {
public:
    Mat image;	//матрица изображения для нейросети
	string name;  //имя картинки
    string type;  //класс (тип) картинки
	int typenum;  //номер класса (номер выходного нейрона для обучения)

	//конструктор класса
    pict(Mat& image, string nam, string typ) :image(image) {
		name = nam;
		type = typ;
		if (types.find(typ) != types.end()) {
			typenum = types[typ];
		} else {
			int tn = types.size();
			types[typ] = tn;
			typenum = types[typ];
		};
    };
};

//загрузить картинки из папки
vector<pict> loadpictsFromFolder(String folderName) {
    vector<pict> roadpicts;
	vector<string> fnames = getFilesNamesInFolder(folderName, ""); //считываем имена файлов из заданной папки в виде "подпапка \ имя"

    for (int i=0;i<fnames.size();i++) { //цикл по каждому имени
		string fileName = fnames[i];
        Mat image;
        loadImage(folderName + "\\" + fileName, image); //грузим картинку
		std::size_t pos;
		if ((pos=fileName.rfind("\\")) != std::string::npos) { //если имя картинки вида "подпапка\имя", то картинка принадлежит классу
			string name = fileName;		//полное имя
			fileName.resize(pos);
			string type = fileName;   //только имя класса картинки
			roadpicts.push_back(pict(image, name, type)); //вектор pict
		} else {
			printf("cant detect type for %s\n", fileName.c_str());
		};
    }

    return roadpicts;
}


//создать вектор входных даннных 
Mat getInputDataFrompictsVector(vector<pict> roadpicts) {
    Mat roadpictsImageData;

    for (int i=0;i<roadpicts.size();i++) { //
		pict &pic = roadpicts[i];
        Mat pictImageDataInOneRow = pic.image.reshape(0, 1);  //помещаем картинку в одномерный вектор со значениями от 0 до 1
        roadpictsImageData.push_back(pictImageDataInOneRow); //добавляем в коллекцию для обучения матрицу изображения
    }

    return roadpictsImageData;
}

//построение вектора соответствия входа и выхода для обучающей выборки
Mat getOutputDataFrompictsVector(vector<pict> roadpicts) {
    Mat roadpictsData(0, types.size(), CV_32FC1);

    int i = 1;
    for (int i=0;i<roadpicts.size();i++) {
		pict &pic = roadpicts[i];
        vector<float> outputTraningVector(types.size());
        fill(outputTraningVector.begin(), outputTraningVector.end(), -1.0);  //значения выходного слоя всем нейронам = -1
        outputTraningVector[pic.typenum] = 1.0;	 // 1.0 только тому нейрону, который отвечает за класс данной картинки
		
        Mat tempMatrix(outputTraningVector, false);
        roadpictsData.push_back(tempMatrix.reshape(0, 1));   //добавляем в вектор (массив) тестовых выходных значений данную матрицу
    }

    return roadpictsData;
}

// обрезаем пробелы строк слева
static inline std::string &ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(),
            std::not1(std::ptr_fun<int, int>(std::isspace))));
    return s;
}

// обрезаем пробелы строк справа
static inline std::string &rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(),
            std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
    return s;
}

// обрезка пустых мест строк слева и справа (для канонизации имен файлов и путей с файла настроек)
static inline std::string &trim(std::string &s) {
    return ltrim(rtrim(s));
}


//основная функция программы
int main(int argc, char* argv[])
{	
	FILE *f = fopen("config.ini","r");	//считываем файл настроек
	if (!f) {
		printf("cant read config.ini");
		return 0;
	};
	char cimagebase[1024], ctestimage[1024];
	fgets(cimagebase,1024,f);  //путь к каталогу с картинками
	fgets(ctestimage,1024,f);  //путь к классифицируемому файлу
	fclose(f);
	
	std::string imagebase = cimagebase;
	std::string testimage = ctestimage;

	imagebase = trim(imagebase);
	testimage = trim(testimage); //канонизация путей
	
	if (imagebase.size()==0 || testimage.size()==0) {
		printf("no image database path or test image path in config");
		return 0;
	};	

	printf("image database folder = %s\n", imagebase.c_str());
	printf("test image = %s\n", testimage.c_str());

	f = fopen(testimage.c_str(),"rb"); //проверка доступности классифицируемого файла
	if (!f) {
		printf("error! cant load test image");
		return 0;
	};
	fclose(f);
	
	Ptr<cv::ml::ANN_MLP> mlp = ANN_MLP::create(); //создаем нейросети
	
	printf("try to load saved ANN: save.ann ...\n");
	f = fopen("save.ann","rb");   //попытка открыть сохраненный файл весов нейросети
	if (!f) {	 // если неуспешно 
		printf("save.ann not found\n");
		
		printf("\nloading training pictures...\n");
		vector<pict> roadpicts = loadpictsFromFolder(imagebase.c_str()); //загружаем обучаемую выборку
		
		if (roadpicts.size() == 0) {
			printf("no pictures loaded!");
			return 0;
		};
				
		Mat inputTrainingData = getInputDataFrompictsVector(roadpicts); //вход нейросети - загруженные картинки
		Mat outputTrainingData = getOutputDataFrompictsVector(roadpicts); //выход
		
		int hiddenLayerSize1 = IMAGE_DATA_SIZE/10;
		int hiddenLayerSize2 = sqrt(float(IMAGE_DATA_SIZE * roadpicts.size()));
				
		printf("creating ANN...\n");
		Mat layersSize = Mat(4, 1, CV_16U);   //5 слоев, один скрытый
		layersSize.row(0) = Scalar(inputTrainingData.cols); //количество нейронов первого слоя = количеству входов
		layersSize.row(1) = Scalar(hiddenLayerSize1);  //скрытый слой 1
		layersSize.row(2) = Scalar(hiddenLayerSize2);  //скрытый слой 2
		layersSize.row(3) = Scalar( types.size() );  //нейронов в выходном слое
		mlp->setLayerSizes(layersSize);
		
		printf("layers:\n");
		printf("layer0 (input) neurons: %d\n",inputTrainingData.cols);
		printf("layer1 (hidden1) neurons: %d\n",hiddenLayerSize1);
		printf("layer2 (hidden2) neurons: %d\n",hiddenLayerSize2);
		printf("layer3 (output) neurons: %d\n",types.size());
				
		mlp->setActivationFunction(ANN_MLP::SIGMOID_SYM, 1, 1);	 //функция нейрона - сигмоидная
		mlp->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 1000, 0.0001)); //критерий останова обучения 300 шагов, либо максимальное изменение весов
		mlp->setTrainMethod(ANN_MLP::BACKPROP, 0.00001);  //метод обучения: обратное распространение ошибки, коэффициент обучения 0.0001 (скорость)

		printf("training samples: %d\n", inputTrainingData.rows);
		
		Ptr<TrainData> trainingData = TrainData::create(   //создаем обучаемую выборку: входные и выходные данные
			inputTrainingData,
			SampleTypes::ROW_SAMPLE,
			outputTrainingData
		);
		
		printf("start training ANN...\n");
		
		//запуск обучения
		mlp->train(trainingData,
			/*ANN_MLP::TrainFlags::UPDATE_WEIGHTS + */
			 ANN_MLP::TrainFlags::NO_INPUT_SCALE 
			+ ANN_MLP::TrainFlags::NO_OUTPUT_SCALE
		);
		
		mlp->save("save.ann");   //сохранение результата обучения в файл для ускорения повторого запуска программы
		printf("ANN weights saved as: save.ann\n");
		
		if (!save_types()) {
			return 0;
		};

		/*
		//вывод значений выходного слоя для каждого файла обучающей выборки
		printf("trained weights:\n");
		for (int i = 0; i < inputTrainingData.rows; i++) {
			Mat result;
			mlp->predict(roadpicts[i].image.reshape(0, 1), result);
			print(result, 2, roadpicts[i].name);
		}
		*/
	} else {   //файл базы весов нейросети существует
		fclose(f);
		mlp = ANN_MLP::load<cv::ml::ANN_MLP>("save.ann");   //попытка загрузить сохраненные веса из файла в нейросеть
		printf("loaded saved ANN\n");
		
		if (load_types()) {  //грузим имена и номера классов нейросети
			printf("loaded %d types\n", types.size());
		} else {
			return 0;
		};		
	};

	printf("loading test image\n");
	
	//загрузка классифицируемого образца
	Mat result;
	Mat image;
    loadImage(testimage, image);
	printf("calculating prediction\n");
	mlp->predict(image.reshape(0, 1), result); //вычисляем значения выходного слоя
	print(result, 2, "test image ");
	
	int best_type = 0;
	float best = result.at<float>(0, 0);
	for (int i=0;i<types.size();i++) {     //поиск нейрона с максимальных выходом
		float cur = result.at<float>(0, i);
		if (cur>best) {
			best = cur;   //значение выхода
			best_type = i;  //номер нейрона
		};
	};	
	string best_t = "";
	int t = 0; //поиск имени выходного класса для найденного нейрона максимального выхода
	for (map<string,int>::iterator it = types.begin(); it!=types.end(); it++, t++) {
		if (t == best_type) {
			best_t = it->first;
			printf("best type num: %d\n", t);
			break;
		};
	};
		
	//вывод полученного классификатором значения
	printf("test image type: %s\nprobability: %f\n", best_t.c_str(), best);
	system("pause");
	return 0;
}