#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <iostream>
#include <fstream>
#include <cassert>
#include <vector>
#include <cmath>
#include <string>
#include <random>
#include <sstream>
#include <array>
#include <stdio.h>
#include <unistd.h>

#define _NB_LAYERS_ 3 													//counting input and output layer
#define _NB_INPUTS_ 4
#define _NB_NEURONS1_ 500
#define _NB_OUTPUTS_ 3

typedef double Neuron;
typedef std::vector<double> Layer;
typedef std::vector<std::vector<double> > Matrice;
typedef std::array<std::vector<double>, _NB_LAYERS_> MatriceFixe;
typedef std::pair<std::vector<double>, double> AllInfos;

static std::random_device rd;
static std::mt19937 gen(rd());

class Utilities{
public:
	MatriceFixe getActivations()  ;
	Matrice getCorrectOutput()  ;
	double sigmoid(double valeur)  ;
	double deriveeSigmoid(double valeur)  ;
	bool checkActivationBounds(double valeur)  ;
	double flowerTypeToDouble(std::string type1)  ;
	int newInt(std::vector<int>& deja_tires, int data_size)  ;
	double sum(const  std::vector<double>& tab1)  ;
	std::vector<double> reArrangeVect(int indexLayer, int indexTargetNeur)  ;
	Layer vectSum(const Layer& tab1,const Layer& tab2) ;
	Layer vectSub(const Layer& tab1,const Layer& tab2) ;
		
	Layer prodElement(const Layer& tab1, const Layer& tab2) ;
	double prodScal(const Layer& tab1, const Layer& tab2) ;
	void afficheVect(const Layer& tab) ;	
	void displayLoadingBar(int i);
	void afficheWeights()  ;
};

	


#endif

