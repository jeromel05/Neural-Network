#ifndef NETWORK_HPP
#define NETWORK_HPP

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

#define  _NB_LAYERS_ 3 													//counting input and output layer
#define _NB_INPUTS_ 4
#define _NB_NEURONS1_ 500
#define _NB_OUTPUTS_ 3

typedef double Neuron;
typedef std::vector<double> Layer;
typedef std::array<double, _NB_INPUTS_> Layer0;
typedef std::array<double, _NB_NEURONS1_> Layer1;
typedef std::array<double, _NB_OUTPUTS_> Layer2;
typedef std::vector<std::vector<double>> Matrice;
typedef std::array<std::vector<double>, _NB_LAYERS_> MatriceFixe;
typedef std::pair<std::vector<double>, double> AllInfos;

static std::random_device rd;
static std::mt19937 gen(rd());


class Network
{
private:
	double eta_;	// learning rate
	std::array<Matrice, _NB_LAYERS_ - 1> weights_;
	MatriceFixe neurons_;
	unsigned int iterations_tot_;
	Matrice correctOutputs_;
	MatriceFixe deltas_;		
	std::vector<AllInfos> wholeData_;
	double delta_final_scal_;
	int data_size_;
	Matrice bias_;
	
public:
	//---------Utilities--------------
	MatriceFixe getActivations() const;
	Matrice getCorrectOutput()  const;
	double sigmoid(double valeur) const ;
	double deriveeSigmoid(double valeur) const ;
	bool checkActivationBounds(double valeur) const;
	double flowerTypeToDouble(const std::string& type1) const ;
	int newInt(std::vector<int>& deja_tires) const ;
	double sum(const  std::vector<double>& tab1)  const;
	Layer vectSum(const Layer& tab1,const Layer& tab2) const;
	Layer vectSub(const Layer& tab1,const Layer& tab2) const;
	Layer prodElement(const Layer& tab1, const Layer& tab2) const;
	double prodScal(const Layer& tab1, const Layer& tab2) const;
	void afficheVect(const Layer& tab) const;	
	void displayLoadingBar(int i) const;
	
	void calculate_delta_final_scal();
	Layer reArrangeVect(int indexLayer, int indexTargetNeur) const;
	void writeWeights(std::vector<std::ofstream>& out, int step) const;
	void writeSingleWeight(std::ostream& out, int layer, int step) const;
	
	void buildRandomWeights();
	void buildRandomBiases();
	void readWholeInput(std::ifstream& inputFile);
	Layer computeError(std::ifstream& fichier) const;
	
	void run();
	void update(std::ofstream& errorsFile, std::vector<std::ofstream>& weightFiles, int step, std::vector<int>& deja_tires);
	void activateLayer(int index);
	
	//-----BackPropagation-----------
	void calculateDeltas(int randomStep);
	void deltaLayer(int index);
	void updateWeights();
	void updateBiases();


	Network(unsigned int iterations_tot = 100, double learningRate = 0.5);
	~Network();
};

#endif



